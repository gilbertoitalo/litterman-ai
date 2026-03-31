"""
state_db.py — Camada de persistência local para o Litterman.ai MVP.

Substitui o Firestore (shared_state.py do hackathon) por SQLite.
Sem dependências de cloud — funciona completamente offline.

Schema:
    tabela: sessions
        id          INTEGER PRIMARY KEY AUTOINCREMENT
        timestamp   TEXT    ISO-8601, gerado automaticamente
        transcript  TEXT    texto da notícia/input que gerou o evento
        views_json  TEXT    JSON serializado da lista de views BL
        weights_before_json TEXT  pesos antes do rebalanceamento
        weights_after_json  TEXT  pesos recomendados pelo BL
        sharpe      REAL    Sharpe ratio resultante
        status      TEXT    'pending' | 'confirmed' | 'rejected'

    tabela: portfolio
        id          INTEGER PRIMARY KEY (sempre 1 — singleton)
        weights_json TEXT   pesos actuais confirmados
        updated_at  TEXT    ISO-8601 do último confirm_rebalance

Filosofia de design:
    - Portfolio singleton (id=1) → pesos actuais confirmados pelo manager
    - Sessions → histórico imutável de todos os eventos BL
    - confirm_rebalance() → move weights_after → portfolio e marca session como 'confirmed'
    - reset_state() → restaura pesos iniciais e apaga histórico (útil para demos)
    - API pública idêntica ao shared_state.py — drop-in replacement
"""

import json
import sqlite3
import copy
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

# ── Configuração ──────────────────────────────────────────────────────────────

DB_PATH = Path(__file__).parent.parent / "data" / "litterman.db"

INITIAL_WEIGHTS = {
    "Stocks_USA": 0.60,
    "Stocks_EM":  0.30,
    "Bonds_USA":  0.10,
}

# ── Helpers internos ──────────────────────────────────────────────────────────

@contextmanager
def _get_conn():
    """Context manager que abre, commita/rollback e fecha a conexão SQLite."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row          # acesso por nome de coluna
    conn.execute("PRAGMA journal_mode=WAL") # write-ahead log: melhor concorrência
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _init_db() -> None:
    """Cria as tabelas se não existirem e inicializa o portfolio singleton."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp           TEXT    NOT NULL,
                transcript          TEXT    NOT NULL DEFAULT '',
                views_json          TEXT    NOT NULL DEFAULT '[]',
                weights_before_json TEXT    NOT NULL DEFAULT '{}',
                weights_after_json  TEXT    NOT NULL DEFAULT '{}',
                sharpe              REAL,
                status              TEXT    NOT NULL DEFAULT 'pending'
            );

            CREATE TABLE IF NOT EXISTS portfolio (
                id           INTEGER PRIMARY KEY,
                weights_json TEXT NOT NULL,
                updated_at   TEXT NOT NULL
            );
        """)

        # Garante que o singleton existe (INSERT OR IGNORE)
        conn.execute("""
            INSERT OR IGNORE INTO portfolio (id, weights_json, updated_at)
            VALUES (1, ?, ?)
        """, (
            json.dumps(INITIAL_WEIGHTS),
            datetime.now().isoformat(timespec="seconds"),
        ))


# Inicializa na importação (idempotente)
_init_db()


# ── API pública ───────────────────────────────────────────────────────────────

def get_current_weights() -> dict:
    """
    Devolve os pesos actuais confirmados do portfólio.

    Equivalente a get_state()['portfolio']['current'] no Firestore.
    """
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT weights_json FROM portfolio WHERE id = 1"
        ).fetchone()
        return json.loads(row["weights_json"])


def get_latest_session() -> dict | None:
    """
    Devolve a sessão mais recente (último evento BL), ou None se não existir.

    Inclui campos: id, timestamp, views, weights_before, weights_after, sharpe, status.
    """
    with _get_conn() as conn:
        row = conn.execute("""
            SELECT * FROM sessions
            ORDER BY id DESC
            LIMIT 1
        """).fetchone()

        if row is None:
            return None

        return _row_to_session(row)


def get_state() -> dict:
    """
    Devolve o estado completo do sistema.

    Estrutura idêntica ao shared_state.py do hackathon para compatibilidade:
    {
        'portfolio': {
            'current':     {Stocks_USA: 0.60, ...},
            'recommended': {Stocks_USA: 0.55, ...} | None
        },
        'views':       [...],
        'sharpe_ratio': float | None,
        'events':      [...],   # últimos 20
        'status':      'idle' | 'processing' | 'speaking',
        'last_updated': str | None
    }
    """
    current = get_current_weights()
    latest = get_latest_session()

    recommended = None
    views = []
    sharpe = None
    last_updated = None

    if latest and latest["status"] == "pending":
        recommended = latest["weights_after"]
        views = latest["views"]
        sharpe = latest["sharpe"]
        last_updated = latest["timestamp"]

    events = get_recent_events(limit=20)

    return {
        "portfolio": {
            "current":     current,
            "recommended": recommended,
        },
        "views":       views,
        "sharpe_ratio": sharpe,
        "events":      events,
        "status":      "idle",       # MVP: sem voice agent → sempre idle
        "last_updated": last_updated,
    }


def push_bl_result(
    transcript: str,
    views: list,
    weights_after: dict,
    sharpe_after: float,
) -> int:
    """
    Regista um novo evento BL como sessão 'pending'.

    Equivalente ao push_bl_result() do Firestore.
    Devolve o id da sessão criada.
    """
    weights_before = get_current_weights()

    with _get_conn() as conn:
        cursor = conn.execute("""
            INSERT INTO sessions
                (timestamp, transcript, views_json,
                 weights_before_json, weights_after_json, sharpe, status)
            VALUES (?, ?, ?, ?, ?, ?, 'pending')
        """, (
            datetime.now().isoformat(timespec="seconds"),
            transcript[:500],
            json.dumps(views),
            json.dumps({k: round(v, 4) for k, v in weights_before.items()}),
            json.dumps({k: round(v, 4) for k, v in weights_after.items()}),
            round(sharpe_after, 4),
        ))
        return cursor.lastrowid


def confirm_rebalance(session_id: int | None = None) -> bool:
    """
    Confirma o rebalanceamento:
        1. Copia weights_after → portfolio.weights_json
        2. Marca a sessão como 'confirmed'

    Se session_id=None, confirma a sessão 'pending' mais recente.
    Devolve True se confirmou, False se não havia nada pendente.
    """
    with _get_conn() as conn:
        if session_id is None:
            row = conn.execute("""
                SELECT id, weights_after_json FROM sessions
                WHERE status = 'pending'
                ORDER BY id DESC LIMIT 1
            """).fetchone()
        else:
            row = conn.execute("""
                SELECT id, weights_after_json FROM sessions
                WHERE id = ? AND status = 'pending'
            """, (session_id,)).fetchone()

        if row is None:
            return False

        conn.execute("""
            UPDATE portfolio
            SET weights_json = ?, updated_at = ?
            WHERE id = 1
        """, (
            row["weights_after_json"],
            datetime.now().isoformat(timespec="seconds"),
        ))

        conn.execute("""
            UPDATE sessions SET status = 'confirmed' WHERE id = ?
        """, (row["id"],))

        return True


def reject_rebalance(session_id: int | None = None) -> bool:
    """
    Rejeita o rebalanceamento pendente (manager decidiu não agir).
    Marca a sessão como 'rejected'. Pesos actuais não são alterados.
    Devolve True se rejeitou, False se não havia nada pendente.
    """
    with _get_conn() as conn:
        if session_id is None:
            row = conn.execute("""
                SELECT id FROM sessions
                WHERE status = 'pending'
                ORDER BY id DESC LIMIT 1
            """).fetchone()
        else:
            row = conn.execute("""
                SELECT id FROM sessions WHERE id = ? AND status = 'pending'
            """, (session_id,)).fetchone()

        if row is None:
            return False

        conn.execute("""
            UPDATE sessions SET status = 'rejected' WHERE id = ?
        """, (row["id"],))

        return True


def get_recent_events(limit: int = 20) -> list:
    """
    Devolve os últimos N eventos BL, do mais recente para o mais antigo.
    Formato compatível com o campo 'events' do get_state() do Firestore.
    """
    with _get_conn() as conn:
        rows = conn.execute("""
            SELECT * FROM sessions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

        return [_row_to_session(r) for r in rows]


def reset_state() -> None:
    """
    Reset completo: apaga todo o histórico e restaura pesos iniciais.
    Útil para demos e testes.
    """
    with _get_conn() as conn:
        conn.execute("DELETE FROM sessions")
        conn.execute("""
            UPDATE portfolio
            SET weights_json = ?, updated_at = ?
            WHERE id = 1
        """, (
            json.dumps(INITIAL_WEIGHTS),
            datetime.now().isoformat(timespec="seconds"),
        ))


# ── Helpers de serialização ───────────────────────────────────────────────────

def _row_to_session(row: sqlite3.Row) -> dict:
    """Converte uma Row SQLite num dict limpo para consumo externo."""
    return {
        "id":             row["id"],
        "timestamp":      row["timestamp"],
        "transcript":     row["transcript"],
        "views":          json.loads(row["views_json"]),
        "weights_before": json.loads(row["weights_before_json"]),
        "weights_after":  json.loads(row["weights_after_json"]),
        "sharpe":         row["sharpe"],
        "status":         row["status"],
    }


# ── Teste manual ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint

    print("=== Reset state ===")
    reset_state()

    print("\n=== Estado inicial ===")
    pprint.pprint(get_state())

    print("\n=== Push Event 1 — Fed hawkish ===")
    sid1 = push_bl_result(
        transcript="Federal Reserve signaled rates higher for longer amid persistent inflation.",
        views=[
            {"description": "Bonds sell off on hawkish Fed", "confidence": 0.80,
             "type": "absolute", "asset": "Bonds_USA", "expected_return": -0.03},
            {"description": "US equities slightly negative on tighter policy", "confidence": 0.65,
             "type": "absolute", "asset": "Stocks_USA", "expected_return": -0.04},
        ],
        weights_after={"Stocks_USA": 0.53, "Stocks_EM": 0.37, "Bonds_USA": 0.10},
        sharpe_after=-0.18,
    )
    print(f"Session id: {sid1}")

    print("\n=== Estado após push (recommended != None) ===")
    state = get_state()
    print("current:    ", state["portfolio"]["current"])
    print("recommended:", state["portfolio"]["recommended"])
    print("sharpe:     ", state["sharpe_ratio"])

    print("\n=== Confirm rebalance ===")
    ok = confirm_rebalance()
    print(f"Confirmed: {ok}")

    print("\n=== Estado após confirm (current atualizado) ===")
    state = get_state()
    print("current:    ", state["portfolio"]["current"])
    print("recommended:", state["portfolio"]["recommended"])

    print("\n=== Push Event 2 — Strong jobs report ===")
    sid2 = push_bl_result(
        transcript="Non-farm payrolls beat expectations at 350k. Unemployment fell to 3.7%.",
        views=[
            {"description": "Strong labour market boosts US equities", "confidence": 0.75,
             "type": "absolute", "asset": "Stocks_USA", "expected_return": 0.06},
        ],
        weights_after={"Stocks_USA": 0.68, "Stocks_EM": 0.26, "Bonds_USA": 0.06},
        sharpe_after=0.42,
    )

    print("\n=== Reject rebalance ===")
    ok = reject_rebalance()
    print(f"Rejected: {ok}")

    print("\n=== Histórico (últimos 5) ===")
    for ev in get_recent_events(limit=5):
        print(f"  [{ev['id']}] {ev['timestamp']} | {ev['status']} | sharpe={ev['sharpe']}")
