"""
shared_state.py — Shared state bridge between voice agent and Streamlit dashboard.

Backend: Google Cloud Firestore (replaces local state.json for Cloud Run compatibility).
The voice agent (local) and the dashboard (Cloud Run) both read/write the same
Firestore document — no filesystem dependency.

Collection : litterman
Document   : state

Schema (identical to v1 state.json):
{
    "portfolio": {
        "current":     {"Stocks_USA": 0.60, "Stocks_EM": 0.30, "Bonds_USA": 0.10},
        "recommended": {"Stocks_USA": 0.53, "Stocks_EM": 0.17, "Bonds_USA": 0.30}
    },
    "views": [...],
    "sharpe_ratio": -0.18,
    "events": [...],
    "status": "idle",
    "last_updated": "2026-02-24T15:32:05"
}

Environment variables required:
    GOOGLE_CLOUD_PROJECT  — GCP project ID (e.g. "litterman-ai")
    GOOGLE_APPLICATION_CREDENTIALS — path to service account JSON (local only)
                                     Not needed on Cloud Run (uses ADC automatically)
"""

import os
import copy
from datetime import datetime
from google.cloud import firestore

# ── Firestore client ──────────────────────────────────────────────────────────

_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
_db = firestore.Client(project=_PROJECT)
_DOC_REF = _db.collection("litterman").document("state")

# ── Constants ─────────────────────────────────────────────────────────────────

INITIAL_WEIGHTS = {
    "Stocks_USA": 0.60,
    "Stocks_EM": 0.30,
    "Bonds_USA": 0.10,
}

_DEFAULT_STATE = {
    "portfolio": {
        "current": copy.deepcopy(INITIAL_WEIGHTS),
        "recommended": None,
    },
    "views": [],
    "sharpe_ratio": None,
    "events": [],
    "status": "idle",
    "last_updated": None,
}

# ── Internal helpers ──────────────────────────────────────────────────────────

def _load() -> dict:
    """
    Reads current state from Firestore.
    Returns default state if document does not exist yet.
    """
    doc = _DOC_REF.get()
    if doc.exists:
        return doc.to_dict()
    return copy.deepcopy(_DEFAULT_STATE)


def _save(state: dict) -> None:
    """
    Writes full state to Firestore.
    Uses set() with merge=False to replace the entire document atomically.
    """
    state["last_updated"] = datetime.now().isoformat(timespec="seconds")
    _DOC_REF.set(state)


# ── Public API (identical to v1) ──────────────────────────────────────────────

def get_state() -> dict:
    """Returns the current full state. Safe to call from dashboard (read-only)."""
    return _load()


def set_status(status: str) -> None:
    """
    Update agent status: 'idle' | 'listening' | 'processing' | 'speaking'

    Uses Firestore update() instead of full set() — only touches the status
    field, avoids overwriting concurrent writes from the voice agent.
    """
    _DOC_REF.set(
        {
            "status": status,
            "last_updated": datetime.now().isoformat(timespec="seconds"),
        },
        merge=True,   # patch — do not overwrite other fields
    )


def push_bl_result(
    transcript: str,
    views: list,
    weights_after: dict,
    sharpe_after: float,
) -> None:
    """
    Called by voice agent after a successful BL run.

    - Saves the new recommended weights (current weights stay unchanged until
      the manager manually confirms a rebalance).
    - Appends event to history (last 20 kept).
    - Updates views and Sharpe.
    """
    state = _load()

    weights_before = state["portfolio"]["current"].copy()

    event = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "transcript": transcript[:300],
        "sharpe_before": state.get("sharpe_ratio"),
        "sharpe_after": round(sharpe_after, 4),
        "weights_before": weights_before,
        "weights_after": {k: round(v, 4) for k, v in weights_after.items()},
    }
    state["events"] = ([event] + state["events"])[:20]

    state["portfolio"]["recommended"] = {k: round(v, 4) for k, v in weights_after.items()}
    state["views"] = views
    state["sharpe_ratio"] = round(sharpe_after, 4)
    state["status"] = "idle"

    _save(state)


def confirm_rebalance() -> None:
    """
    Moves recommended weights into current weights.
    Called when manager confirms the rebalance (dashboard button).
    """
    state = _load()
    if state["portfolio"].get("recommended"):
        state["portfolio"]["current"] = state["portfolio"]["recommended"].copy()
        state["portfolio"]["recommended"] = None
    _save(state)


def reset_state() -> None:
    """Resets to default state. Useful for testing and demo resets."""
    _save(copy.deepcopy(_DEFAULT_STATE))


# ── Manual test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    import pprint

    print("Resetting state...")
    reset_state()
    print("Done.\n")

    print("Pushing Event 1 — Fed rate hike (bearish)...")
    push_bl_result(
        transcript="The Federal Reserve signaled today that it will keep interest rates higher for longer, citing persistent inflation concerns. Markets reacted with US equities falling sharply while emerging markets showed resilience.",
        views=[
            {"description": "Bonds sell off on hawkish Fed guidance", "confidence": 0.80,
             "type": "absolute", "asset": "Bonds_USA", "expected_return": -0.03},
            {"description": "US equities slightly negative on tighter policy", "confidence": 0.65,
             "type": "absolute", "asset": "Stocks_USA", "expected_return": -0.04},
            {"description": "EM equities resilient vs US on relative basis", "confidence": 0.50,
             "type": "absolute", "asset": "Stocks_EM", "expected_return": -0.02},
        ],
        weights_after={"Stocks_USA": 0.53, "Stocks_EM": 0.17, "Bonds_USA": 0.30},
        sharpe_after=-0.18,
    )
    print("Event 1 pushed.\n")

    time.sleep(1)

    print("Pushing Event 2 — Strong jobs report (bullish)...")
    push_bl_result(
        transcript="US non-farm payrolls came in at 350k, well above the 200k consensus. Unemployment fell to 3.7%. Equity futures rallied strongly in pre-market trading.",
        views=[
            {"description": "Strong labour market boosts US equities", "confidence": 0.75,
             "type": "absolute", "asset": "Stocks_USA", "expected_return": 0.06},
            {"description": "EM benefits from positive global risk sentiment", "confidence": 0.55,
             "type": "absolute", "asset": "Stocks_EM", "expected_return": 0.04},
        ],
        weights_after={"Stocks_USA": 0.68, "Stocks_EM": 0.26, "Bonds_USA": 0.06},
        sharpe_after=0.42,
    )
    print("Event 2 pushed.\n")

    print("Final state:")
    pprint.pprint(get_state())