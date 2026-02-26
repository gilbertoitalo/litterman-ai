"""
shared_state.py — Shared state bridge between voice agent and Streamlit dashboard.

Voice agent writes to STATE_FILE after each BL event.
Dashboard polls STATE_FILE every 2 seconds and re-renders.

Schema:
{
    "portfolio": {
        "current":     {"Stocks_USA": 0.60, "Stocks_EM": 0.30, "Bonds_USA": 0.10},
        "recommended": {"Stocks_USA": 0.53, "Stocks_EM": 0.17, "Bonds_USA": 0.30}
    },
    "views": [
        {
            "description": "...",
            "type": "absolute",
            "asset": "Bonds_USA",
            "expected_return": -0.03,
            "confidence": 0.75
        }
    ],
    "sharpe_ratio": -0.18,
    "events": [
        {
            "timestamp": "2026-02-24T15:32:01",
            "transcript": "The Fed signaled rates higher for longer...",
            "sharpe_before": null,
            "sharpe_after": -0.18,
            "weights_before": {"Stocks_USA": 0.60, ...},
            "weights_after":  {"Stocks_USA": 0.53, ...}
        }
    ],
    "status": "idle",          // "idle" | "listening" | "processing" | "speaking"
    "last_updated": "2026-02-24T15:32:05"
}
"""

import json
import os
from datetime import datetime
from pathlib import Path

STATE_FILE = Path(__file__).parent / "state.json"

INITIAL_WEIGHTS = {
    "Stocks_USA": 0.60,
    "Stocks_EM": 0.30,
    "Bonds_USA": 0.10,
}

_DEFAULT_STATE = {
    "portfolio": {
        "current": INITIAL_WEIGHTS.copy(),
        "recommended": None,
    },
    "views": [],
    "sharpe_ratio": None,
    "events": [],
    "status": "idle",
    "last_updated": None,
}


def _load() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return json.loads(json.dumps(_DEFAULT_STATE))  # deep copy


def _save(state: dict) -> None:
    state["last_updated"] = datetime.now().isoformat(timespec="seconds")
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── Public API ────────────────────────────────────────────────────────────────

def get_state() -> dict:
    """Returns the current full state. Safe to call from dashboard (read-only)."""
    return _load()


def set_status(status: str) -> None:
    """Update agent status: 'idle' | 'listening' | 'processing' | 'speaking'"""
    state = _load()
    state["status"] = status
    _save(state)


def push_bl_result(
    transcript: str,
    views: list,
    weights_after: dict,
    sharpe_after: float,
) -> None:
    """
    Called by voice agent after a successful BL run.

    - Saves the new recommended weights (current weights stay unchanged until
      the manager manually confirms a rebalance — not implemented in v1).
    - Appends event to history.
    - Updates views and Sharpe.
    """
    state = _load()

    weights_before = state["portfolio"]["current"].copy()

    # Append to event history (keep last 20)
    event = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "transcript": transcript[:300],  # truncate for display
        "sharpe_before": state.get("sharpe_ratio"),
        "sharpe_after": round(sharpe_after, 4),
        "weights_before": weights_before,
        "weights_after": {k: round(v, 4) for k, v in weights_after.items()},
    }
    state["events"] = ([event] + state["events"])[:20]

    # Update portfolio recommendation
    state["portfolio"]["recommended"] = {k: round(v, 4) for k, v in weights_after.items()}

    # Update views and Sharpe
    state["views"] = views
    state["sharpe_ratio"] = round(sharpe_after, 4)
    state["status"] = "idle"

    _save(state)


def confirm_rebalance() -> None:
    """
    Moves recommended weights into current weights.
    Called when manager confirms the rebalance (future feature).
    """
    state = _load()
    if state["portfolio"]["recommended"]:
        state["portfolio"]["current"] = state["portfolio"]["recommended"].copy()
        state["portfolio"]["recommended"] = None
    _save(state)


def reset_state() -> None:
    """Resets to default state (useful for testing)."""
    state = json.loads(json.dumps(_DEFAULT_STATE))
    _save(state)


if __name__ == "__main__":
    # Quick smoke test
    reset_state()
    print("State reset.")
    push_bl_result(
        transcript="The Federal Reserve signaled rates higher for longer.",
        views=[
            {"description": "Bonds_USA yields rise on hawkish Fed", "confidence": 0.8,
             "type": "absolute", "asset": "Bonds_USA", "expected_return": -0.03},
            {"description": "US equities slightly negative on tighter policy", "confidence": 0.6,
             "type": "absolute", "asset": "Stocks_USA", "expected_return": -0.04},
        ],
        weights_after={"Stocks_USA": 0.53, "Stocks_EM": 0.17, "Bonds_USA": 0.30},
        sharpe_after=-0.18,
    )
    import pprint
    pprint.pprint(get_state())
