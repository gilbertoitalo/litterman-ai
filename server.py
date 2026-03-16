"""
server.py — Litterman AI · Cloud Run backend

Serves:
  GET  /           → dashboard.html
  POST /analyse    → runs BL pipeline on URL or text, writes result to Firestore
  GET  /health     → health check
"""

import os
import json
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ── Rate limiting ─────────────────────────────────────────────────────────────
# 60 requests/hour per IP globally; /analyse capped at 10/minute per IP
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["60 per hour"],
    storage_uri="memory://",
)

ASSETS = ['Stocks_USA', 'Stocks_EM', 'Bonds_USA']
COV = np.diag([0.0225, 0.0324, 0.0025])

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return send_file("dashboard.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/analyse", methods=["POST"])
@limiter.limit("10 per minute")
def analyse():
    from core.gemini_agent import run_bl_pipeline, fetch_url_content
    from core.shared_state import push_bl_result, get_state

    data = request.get_json(force=True, silent=True) or {}
    url = (data.get("url") or "").strip()
    text = (data.get("text") or "").strip()

    if not url and not text:
        return jsonify({"error": "Provide 'url' or 'text' in request body."}), 400

    try:
        # Fetch URL content if provided
        if url:
            content = fetch_url_content(url)
            source_label = url[:60] + ("..." if len(url) > 60 else "")
        else:
            content = text
            source_label = "manual scenario"

        # Read live weights from Firestore
        state = get_state()
        current_weights = state["portfolio"]["current"]
        weights_live = np.array([current_weights[a] for a in ASSETS])

        # Run BL pipeline
        result = run_bl_pipeline(content, ASSETS, weights_live, COV)

        # Push to Firestore
        push_bl_result(
            transcript=f"[Manual — {source_label}] {content[:200]}",
            views=result["views"],
            weights_after=result["weights"],
            sharpe_after=result["sharpe_ratio"],
        )

        return jsonify({
            "sharpe_ratio": result["sharpe_ratio"],
            "weights": result["weights"],
            "views": result["views"],
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        return jsonify({"error": str(e)[:200]}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)