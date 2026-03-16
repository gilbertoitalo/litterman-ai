"""
server.py — Litterman AI · Cloud Run backend
"""

import os
import time
import numpy as np
from collections import defaultdict
from functools import wraps
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv
from werkzeug.middleware.proxy_fix import ProxyFix

load_dotenv()

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

_request_log: dict = defaultdict(list)
RATE_LIMIT = 10
WINDOW_SEC = 60

def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = request.headers.get("X-Forwarded-For", request.remote_addr)
        ip = ip.split(",")[0].strip()
        now = time.time()
        window_start = now - WINDOW_SEC

        _request_log[ip] = [t for t in _request_log[ip] if t > window_start]
        count = len(_request_log[ip])

        # Visible in Cloud Run logs
        print(f"[RATE] ip={ip} count={count}/{RATE_LIMIT}", flush=True)

        if count >= RATE_LIMIT:
            print(f"[RATE] BLOCKED ip={ip}", flush=True)
            return jsonify({"error": "Rate limit exceeded. Try again in a minute."}), 429

        _request_log[ip].append(now)
        return f(*args, **kwargs)
    return decorated

ASSETS = ['Stocks_USA', 'Stocks_EM', 'Bonds_USA']
COV = np.diag([0.0225, 0.0324, 0.0025])

@app.route("/", methods=["GET"])
def index():
    return send_file("dashboard.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/analyse", methods=["POST"])
@rate_limit
def analyse():
    from core.gemini_agent import run_bl_pipeline, fetch_url_content
    from core.shared_state import push_bl_result, get_state

    data = request.get_json(force=True, silent=True) or {}
    url = (data.get("url") or "").strip()
    text = (data.get("text") or "").strip()

    if not url and not text:
        return jsonify({"error": "Provide 'url' or 'text' in request body."}), 400

    try:
        if url:
            content = fetch_url_content(url)
            source_label = url[:60] + ("..." if len(url) > 60 else "")
        else:
            content = text
            source_label = "manual scenario"

        state = get_state()
        current_weights = state["portfolio"]["current"]
        weights_live = np.array([current_weights[a] for a in ASSETS])

        result = run_bl_pipeline(content, ASSETS, weights_live, COV)

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)