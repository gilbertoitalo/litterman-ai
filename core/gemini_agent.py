import json
import numpy as np
from google import genai
from dotenv import load_dotenv
import os
from core.bl_engine import BlackLittermanEngine

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def extract_views_from_news(news_text: str, assets: list) -> dict:
    """
    Uses Gemini to extract Black-Litterman views from a news article.

    Returns dict with:
        - views: list of view descriptions
        - P: pick matrix
        - Q: expected returns vector
        - confidence: confidence level per view (0 to 1)
    """
    prompt = f"""
You are a quantitative analyst assistant. Analyse the following news and extract 
Black-Litterman views for a portfolio with these assets: {assets}

News:
{news_text}

Respond ONLY with a JSON object in this exact format, no explanation:
{{
    "views": [
        {{
            "description": "brief description of the view",
            "type": "absolute or relative",
            "asset": "asset name if absolute",
            "asset_long": "asset to long if relative",
            "asset_short": "asset to short if relative",
            "expected_return": 0.00,
            "confidence": 0.00
        }}
    ]
}}

Rules:
- expected_return must be annual, as decimal (e.g. 0.09 for 9%)
- confidence between 0.1 (low) and 0.9 (high)
- only include assets from the provided list
- maximum 3 views
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    text = response.text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    return json.loads(text)


def views_to_matrices(views_data: dict, assets: list):
    """
    Converts Gemini JSON output to P and Q matrices for Black-Litterman.
    """
    views = views_data["views"]
    n_assets = len(assets)
    n_views = len(views)

    P = np.zeros((n_views, n_assets))
    Q = np.zeros(n_views)

    for i, view in enumerate(views):
        if view["type"] == "absolute":
            asset_idx = assets.index(view["asset"])
            P[i, asset_idx] = 1.0
            Q[i] = view["expected_return"]
        elif view["type"] == "relative":
            long_idx = assets.index(view["asset_long"])
            short_idx = assets.index(view["asset_short"])
            P[i, long_idx] = 1.0
            P[i, short_idx] = -1.0
            Q[i] = view["expected_return"]

    return P, Q


def run_bl_pipeline(news_text: str, assets: list, weights: np.ndarray, cov: np.ndarray) -> dict:
    """
    Full pipeline: news -> Gemini views -> Black-Litterman -> optimal weights.

    Returns dict with views, optimal weights, and Sharpe ratio.
    Ready to be injected into the voice session as a structured prompt.
    """
    # Step 1 — extract views from news
    views_data = extract_views_from_news(news_text, assets)

    # Step 2 — convert to matrices
    P, Q = views_to_matrices(views_data, assets)

    # Step 3 — run Black-Litterman
    bl = BlackLittermanEngine(assets, weights, cov)
    bl.add_views(P, Q)
    result = bl.optimize()

    return {
        "views": views_data["views"],
        "weights": result["weights"],
        "sharpe_ratio": result["sharpe_ratio"]
    }


def format_bl_result_for_voice(bl_result: dict, original_weights: dict) -> str:
    """
    Formats BL pipeline result into a structured prompt for the voice agent.
    Gemini will read this and respond verbally to the manager.
    """
    views_text = "\n".join(
        f"- {v['description']} (confidence: {v['confidence']:.0%})"
        for v in bl_result["views"]
    )

    weights_text = "\n".join(
        f"- {asset}: {original_weights.get(asset, 0)*100:.1f}% -> {weight*100:.1f}%"
        for asset, weight in bl_result["weights"].items()
    )

    return f"""
The manager just described a market event. You have already run the Black-Litterman model.
Present the results clearly and concisely, as a senior quant analyst would in a brief verbal briefing.

Views extracted:
{views_text}

Portfolio rebalancing recommendation:
{weights_text}

Sharpe Ratio: {bl_result["sharpe_ratio"]:.4f}

Speak naturally. Highlight the most significant weight changes and what is driving them.
Do not read all numbers — synthesise the key message.
"""


if __name__ == "__main__":
    assets = ['Stocks_USA', 'Stocks_EM', 'Bonds_USA']
    weights = np.array([0.60, 0.30, 0.10])
    cov = np.diag([0.0225, 0.0324, 0.0025])

    news = """
    The Federal Reserve signaled today that it will keep interest rates higher for longer,
    citing persistent inflation concerns. Markets reacted with US equities falling sharply
    while emerging markets showed resilience. Bond yields rose significantly as investors
    priced in fewer rate cuts for the year ahead.
    """

    print("Running full BL pipeline...\n")
    result = run_bl_pipeline(news, assets, weights, cov)

    print("Views:")
    for v in result["views"]:
        print(f"  - {v['description']} ({v['confidence']:.0%})")

    print("\nOptimal Weights:")
    for asset, weight in result["weights"].items():
        print(f"  {asset}: {weight*100:.2f}%")

    print(f"\nSharpe Ratio: {result['sharpe_ratio']:.4f}")