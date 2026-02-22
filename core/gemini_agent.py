import json
import numpy as np
from google import genai
from dotenv import load_dotenv
import os
from .bl_engine import BlackLittermanEngine

import os


load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

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
        model="gemini-3-flash-preview",
        contents=prompt
    )
    
    # Clean response and parse JSON
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


# --- TEST ---
if __name__ == "__main__":
    assets = ['Stocks_USA', 'Stocks_EM', 'Bonds_USA']
    weights = np.array([0.60, 0.30, 0.10])
    cov = np.diag([0.0225, 0.0324, 0.0025])

    # Sample news
    news = """
    The Federal Reserve signaled today that it will keep interest rates higher for longer,
    citing persistent inflation concerns. Markets reacted with US equities falling sharply
    while emerging markets showed resilience. Bond yields rose significantly as investors
    priced in fewer rate cuts for the year ahead.
    """

    print("Extracting views from news...\n")
    views_data = extract_views_from_news(news, assets)
    
    print("Views extracted by Gemini:")
    for v in views_data["views"]:
        print(f"  - {v['description']} (confidence: {v['confidence']})")
    
    print("\nConverting to matrices...")
    P, Q = views_to_matrices(views_data, assets)
    
    print(f"  P matrix:\n{P}")
    print(f"  Q vector: {Q}")
    
    print("\nRunning Black-Litterman optimization...")
    bl = BlackLittermanEngine(assets, weights, cov)
    bl.add_views(P, Q)
    
    result = bl.optimize()
    
    print("\nOptimal Portfolio Weights:")
    for asset, weight in result['weights'].items():
        print(f"  {asset}: {weight*100:.2f}%")
    
    print(f"\nSharpe Ratio: {result['sharpe_ratio']:.4f}")