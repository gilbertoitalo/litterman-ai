import json
import numpy as np
from google import genai
from dotenv import load_dotenv
import os
from core.bl_engine import BlackLittermanEngine

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def fetch_url_content(url: str, max_chars: int = 4000) -> str:
    """
    Fetches a news article URL and extracts readable text content.
    Uses Gemini to summarise if the raw content is too long.

    Returns a clean text string suitable for BL view extraction.
    Raises ValueError if the URL cannot be fetched or parsed.
    """
    import urllib.request
    import urllib.error
    import re

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            raw_bytes = response.read()
            encoding = response.headers.get_content_charset() or "utf-8"
            html = raw_bytes.decode(encoding, errors="replace")
    except urllib.error.URLError as e:
        raise ValueError(f"Could not fetch URL: {e.reason}")
    except Exception as e:
        raise ValueError(f"Fetch error: {str(e)}")

    # Strip HTML tags
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'\s+', ' ', text).strip()

    if not text or len(text) < 100:
        raise ValueError("Could not extract readable text from this URL. Try pasting the article text directly.")

    # If content is too long, use Gemini to extract the financial news summary
    if len(text) > max_chars:
        summary_prompt = f"""
Extract only the key financial and market information from this article text.
Focus on: market events, economic data, central bank decisions, corporate earnings, or geopolitical events that affect asset prices.
Return a concise 2-3 paragraph summary. If this is not a financial article, say "NOT_FINANCIAL".

Article text (first 8000 chars):
{text[:8000]}
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=summary_prompt
        )
        summary = response.text.strip()

        if "NOT_FINANCIAL" in summary:
            raise ValueError("This URL does not appear to contain financial market news.")

        return summary

    return text[:max_chars]


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
- expected_return must be REALISTIC and BOUNDED:
    * Stocks_USA normal range: -0.15 to +0.15 (e.g. bearish: -0.05, bullish: +0.08)
    * Stocks_EM normal range:  -0.20 to +0.20 (e.g. bearish: -0.08, bullish: +0.10)
    * Bonds_USA normal range:  -0.08 to +0.08 (e.g. bearish: -0.03, bullish: +0.04)
- NEVER use expected_return outside [-0.25, +0.25] — these are annual return adjustments, not price shocks
- confidence between 0.1 (low) and 0.9 (high)
- only include assets from the provided list
- maximum 3 views
- for rate hike news: Bonds_USA expected_return should be NEGATIVE (price falls as yields rise)
- for rate hike news: Stocks_USA and Stocks_EM expected_return should be slightly NEGATIVE
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
    Formats BL result as a direct spoken script — not a prompt asking the model
    to generate something, but the exact structure it should speak aloud.
    This prevents the model from reasoning about what to say.
    """
    # Find the 2 biggest weight changes
    changes = []
    for asset, new_w in bl_result["weights"].items():
        old_w = original_weights.get(asset, 0)
        delta_pp = (new_w - old_w) * 100
        if abs(delta_pp) >= 0.5:
            direction = "up" if delta_pp > 0 else "down"
            changes.append((abs(delta_pp), asset.replace("_", " "), new_w * 100, direction))
    changes.sort(reverse=True)
    top = changes[:2]

    # Key view driver
    top_view = bl_result["views"][0]["description"] if bl_result["views"] else "market event"
    sharpe = bl_result["sharpe_ratio"]

    # Build the exact spoken lines — model just reads this
    lines = []
    if top:
        a1_name, a1_w, a1_dir = top[0][1], top[0][2], top[0][3]
        lines.append(f"Following the {top_view}, the model recommends moving {a1_name} {a1_dir} to {a1_w:.1f} percent.")
    if len(top) > 1:
        a2_name, a2_w, a2_dir = top[1][1], top[1][2], top[1][3]
        lines.append(f"{a2_name} moves {a2_dir} to {a2_w:.1f} percent.")
    sharpe_comment = "above breakeven" if sharpe > 0 else "negative given current headwinds"
    lines.append(f"The portfolio Sharpe ratio is {sharpe:.4f}, {sharpe_comment}.")

    script = " ".join(lines)

    return f"""Say exactly this, word for word, with no additions, no preamble, no headers:

{script}"""


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
