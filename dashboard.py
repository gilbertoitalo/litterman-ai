"""
dashboard.py — Litterman Voice Agent Dashboard

Run with:
    streamlit run dashboard.py

Polls shared_state.json every 2 seconds and re-renders.
"""

import streamlit as st
import json
import time
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path

# Import shared_state from same directory
import sys
sys.path.insert(0, str(Path(__file__).parent))
from core.shared_state import get_state, INITIAL_WEIGHTS

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Litterman",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS — dark terminal aesthetic ─────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0a0f;
    color: #c8cdd8;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem; max-width: 100%; }

/* Header */
.lt-header {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 2rem;
    border-bottom: 1px solid #1e2230;
    padding-bottom: 1rem;
}
.lt-logo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #e8eaf0;
    letter-spacing: -0.02em;
}
.lt-logo span { color: #4a9eff; }
.lt-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #4a5060;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* Status pill */
.status-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 2px;
    border: 1px solid;
    display: inline-block;
}
.status-idle     { color: #4a5060; border-color: #1e2230; }
.status-listening{ color: #4a9eff; border-color: #1a3050; background: #0d1a28; }
.status-processing{ color: #f0b040; border-color: #3a2a10; background: #1a1408; }
.status-speaking { color: #40c878; border-color: #103020; background: #081408; }

/* Section headers */
.lt-section {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4a5060;
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #12151e;
}

/* Metric cards */
.metric-card {
    background: #0d0f18;
    border: 1px solid #1e2230;
    border-radius: 3px;
    padding: 1rem 1.2rem;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a5060;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    color: #e8eaf0;
}
.metric-value.positive { color: #40c878; }
.metric-value.negative { color: #e85040; }

/* Views table */
.view-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid #12151e;
    font-size: 0.85rem;
}
.view-row:last-child { border-bottom: none; }
.view-asset {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #4a9eff;
    min-width: 100px;
}
.view-desc { color: #8890a0; flex: 1; }
.view-return {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    min-width: 60px;
    text-align: right;
}
.view-return.pos { color: #40c878; }
.view-return.neg { color: #e85040; }
.view-conf {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #4a5060;
    min-width: 50px;
    text-align: right;
}

/* Events list */
.event-item {
    padding: 0.8rem 0;
    border-bottom: 1px solid #12151e;
}
.event-item:last-child { border-bottom: none; }
.event-time {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #4a5060;
    margin-bottom: 0.25rem;
}
.event-transcript {
    font-size: 0.82rem;
    color: #8890a0;
    margin-bottom: 0.3rem;
    font-style: italic;
}
.event-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #4a5060;
}
.event-sharpe {
    color: #f0b040;
}

/* Empty states */
.empty-state {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #2a2e3a;
    text-align: center;
    padding: 2rem;
    letter-spacing: 0.05em;
}

/* No views */
.no-recommendation {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #2a2e3a;
    font-style: italic;
}
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

def fmt_pct(val: float, decimals: int = 1) -> str:
    return f"{val * 100:.{decimals}f}%"

def fmt_delta(before: float, after: float) -> str:
    d = (after - before) * 100
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}pp"

def delta_color(before: float, after: float) -> str:
    return "#40c878" if after >= before else "#e85040"

def sharpe_color(s: float) -> str:
    if s is None: return "#4a5060"
    return "#40c878" if s > 0 else "#e85040"

def status_html(status: str) -> str:
    labels = {
        "idle": ("IDLE", "status-idle"),
        "listening": ("● LISTENING", "status-listening"),
        "processing": ("◎ PROCESSING", "status-processing"),
        "speaking": ("◉ SPEAKING", "status-speaking"),
    }
    label, cls = labels.get(status, ("UNKNOWN", "status-idle"))
    return f'<span class="status-pill {cls}">{label}</span>'


def make_weight_chart(current: dict, recommended: dict | None) -> go.Figure:
    """Horizontal bar chart: current weights vs recommended."""
    assets = list(current.keys())
    cur_vals = [current[a] * 100 for a in assets]
    rec_vals = [recommended[a] * 100 for a in assets] if recommended else None

    fig = go.Figure()

    # Current weights
    fig.add_trace(go.Bar(
        y=assets,
        x=cur_vals,
        name="Current",
        orientation="h",
        marker=dict(color="#1e2a40", line=dict(color="#2a3a58", width=1)),
        text=[f"{v:.1f}%" for v in cur_vals],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11, color="#8890a0"),
    ))

    # Recommended weights (overlay)
    if rec_vals:
        colors = [delta_color(c/100, r/100) for c, r in zip(cur_vals, rec_vals)]
        fig.add_trace(go.Bar(
            y=assets,
            x=rec_vals,
            name="Recommended",
            orientation="h",
            marker=dict(
                color=[c + "30" for c in colors],  # semi-transparent fill
                line=dict(color=colors, width=1.5),
            ),
            text=[f"{v:.1f}%" for v in rec_vals],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=11, color="#c8cdd8"),
        ))

    fig.update_layout(
        barmode="overlay",
        plot_bgcolor="#0a0a0f",
        paper_bgcolor="#0a0a0f",
        font=dict(family="IBM Plex Mono", color="#8890a0"),
        margin=dict(l=0, r=60, t=10, b=10),
        height=180,
        xaxis=dict(
            showgrid=True,
            gridcolor="#12151e",
            zeroline=False,
            ticksuffix="%",
            range=[0, 85],
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=11, color="#c8cdd8"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=bool(rec_vals),
    )
    return fig


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    state = get_state()
    portfolio = state.get("portfolio", {})
    current = portfolio.get("current", INITIAL_WEIGHTS)
    recommended = portfolio.get("recommended")
    views = state.get("views", [])
    sharpe = state.get("sharpe_ratio")
    events = state.get("events", [])
    status = state.get("status", "idle")
    last_updated = state.get("last_updated")

    # ── Header ────────────────────────────────────────────────────────────────
    col_logo, col_status, col_time = st.columns([3, 1, 1])

    with col_logo:
        st.markdown(
            '<div class="lt-header">'
            '<span class="lt-logo">◈ Letter<span>AI</span></span>'
            '<span class="lt-subtitle">Black-Litterman Voice Co-Pilot</span>'
            '</div>',
            unsafe_allow_html=True
        )

    with col_status:
        st.markdown(
            f'<div style="padding-top: 1.4rem">{status_html(status)}</div>',
            unsafe_allow_html=True
        )

    with col_time:
        if last_updated:
            ts = last_updated.split("T")[1] if "T" in last_updated else last_updated
            st.markdown(
                f'<div style="padding-top: 1.5rem; text-align: right; '
                f'font-family: IBM Plex Mono; font-size: 0.65rem; color: #2a2e3a;">'
                f'UPDATED {ts}</div>',
                unsafe_allow_html=True
            )

    # ── Top metrics row ───────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)

    sharpe_cls = "positive" if sharpe and sharpe > 0 else "negative"
    sharpe_str = f"{sharpe:.4f}" if sharpe is not None else "—"

    with m1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Sharpe Ratio</div>'
            f'<div class="metric-value {sharpe_cls}">{sharpe_str}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with m2:
        n_events = len(events)
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">BL Events Today</div>'
            f'<div class="metric-value">{n_events}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with m3:
        n_views = len(views)
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Active Views</div>'
            f'<div class="metric-value">{n_views}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with m4:
        has_rec = recommended is not None
        rec_label = "PENDING" if has_rec else "NONE"
        rec_cls = "positive" if has_rec else ""
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Rebalance</div>'
            f'<div class="metric-value {rec_cls}" style="font-size: 1.1rem; padding-top: 0.3rem">{rec_label}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

    # ── Main content: 3 columns ───────────────────────────────────────────────
    col_chart, col_views, col_events = st.columns([2, 1.5, 1.5])

    # ── Column 1: Portfolio weights chart ─────────────────────────────────────
    with col_chart:
        st.markdown('<div class="lt-section">Portfolio Allocation</div>', unsafe_allow_html=True)

        fig = make_weight_chart(current, recommended)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Weight delta table
        if recommended:
            st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
            cols = st.columns(len(current))
            for i, (asset, cur_w) in enumerate(current.items()):
                rec_w = recommended.get(asset, cur_w)
                delta_str = fmt_delta(cur_w, rec_w)
                dcolor = delta_color(cur_w, rec_w)
                with cols[i]:
                    st.markdown(
                        f'<div style="text-align: center; padding: 0.5rem; background: #0d0f18; '
                        f'border: 1px solid #1e2230; border-radius: 3px;">'
                        f'<div style="font-family: IBM Plex Mono; font-size: 0.6rem; '
                        f'color: #4a5060; letter-spacing: 0.1em; margin-bottom: 0.2rem;">'
                        f'{asset.replace("_", " ")}</div>'
                        f'<div style="font-family: IBM Plex Mono; font-size: 0.85rem; color: #e8eaf0;">'
                        f'{fmt_pct(cur_w)} → {fmt_pct(rec_w)}</div>'
                        f'<div style="font-family: IBM Plex Mono; font-size: 0.8rem; color: {dcolor};">'
                        f'{delta_str}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    # ── Column 2: Extracted views ──────────────────────────────────────────────
    with col_views:
        st.markdown('<div class="lt-section">Extracted Views</div>', unsafe_allow_html=True)

        if not views:
            st.markdown(
                '<div class="empty-state">No views extracted yet.<br>Speak market news to begin.</div>',
                unsafe_allow_html=True
            )
        else:
            views_html = ""
            for v in views:
                asset = v.get("asset") or v.get("asset_long", "")
                exp_ret = v.get("expected_return", 0)
                conf = v.get("confidence", 0)
                desc = v.get("description", "")
                ret_cls = "pos" if exp_ret >= 0 else "neg"
                ret_str = f"{'+' if exp_ret >= 0 else ''}{exp_ret*100:.1f}%"

                views_html += (
                    f'<div class="view-row">'
                    f'<div class="view-asset">{asset}</div>'
                    f'<div class="view-desc">{desc}</div>'
                    f'<div class="view-return {ret_cls}">{ret_str}</div>'
                    f'<div class="view-conf">{conf:.0%}</div>'
                    f'</div>'
                )

            st.markdown(
                f'<div style="background: #0d0f18; border: 1px solid #1e2230; '
                f'border-radius: 3px; padding: 0.5rem 1rem;">{views_html}</div>',
                unsafe_allow_html=True
            )

    # ── Column 3: Event history ────────────────────────────────────────────────
    with col_events:
        st.markdown('<div class="lt-section">Event Log</div>', unsafe_allow_html=True)

        if not events:
            st.markdown(
                '<div class="empty-state">No events yet.</div>',
                unsafe_allow_html=True
            )
        else:
            events_html = ""
            for evt in events[:8]:  # show last 8
                ts = evt.get("timestamp", "")
                time_str = ts.split("T")[1] if "T" in ts else ts
                transcript = evt.get("transcript", "")[:120]
                sharpe_after = evt.get("sharpe_after")
                sharpe_str = f"{sharpe_after:.4f}" if sharpe_after is not None else "—"

                # Summarise weight changes
                wb = evt.get("weights_before", {})
                wa = evt.get("weights_after", {})
                changes = []
                for asset in wa:
                    if asset in wb:
                        d = (wa[asset] - wb[asset]) * 100
                        if abs(d) >= 0.5:
                            sign = "+" if d >= 0 else ""
                            changes.append(f"{asset.split('_')[0]} {sign}{d:.1f}pp")
                changes_str = " · ".join(changes) if changes else "no change"

                events_html += (
                    f'<div class="event-item">'
                    f'<div class="event-time">{time_str}</div>'
                    f'<div class="event-transcript">"{transcript}..."</div>'
                    f'<div class="event-meta">{changes_str} &nbsp;|&nbsp; '
                    f'<span class="event-sharpe">Sharpe {sharpe_str}</span></div>'
                    f'</div>'
                )

            st.markdown(
                f'<div style="background: #0d0f18; border: 1px solid #1e2230; '
                f'border-radius: 3px; padding: 0.5rem 1rem;">{events_html}</div>',
                unsafe_allow_html=True
            )

    # ── Last transcript ────────────────────────────────────────────────────────
    if events:
        last = events[0]
        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="lt-section">Last Transcript</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="background: #0d0f18; border: 1px solid #1e2230; border-radius: 3px; '
            f'padding: 0.8rem 1.2rem; font-size: 0.85rem; color: #8890a0; font-style: italic;">'
            f'"{last.get("transcript", "")}"'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    time.sleep(2)
    st.rerun()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__" or True:
    render()
