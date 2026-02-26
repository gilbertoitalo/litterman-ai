"""
dashboard.py — Litterman Voice Agent Dashboard

Run with:
    streamlit run dashboard.py

Polls shared_state.json every 2 seconds and re-renders.
"""

import streamlit as st
import time
import plotly.graph_objects as go
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from core.shared_state import get_state, INITIAL_WEIGHTS, confirm_rebalance

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Litterman",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme init ────────────────────────────────────────────────────────────────

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

T = st.session_state.theme

# ── Theme tokens ──────────────────────────────────────────────────────────────

DARK = {
    "bg":           "#0a0a0f",
    "surface":      "#0d0f18",
    "border":       "#1e2230",
    "border_subtle":"#12151e",
    "text":         "#c8cdd8",
    "text_strong":  "#e8eaf0",
    "text_muted":   "#4a5060",
    "text_dim":     "#2a2e3a",
    "accent":       "#4a9eff",
    "plot_bg":      "#0a0a0f",
    "plot_paper":   "#0a0a0f",
    "plot_grid":    "#12151e",
    "plot_text":    "#8890a0",
    "bar_current":  "#1e2a40",
    "bar_border":   "#2a3a58",
}

LIGHT = {
    "bg":           "#f4f5f7",
    "surface":      "#ffffff",
    "border":       "#dde1ea",
    "border_subtle":"#eaecf0",
    "text":         "#3a3f4a",
    "text_strong":  "#111318",
    "text_muted":   "#8890a0",
    "text_dim":     "#c0c4cc",
    "accent":       "#1a6fd4",
    "plot_bg":      "#ffffff",
    "plot_paper":   "#f4f5f7",
    "plot_grid":    "#eaecf0",
    "plot_text":    "#8890a0",
    "bar_current":  "#d0dff5",
    "bar_border":   "#a0b8e0",
}

C = DARK if T == "dark" else LIGHT

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: {C['bg']};
    color: {C['text']};
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 1.5rem 2rem 2rem; max-width: 100%; }}

/* Streamlit button reset */
.stButton > button {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 3px 12px !important;
    border-radius: 2px !important;
    border: 1px solid {C['border']} !important;
    background: transparent !important;
    color: {C['text_muted']} !important;
    height: auto !important;
    line-height: 1.6 !important;
    transition: all 0.15s ease !important;
}}
.stButton > button:hover {{
    border-color: {C['accent']} !important;
    color: {C['accent']} !important;
    background: transparent !important;
}}

/* Confirm rebalance button — full width green */
div[data-testid="column"]:last-child .stButton > button {{
    width: 100% !important;
    border-color: #2ea85a !important;
    color: #2ea85a !important;
    font-size: 0.7rem !important;
    padding: 6px 12px !important;
    margin-top: 0.4rem !important;
}}
div[data-testid="column"]:last-child .stButton > button:hover {{
    background: rgba(46,168,90,0.08) !important;
    border-color: #2ea85a !important;
    color: #2ea85a !important;
}}

.lt-header {{
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 2rem;
    border-bottom: 1px solid {C['border']};
    padding-bottom: 1rem;
}}
.lt-logo {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: {C['text_strong']};
    letter-spacing: -0.02em;
}}
.lt-logo span {{ color: {C['accent']}; }}
.lt-subtitle {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: {C['text_muted']};
    letter-spacing: 0.15em;
    text-transform: uppercase;
}}

.status-pill {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 2px;
    border: 1px solid;
    display: inline-block;
}}
.status-idle      {{ color: {C['text_muted']}; border-color: {C['border']}; }}
.status-listening {{ color: #4a9eff; border-color: #1a3050; background: #0d1a28; }}
.status-processing{{ color: #f0b040; border-color: #3a2a10; background: #1a1408; }}
.status-speaking  {{ color: #40c878; border-color: #103020; background: #081408; }}

.lt-section {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: {C['text_muted']};
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid {C['border_subtle']};
}}

.metric-card {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 3px;
    padding: 1rem 1.2rem;
}}
.metric-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {C['text_muted']};
    margin-bottom: 0.3rem;
}}
.metric-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 500;
    color: {C['text_strong']};
}}
.metric-value.positive {{ color: #2ea85a; }}
.metric-value.negative {{ color: #e85040; }}

.view-row {{
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid {C['border_subtle']};
    font-size: 0.85rem;
}}
.view-row:last-child {{ border-bottom: none; }}
.view-asset {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: {C['accent']};
    min-width: 100px;
}}
.view-desc {{ color: {C['text_muted']}; flex: 1; }}
.view-return {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    min-width: 60px;
    text-align: right;
}}
.view-return.pos {{ color: #2ea85a; }}
.view-return.neg {{ color: #e85040; }}
.view-conf {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: {C['text_muted']};
    min-width: 50px;
    text-align: right;
}}

.event-item {{
    padding: 0.8rem 0;
    border-bottom: 1px solid {C['border_subtle']};
}}
.event-item:last-child {{ border-bottom: none; }}
.event-time {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: {C['text_muted']};
    margin-bottom: 0.25rem;
}}
.event-transcript {{
    font-size: 0.82rem;
    color: {C['text_muted']};
    margin-bottom: 0.3rem;
    font-style: italic;
}}
.event-meta {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: {C['text_muted']};
}}
.event-sharpe {{ color: #f0b040; }}

.empty-state {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: {C['text_dim']};
    text-align: center;
    padding: 2rem;
    letter-spacing: 0.05em;
}}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_pct(val, decimals=1):
    return f"{val * 100:.{decimals}f}%"

def fmt_delta(before, after):
    d = (after - before) * 100
    return f"{'+' if d >= 0 else ''}{d:.1f}pp"

def delta_color(before, after):
    return "#2ea85a" if after >= before else "#e85040"

def status_html(status):
    labels = {
        "idle":       ("IDLE",          "status-idle"),
        "listening":  ("● LISTENING",   "status-listening"),
        "processing": ("◎ PROCESSING",  "status-processing"),
        "speaking":   ("◉ SPEAKING",    "status-speaking"),
    }
    label, cls = labels.get(status, ("UNKNOWN", "status-idle"))
    return f'<span class="status-pill {cls}">{label}</span>'


def make_weight_chart(current, recommended):
    assets = list(current.keys())
    cur_vals = [current[a] * 100 for a in assets]
    rec_vals = [recommended[a] * 100 for a in assets] if recommended else None

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=assets, x=cur_vals, name="Current", orientation="h",
        marker=dict(color=C["bar_current"], line=dict(color=C["bar_border"], width=1)),
        text=[f"{v:.1f}%" for v in cur_vals],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11, color=C["plot_text"]),
    ))

    if rec_vals:
        colors = [delta_color(c/100, r/100) for c, r in zip(cur_vals, rec_vals)]

        def hex_to_rgba(h, alpha=0.18):
            h = h.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        fig.add_trace(go.Bar(
            y=assets, x=rec_vals, name="Recommended", orientation="h",
            marker=dict(
                color=[hex_to_rgba(c) for c in colors],
                line=dict(color=colors, width=1.5),
            ),
            text=[f"{v:.1f}%" for v in rec_vals],
            textposition="outside",
            textfont=dict(family="IBM Plex Mono", size=11, color=C["text_strong"]),
        ))

    fig.update_layout(
        barmode="overlay",
        plot_bgcolor=C["plot_bg"],
        paper_bgcolor=C["plot_paper"],
        font=dict(family="IBM Plex Mono", color=C["plot_text"]),
        margin=dict(l=0, r=60, t=10, b=10),
        height=180,
        xaxis=dict(
            showgrid=True, gridcolor=C["plot_grid"], zeroline=False,
            ticksuffix="%", range=[0, 85], tickfont=dict(size=10),
        ),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, color=C["text_strong"])),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=10),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=bool(rec_vals),
    )
    return fig


# ── Render ────────────────────────────────────────────────────────────────────

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
    ts_str = last_updated.split("T")[1] if last_updated and "T" in last_updated else ""

    # ── Header ────────────────────────────────────────────────────────────────
    # Use columns so the Streamlit button renders natively (can't put st.button in HTML)
    h_left, h_right = st.columns([6, 1])

    with h_left:
        st.markdown(
            f'<div class="lt-header">'
            f'<span class="lt-logo">◈ Letter<span>AI</span></span>'
            f'<span class="lt-subtitle">Black-Litterman Voice Co-Pilot</span>'
            f'<div style="flex:1"></div>'
            f'{status_html(status)}'
            f'<span style="font-family: IBM Plex Mono; font-size: 0.65rem; '
            f'color: {C["text_dim"]}; margin-left: 1rem;">'
            f'{"UPDATED " + ts_str if ts_str else ""}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    with h_right:
        icon = "☀ LIGHT" if T == "dark" else "☾ DARK"
        st.markdown("<div style='padding-top: 0.6rem'>", unsafe_allow_html=True)
        if st.button(icon, key="theme_toggle"):
            st.session_state.theme = "light" if T == "dark" else "dark"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)

    if sharpe is None:
        sharpe_cls, sharpe_str = "", "—"
    elif sharpe > 0:
        sharpe_cls, sharpe_str = "positive", f"{sharpe:.4f}"
    else:
        sharpe_cls, sharpe_str = "negative", f"{sharpe:.4f}"

    with m1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Sharpe Ratio</div>'
            f'<div class="metric-value {sharpe_cls}">{sharpe_str}</div></div>',
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">BL Events Today</div>'
            f'<div class="metric-value">{len(events)}</div></div>',
            unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Active Views</div>'
            f'<div class="metric-value">{len(views)}</div></div>',
            unsafe_allow_html=True
        )
    with m4:
        has_rec = recommended is not None
        if has_rec:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Rebalance</div>',
                unsafe_allow_html=True
            )
            if st.button("✓ CONFIRM REBALANCE", key="confirm_rebalance"):
                confirm_rebalance()
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Rebalance</div><div class="metric-value" style="font-size:1.1rem; padding-top:0.3rem; color:{C["text_muted"]}">NONE</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Main 3 columns ────────────────────────────────────────────────────────
    col_chart, col_views, col_events = st.columns([2, 1.5, 1.5])

    # Portfolio chart
    with col_chart:
        st.markdown('<div class="lt-section">Portfolio Allocation</div>', unsafe_allow_html=True)
        fig = make_weight_chart(current, recommended)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if recommended:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            cols = st.columns(len(current))
            for i, (asset, cur_w) in enumerate(current.items()):
                rec_w = recommended.get(asset, cur_w)
                dcolor = delta_color(cur_w, rec_w)
                with cols[i]:
                    st.markdown(
                        f'<div style="text-align:center; padding:0.5rem; background:{C["surface"]}; '
                        f'border:1px solid {C["border"]}; border-radius:3px;">'
                        f'<div style="font-family:IBM Plex Mono; font-size:0.6rem; '
                        f'color:{C["text_muted"]}; letter-spacing:0.1em; margin-bottom:0.2rem;">'
                        f'{asset.replace("_", " ")}</div>'
                        f'<div style="font-family:IBM Plex Mono; font-size:0.85rem; color:{C["text_strong"]};">'
                        f'{fmt_pct(cur_w)} → {fmt_pct(rec_w)}</div>'
                        f'<div style="font-family:IBM Plex Mono; font-size:0.8rem; color:{dcolor};">'
                        f'{fmt_delta(cur_w, rec_w)}</div></div>',
                        unsafe_allow_html=True
                    )

    # Extracted views
    with col_views:
        st.markdown('<div class="lt-section">Extracted Views</div>', unsafe_allow_html=True)
        if not views:
            st.markdown('<div class="empty-state">No views extracted yet.<br>Speak market news to begin.</div>', unsafe_allow_html=True)
        else:
            html = ""
            for v in views:
                asset = v.get("asset") or v.get("asset_long", "")
                exp_ret = v.get("expected_return", 0)
                conf = v.get("confidence", 0)
                ret_cls = "pos" if exp_ret >= 0 else "neg"
                ret_str = f"{'+' if exp_ret >= 0 else ''}{exp_ret*100:.1f}%"
                html += (
                    f'<div class="view-row">'
                    f'<div class="view-asset">{asset}</div>'
                    f'<div class="view-desc">{v.get("description","")}</div>'
                    f'<div class="view-return {ret_cls}">{ret_str}</div>'
                    f'<div class="view-conf">{conf:.0%}</div>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="background:{C["surface"]}; border:1px solid {C["border"]}; '
                f'border-radius:3px; padding:0.5rem 1rem;">{html}</div>',
                unsafe_allow_html=True
            )

    # Event log
    with col_events:
        st.markdown('<div class="lt-section">Event Log</div>', unsafe_allow_html=True)
        if not events:
            st.markdown('<div class="empty-state">No events yet.</div>', unsafe_allow_html=True)
        else:
            html = ""
            for evt in events[:8]:
                ts = evt.get("timestamp", "")
                time_str = ts.split("T")[1] if "T" in ts else ts
                transcript = evt.get("transcript", "")[:120]
                s_after = evt.get("sharpe_after")
                s_str = f"{s_after:.4f}" if s_after is not None else "—"
                wb, wa = evt.get("weights_before", {}), evt.get("weights_after", {})
                changes = []
                for a in wa:
                    if a in wb:
                        d = (wa[a] - wb[a]) * 100
                        if abs(d) >= 0.5:
                            changes.append(f"{a.split('_')[0]} {'+' if d>=0 else ''}{d:.1f}pp")
                changes_str = " · ".join(changes) if changes else "no change"
                html += (
                    f'<div class="event-item">'
                    f'<div class="event-time">{time_str}</div>'
                    f'<div class="event-transcript">"{transcript}..."</div>'
                    f'<div class="event-meta">{changes_str} &nbsp;|&nbsp; '
                    f'<span class="event-sharpe">Sharpe {s_str}</span></div>'
                    f'</div>'
                )
            st.markdown(
                f'<div style="background:{C["surface"]}; border:1px solid {C["border"]}; '
                f'border-radius:3px; padding:0.5rem 1rem;">{html}</div>',
                unsafe_allow_html=True
            )

    # Last transcript
    if events:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="lt-section">Last Transcript</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:{C["surface"]}; border:1px solid {C["border"]}; '
            f'border-radius:3px; padding:0.8rem 1.2rem; font-size:0.85rem; '
            f'color:{C["text_muted"]}; font-style:italic;">'
            f'"{events[0].get("transcript","")}"</div>',
            unsafe_allow_html=True
        )

    # Auto-refresh
    time.sleep(2)
    st.rerun()


render()
