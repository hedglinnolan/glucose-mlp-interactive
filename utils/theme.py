"""
Global theme and styling for Tabular ML Lab.

Material Design inspired, clean and modern.
Inject via inject_custom_css() at the top of each page.
"""
import streamlit as st


def inject_custom_css():
    """Inject global Material Design-inspired CSS."""
    st.markdown("""
    <style>
    /* ── Global ──────────────────────────────────────────────── */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* ── Cards / Containers ──────────────────────────────────── */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .info-card h3 { color: white; margin-top: 0; font-size: 1.1rem; }
    .info-card p { color: rgba(255,255,255,0.9); margin: 0; }

    .guidance-card {
        background: #f8f9ff;
        border-left: 4px solid #667eea;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .guidance-card strong { color: #4a5568; }

    .warning-card {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }

    .success-card {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }

    /* ── Metric Cards ────────────────────────────────────────── */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        flex: 1;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e40af;
        margin: 0.25rem 0;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-ci {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }

    /* ── Section Headers ─────────────────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .section-header h2 {
        margin: 0;
        font-size: 1.4rem;
        color: #1e293b;
    }

    /* ── Reviewer Concern Badge ──────────────────────────────── */
    .reviewer-concern {
        background: #fef3c7;
        border: 1px solid #fbbf24;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.88rem;
    }
    .reviewer-concern::before {
        content: "⚠️ Reviewer concern: ";
        font-weight: 600;
    }

    /* ── Step Indicator ──────────────────────────────────────── */
    .step-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #eff6ff;
        border-radius: 20px;
        padding: 0.35rem 1rem;
        font-size: 0.85rem;
        color: #1e40af;
        font-weight: 500;
        margin-bottom: 1rem;
    }

    /* ── Tables ──────────────────────────────────────────────── */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* ── Buttons ─────────────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.15s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* ── Expanders ───────────────────────────────────────────── */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* ── Progress indicator ──────────────────────────────────── */
    .workflow-step {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem 0;
    }
    .workflow-step-number {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .workflow-step-active .workflow-step-number {
        background: #667eea;
        color: white;
    }
    .workflow-step-complete .workflow-step-number {
        background: #22c55e;
        color: white;
    }
    .workflow-step-pending .workflow-step-number {
        background: #e2e8f0;
        color: #94a3b8;
    }

    /* ── Tooltips / Why badges ───────────────────────────────── */
    .why-badge {
        display: inline-block;
        background: #eff6ff;
        color: #3b82f6;
        border-radius: 4px;
        padding: 0.15rem 0.4rem;
        font-size: 0.75rem;
        cursor: help;
        margin-left: 0.25rem;
    }

    /* ── Hide default Streamlit footer ───────────────────────── */
    footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)


def render_guidance(text: str, icon: str = "💡"):
    """Render a guidance card with actionable advice."""
    st.markdown(f"""
    <div class="guidance-card">
        {icon} {text}
    </div>
    """, unsafe_allow_html=True)


def render_reviewer_concern(text: str):
    """Render a reviewer concern badge."""
    st.markdown(f"""
    <div class="reviewer-concern">{text}</div>
    """, unsafe_allow_html=True)


def render_step_indicator(step_number: int, step_name: str, total_steps: int = 7):
    """Render a step indicator badge."""
    st.markdown(f"""
    <div class="step-indicator">
        Step {step_number} of {total_steps} · {step_name}
    </div>
    """, unsafe_allow_html=True)


def render_info_card(title: str, body: str):
    """Render a gradient info card."""
    st.markdown(f"""
    <div class="info-card">
        <h3>{title}</h3>
        <p>{body}</p>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, ci: str = ""):
    """Render a single metric card with optional CI."""
    ci_html = f'<div class="metric-ci">{ci}</div>' if ci else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {ci_html}
    </div>
    """


def render_metric_row(metrics: list):
    """Render a row of metric cards.

    metrics: list of (label, value, ci_text) tuples
    """
    cards = "".join(render_metric_card(l, v, c) for l, v, c in metrics)
    st.markdown(f'<div class="metric-row">{cards}</div>', unsafe_allow_html=True)
