"""
Reusable UI for "Interpret these results using an LLM" (Ollama).
Renders button + optional AI result; auto-starts Ollama if not running.
"""
from __future__ import annotations

from typing import Optional, Dict, Any, List

_MAX_TABLE_ROWS = 20
_MAX_TABLE_CHARS = 2000


def _infer_domain_hint(feature_names: Optional[List[str]] = None) -> str:
    """Infer a simple domain hint (e.g. 'clinical') from feature/column names."""
    if not feature_names:
        return ""
    clinical_like = {"glucose", "bmi", "age", "weight", "height", "bp", "hdl", "ldl", "hb", "waist", "hip"}
    names_lower = " ".join(str(x).lower() for x in feature_names)
    if any(k in names_lower for k in clinical_like):
        return "clinical"
    return ""


def build_eda_full_results_context(result: Dict[str, Any], action_id: str) -> str:
    """
    Build a full EDA-results context string for the LLM: all findings, stats, and
    per-figure/table descriptions or data. Use when an action returns multiple plots/tables.
    """
    import pandas as pd

    parts: List[str] = []
    findings = result.get("findings", [])
    if findings:
        parts.append("Findings: " + "; ".join(findings))

    stats = result.get("stats", {})
    if stats:
        lines: List[str] = []
        if "correlation_tests" in stats:
            for t in stats["correlation_tests"]:
                if len(t) >= 4:
                    feat, r, p, name = t[0], t[1], t[2], t[3]
                    pv = f", p={p:.4f}" if p is not None and p == p else ""
                    lines.append(f"  {feat}: r={r:.3f}{pv} ({name})")
                elif len(t) >= 2:
                    lines.append(f"  {t[0]}: {t[1]}")
        elif "feature_correlations" in stats:
            for t in stats["feature_correlations"]:
                if len(t) >= 2:
                    lines.append(f"  {t[0]}: |r|={t[1]:.3f}")
        if "vif" in stats:
            vifs = stats["vif"]
            if isinstance(vifs, list):
                bits = []
                for c, v in vifs[:15]:
                    if isinstance(v, (int, float)) and v == v:
                        bits.append(f"{c}={v:.1f}")
                if bits:
                    lines.append("  VIF: " + "; ".join(bits))
        for k in ("shapiro_p", "shapiro_stat", "max_leverage", "max_cooks", "n_high_leverage", "n_high_cooks"):
            if k in stats and stats[k] is not None:
                v = stats[k]
                if isinstance(v, (int, float)):
                    lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        if "residual_vs_predicted_corr" in stats and stats["residual_vs_predicted_corr"] is not None:
            lines.append(f"  residual_vs_predicted_corr: {stats['residual_vs_predicted_corr']:.4f}")
        if lines:
            parts.append("Stats: " + " | ".join(lines))

    figures = result.get("figures", [])
    corr_tests = stats.get("correlation_tests", []) if isinstance(stats, dict) else []
    corr_feats = [x[0] for x in corr_tests] if corr_tests else []

    for idx, (fig_type, fig_data) in enumerate(figures):
        n = idx + 1
        if fig_type == "table":
            if hasattr(fig_data, "head") and hasattr(fig_data, "to_csv"):
                df = fig_data
                try:
                    head = df.head(_MAX_TABLE_ROWS)
                    raw = head.to_markdown(index=False)
                except Exception:
                    try:
                        raw = head.to_csv(index=False)
                    except Exception:
                        raw = head.to_string()
                if len(raw) > _MAX_TABLE_CHARS:
                    raw = raw[:_MAX_TABLE_CHARS] + "\n..."
                parts.append(f"Table {n}:\n{raw}")
            else:
                parts.append(f"Table {n}: [tabular data]")
        else:
            desc = ""
            if fig_type == "plotly" and hasattr(fig_data, "layout") and hasattr(fig_data.layout, "title"):
                t = getattr(fig_data.layout.title, "text", None)
                if t:
                    desc = t
            if not desc and idx < len(corr_feats):
                feat = corr_feats[idx]
                for t in (corr_tests or []):
                    if len(t) >= 4 and t[0] == feat:
                        desc = f"{feat}: r={t[1]:.3f}, p={t[2]:.4f} ({t[3]})"
                        break
            if not desc:
                desc = "scatter" if "scatter" in str(action_id).lower() or "linearity" in action_id else "plot"
            parts.append(f"Figure {n}: {desc}.")

    return "\n\n".join(parts) if parts else ""


def build_llm_context(
    plot_type: str,
    stats_summary: str,
    model_name: Optional[str] = None,
    where: Optional[str] = None,
    existing: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
    feature_names: Optional[List[str]] = None,
    sample_size: Optional[int] = None,
    task_type: Optional[str] = None,
    data_domain_hint: Optional[str] = None,
) -> str:
    """Build a rich context string for the LLM (plot type, stats, model, metrics, n, domain, etc.)."""
    parts = [f"Plot/analysis: {plot_type}.", f"Key stats: {stats_summary}."]
    if model_name:
        parts.append(f"Model: {model_name}.")
    if where:
        parts.append(f"Where: {where}.")
    if task_type:
        parts.append(f"Task: {task_type}.")
    if sample_size is not None:
        parts.append(f"Sample size: n={sample_size}.")
    if metrics:
        kv = "; ".join(f"{k}={v}" for k, v in list(metrics.items())[:6])
        parts.append(f"Metrics: {kv}.")
    if feature_names:
        feats = ", ".join(str(x) for x in feature_names[:8])
        if len(feature_names) > 8:
            feats += ", …"
        parts.append(f"Features: {feats}.")
    domain = data_domain_hint or (_infer_domain_hint(feature_names) if feature_names else "")
    if domain:
        parts.append(f"Domain: {domain}.")
    if existing:
        parts.append(f"Existing summary (use only as background; do not simply paraphrase): {existing}.")
    return " ".join(parts)


def render_interpretation_with_llm_button(
    context: str,
    key: str,
    result_session_key: Optional[str] = None,
) -> None:
    """
    Render "Interpret these results using an LLM" button and optional AI output.
    context: rich string (plot type, model, stats, where it appears) for the LLM to interpret.
    Uses ensure_ollama_running + enhance_with_ollama. Model from session_state or default.
    Includes optional "Add your own context" text area; when non-empty, appended to context.
    """
    import streamlit as st
    from ml.llm_local import (
        ensure_ollama_running,
        enhance_with_ollama,
        DEFAULT_OLLAMA_URL,
        DEFAULT_OLLAMA_MODEL,
    )

    sk = result_session_key or f"llm_result_{key}"
    user_ctx_key = f"{key}_user_context"
    model = st.session_state.get("ollama_model", DEFAULT_OLLAMA_MODEL)

    with st.expander("Add your own context (optional)", expanded=False):
        st.caption("Optional extra context for the LLM (e.g. focus on clinical implications).")
        st.text_area(
            "User context",
            key=user_ctx_key,
            placeholder="E.g. focus on clinical implications, or what you care about most.",
            label_visibility="collapsed",
        )

    if st.button("Interpret these results using an LLM", key=key):
        ok = ensure_ollama_running(DEFAULT_OLLAMA_URL)
        if not ok:
            st.session_state[sk] = "__unavailable__"
        else:
            ctx = context or ""
            user_txt = (st.session_state.get(user_ctx_key) or "").strip()
            if user_txt:
                ctx = f"{ctx} User-provided context (consider this when interpreting): {user_txt}"
            out = enhance_with_ollama(DEFAULT_OLLAMA_URL, ctx, model=model)
            st.session_state[sk] = out or "__error__"
        st.rerun()

    res = st.session_state.get(sk)
    if res == "__unavailable__":
        st.caption(
            "To use this feature: (1) Install Ollama from [ollama.ai](https://ollama.ai). "
            "(2) Run `ollama serve` in a terminal (or ensure it is already running). "
            f"(3) If needed, pull a model (e.g. `ollama run {DEFAULT_OLLAMA_MODEL}`)."
        )
    elif res == "__error__":
        st.caption(
            f"Could not get interpretation from Ollama. Check that a model is available (e.g. `ollama run {DEFAULT_OLLAMA_MODEL}`)."
        )
    elif res:
        st.markdown(f"**Interpretation (LLM):** {res}")
