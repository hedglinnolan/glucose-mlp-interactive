"""
Reusable UI for "Interpret these results using an LLM" (Ollama).
Renders button + optional AI result; auto-starts Ollama if not running.
"""
from __future__ import annotations

from typing import Optional, Dict, Any, List


def _infer_domain_hint(feature_names: Optional[List[str]] = None) -> str:
    """Infer a simple domain hint (e.g. 'clinical') from feature/column names."""
    if not feature_names:
        return ""
    clinical_like = {"glucose", "bmi", "age", "weight", "height", "bp", "hdl", "ldl", "hb", "waist", "hip"}
    names_lower = " ".join(str(x).lower() for x in feature_names)
    if any(k in names_lower for k in clinical_like):
        return "clinical"
    return ""


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
    """
    import streamlit as st
    from ml.llm_local import (
        ensure_ollama_running,
        enhance_with_ollama,
        DEFAULT_OLLAMA_URL,
        DEFAULT_OLLAMA_MODEL,
    )

    sk = result_session_key or f"llm_result_{key}"
    model = st.session_state.get("ollama_model", DEFAULT_OLLAMA_MODEL)
    if st.button("Interpret these results using an LLM", key=key):
        ok = ensure_ollama_running(DEFAULT_OLLAMA_URL)
        if not ok:
            st.session_state[sk] = "__unavailable__"
        else:
            out = enhance_with_ollama(DEFAULT_OLLAMA_URL, context or "", model=model)
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
