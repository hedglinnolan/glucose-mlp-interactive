"""
LaTeX report generator.

Generates a complete LaTeX manuscript template populated with actual results
from the modeling workflow. Ready to compile with pdflatex.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    if not isinstance(text, str):
        text = str(text)
    chars = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
        '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
    }
    for char, replacement in chars.items():
        text = text.replace(char, replacement)
    return text


def _metrics_to_latex_table(
    model_results: Dict[str, Dict],
    task_type: str = "regression",
    bootstrap_results: Optional[Dict] = None,
) -> str:
    """Generate a LaTeX metrics comparison table."""
    if task_type == "regression":
        metric_names = ["RMSE", "MAE", "R2", "MedianAE"]
        caption = "Model performance on the held-out test set (regression metrics)."
    else:
        metric_names = ["Accuracy", "F1", "AUC"]
        caption = "Model performance on the held-out test set (classification metrics)."

    # Determine which metrics are actually present
    all_metrics = set()
    for res in model_results.values():
        all_metrics.update(res.get("metrics", {}).keys())
    metric_names = [m for m in metric_names if m in all_metrics]

    if not metric_names:
        return ""

    n_cols = 1 + len(metric_names)
    col_spec = "l" + "c" * len(metric_names)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(r"\label{tab:model_performance}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = "Model & " + " & ".join(metric_names) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for name, res in model_results.items():
        metrics = res.get("metrics", {})
        cis = {}
        if bootstrap_results and name in bootstrap_results:
            cis = bootstrap_results[name]

        cells = [_escape_latex(name.upper())]
        for m in metric_names:
            val = metrics.get(m)
            ci = cis.get(m)
            if val is not None:
                if ci and hasattr(ci, 'ci_lower') and hasattr(ci, 'ci_upper'):
                    cells.append(f"{val:.4f} [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]")
                else:
                    cells.append(f"{val:.4f}")
            else:
                cells.append("---")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def _table1_to_latex(table1_df: pd.DataFrame) -> str:
    """Convert Table 1 DataFrame to LaTeX."""
    if table1_df is None or table1_df.empty:
        return ""

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Characteristics of the study population.}")
    lines.append(r"\label{tab:table1}")

    n_cols = len(table1_df.columns) + 1  # +1 for index
    col_spec = "l" + "c" * len(table1_df.columns)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header
    header = "Characteristic & " + " & ".join(_escape_latex(str(c)) for c in table1_df.columns) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Rows
    for idx, row in table1_df.iterrows():
        cells = [_escape_latex(str(idx))]
        for val in row.values:
            cells.append(_escape_latex(str(val)) if val else "")
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_latex_report(
    title: str = "Prediction Model Development and Validation",
    authors: str = "[Author Names]",
    affiliation: str = "[Institution]",
    abstract: str = "[ABSTRACT PLACEHOLDER]",
    methods_section: str = "",
    table1_df: Optional[pd.DataFrame] = None,
    model_results: Optional[Dict[str, Dict]] = None,
    bootstrap_results: Optional[Dict] = None,
    task_type: str = "regression",
    feature_names: Optional[List[str]] = None,
    target_name: str = "outcome",
    n_total: int = 0,
    n_train: int = 0,
    n_val: int = 0,
    n_test: int = 0,
    tripod_checklist: Optional[pd.DataFrame] = None,
    data_config: Optional[Dict] = None,
    calibration_text: str = "",
    limitations: str = "[Discuss limitations here]",
) -> str:
    """Generate a complete LaTeX manuscript template.

    Returns compilable LaTeX source populated with actual results.
    """
    sections = []

    # ── Preamble ──
    sections.append(r"""\documentclass[12pt, a4paper]{article}

% ── Packages ──
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath, amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\usepackage{natbib}
\usepackage{float}
\usepackage{setspace}
\usepackage{caption}

\doublespacing

% ── Title ──""")

    sections.append(f"\\title{{{_escape_latex(title)}}}")
    sections.append(f"\\author{{{_escape_latex(authors)} \\\\ \\small{{{_escape_latex(affiliation)}}}}}")
    sections.append(r"\date{\today}")
    sections.append("")
    sections.append(r"\begin{document}")
    sections.append(r"\maketitle")

    # ── Abstract ──
    sections.append(r"""
\begin{abstract}""")
    sections.append(f"{_escape_latex(abstract)}")
    sections.append(r"""\end{abstract}

\clearpage
""")

    # ── Introduction ──
    sections.append(r"""\section{Introduction}

[PLACEHOLDER: Provide background on the clinical/research context and rationale for developing this prediction model. Cite relevant prior work.]

\subsection{Objectives}
[PLACEHOLDER: State the specific objectives of this study, including whether you are developing, validating, or both.]

""")

    # ── Methods ──
    sections.append(r"\section{Methods}")
    sections.append("")

    if methods_section:
        # Convert markdown headers to LaTeX subsections
        latex_methods = methods_section.replace("### ", "\\subsection{").replace("\n\n", "}\n\n\\noindent ")
        # Clean up any trailing issues
        latex_methods = _escape_latex(latex_methods)
        sections.append(latex_methods)
    else:
        sections.append(r"""
\subsection{Study Design and Participants}
[PLACEHOLDER: Describe the study design, data source, eligibility criteria, and key dates.]

\subsection{Outcome Definition}
""")
        sections.append(f"The outcome variable was {_escape_latex(target_name)}.")
        sections.append(r"""
\subsection{Predictor Variables}""")
        if feature_names:
            if len(feature_names) <= 15:
                feat_list = ", ".join(_escape_latex(f) for f in feature_names)
                sections.append(f"The following {len(feature_names)} predictor variables were included: {feat_list}.")
            else:
                sections.append(f"A total of {len(feature_names)} predictor variables were included (see Supplementary Table S1).")

        sections.append(r"""
\subsection{Missing Data}
[PLACEHOLDER: Describe how missing data were handled, including the mechanism (MCAR/MAR/MNAR) and imputation strategy.]

\subsection{Model Development}
[PLACEHOLDER: Describe preprocessing, model selection, and internal validation strategy.]
""")
        if n_total > 0:
            sections.append(f"Data were split into training (n={n_train:,}), validation (n={n_val:,}), and test (n={n_test:,}) sets.")

        sections.append(r"""
\subsection{Performance Evaluation}
Model performance was assessed using [METRICS] with 95\% confidence intervals computed via 1,000 BCa bootstrap resamples.
""")

    # ── Results ──
    sections.append(r"""
\section{Results}

\subsection{Study Population}""")

    if n_total > 0:
        sections.append(f"A total of {n_total:,} participants were included in the analysis.")

    # Table 1
    if table1_df is not None and not table1_df.empty:
        sections.append(_table1_to_latex(table1_df))
    else:
        sections.append(r"[INSERT TABLE 1: Characteristics of the study population]")

    sections.append(r"""
\subsection{Model Performance}""")

    # Metrics table
    if model_results:
        sections.append(_metrics_to_latex_table(model_results, task_type, bootstrap_results))
    else:
        sections.append(r"[INSERT TABLE: Model performance metrics with 95\% CIs]")

    # Calibration
    if calibration_text:
        sections.append(r"\subsection{Calibration}")
        sections.append(_escape_latex(calibration_text))
    else:
        sections.append(r"""
\subsection{Calibration}
[PLACEHOLDER: Report calibration results — Brier score, ECE, calibration slope/intercept. Include calibration plot as a figure.]
""")

    # ── Discussion ──
    sections.append(r"""
\section{Discussion}

\subsection{Principal Findings}
[PLACEHOLDER: Summarize the main results in context of the study objectives.]

\subsection{Comparison with Prior Work}
[PLACEHOLDER: Compare your results with existing literature.]

\subsection{Clinical Implications}
[PLACEHOLDER: Discuss practical implications for clinical decision-making or research.]

\subsection{Strengths and Limitations}
""")
    sections.append(_escape_latex(limitations))

    sections.append(r"""
\subsection{Conclusion}
[PLACEHOLDER: State the main conclusion and its implications.]

""")

    # ── References ──
    sections.append(r"""
\section*{References}
\begin{enumerate}
\item [PLACEHOLDER: Add references in journal format]
\item Collins GS, et al. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD). BMJ. 2015;350:g7594.
\end{enumerate}
""")

    # ── Supplementary ──
    sections.append(r"""
\clearpage
\appendix
\section{Supplementary Material}

\subsection{TRIPOD Checklist}
[See exported TRIPOD checklist CSV/PDF]

\subsection{Reproducibility}
This analysis was conducted using Tabular ML Lab (Python). Full reproducibility manifest including software versions, random seeds, and data hashes is available in the exported analysis package.

""")

    sections.append(r"\end{document}")

    return "\n".join(sections)
