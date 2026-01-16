"""
Model Selection Coach: Recommends model families based on dataset characteristics.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from ml.eda_recommender import DatasetSignals


# Canonical group display names
GROUP_DISPLAY_NAMES = {
    'Linear': 'Linear Models',
    'Trees': 'Tree-Based Models',
    'Boosting': 'Gradient Boosting',
    'Distance': 'Distance-Based Models',
    'Margin': 'Margin-Based Models (SVM)',
    'Probabilistic': 'Probabilistic Models',
    'Neural Net': 'Neural Networks'
}


@dataclass
class CoachRecommendation:
    """A recommendation from the model selection coach."""
    group: str  # Canonical group key (e.g., "Linear", "Trees")
    recommended_models: List[str]  # Model keys from registry
    why: List[str]  # Plain language reasons with numbers
    when_not_to_use: List[str]  # Short caveats
    suggested_preprocessing: List[str]  # e.g., "standardize numeric features", "consider PCA"
    priority: int  # Lower = higher priority
    readiness_checks: List[str] = field(default_factory=list)  # Prerequisites that should be completed first
    
    @property
    def display_name(self) -> str:
        """Get the display name for this group."""
        return GROUP_DISPLAY_NAMES.get(self.group, self.group)


def coach_recommendations(
    signals: DatasetSignals,
    optional_results: Optional[Dict[str, Any]] = None,
    eda_insights: Optional[List[Dict[str, Any]]] = None
) -> List[CoachRecommendation]:
    """
    Generate model selection recommendations based on dataset signals.
    
    Args:
        signals: DatasetSignals object from EDA recommender
        optional_results: Optional dict with EDA results (e.g., quick probe baselines)
        
    Returns:
        List of CoachRecommendation objects, sorted by priority
    """
    recommendations = []
    task_type = signals.task_type_final
    n_rows = signals.n_rows
    n_features = len(signals.numeric_cols) + len(signals.categorical_cols)
    p_n_ratio = n_features / n_rows if n_rows > 0 else 0
    
    # Check readiness (prerequisites)
    readiness_checks = []
    if not signals.target_stats:
        readiness_checks.append("Run 'Target Profile' analysis in EDA to understand target distribution")
    if len(signals.high_missing_cols) > 0 and not any('missingness' in str(i.get('id', '')) for i in (eda_insights or [])):
        readiness_checks.append("Run 'Missingness Scan' in EDA to check for informative missingness")
    if signals.unit_sanity_flags and not any('unit' in str(i.get('id', '')) for i in (eda_insights or [])):
        readiness_checks.append("Run 'Physiologic Plausibility Check' in EDA to verify units")
    
    # Always recommend a simple baseline first
    # For regression: include Huber if outliers are high, otherwise just GLM/Ridge
    if task_type == 'regression':
        outlier_rate = signals.target_stats.get('outlier_rate', 0)
        recommended_models = ['glm', 'ridge']
        why_text = [
            "Start with interpretable linear models for baseline performance",
            "Linear models provide coefficient interpretation and fast training"
        ]
        
        # Add Huber if outliers are significant
        if outlier_rate > 0.1:
            recommended_models.append('huber')
            why_text.append(f"Outlier rate: {outlier_rate:.1%} - include Huber for robust loss")
        
        recommendations.append(CoachRecommendation(
            group='Linear',
            recommended_models=recommended_models,
            why=why_text,
            when_not_to_use=[
                "If data shows strong nonlinear patterns (check quick probe baselines)",
                "If outliers are data errors, consider removing them first"
            ],
            suggested_preprocessing=['standardize numeric features'],
            priority=1,
            readiness_checks=readiness_checks[:2] if readiness_checks else []
        ))
    else:  # classification
        recommendations.append(CoachRecommendation(
            group='Linear',
            recommended_models=['logreg', 'glm'],
            why=[
                "Start with interpretable logistic regression for baseline",
                "Provides probability estimates and coefficient interpretation"
            ],
            when_not_to_use=[
                "If class imbalance > 3:1 (consider class weighting or boosting)",
                "If nonlinear decision boundaries needed"
            ],
            suggested_preprocessing=['standardize numeric features'],
            priority=1,
            readiness_checks=readiness_checks[:2] if readiness_checks else []
        ))
    
    # Class imbalance (classification)
    if task_type == 'classification':
        imbalance_ratio = signals.target_stats.get('class_imbalance_ratio', 1.0)
        if imbalance_ratio > 3.0:
            recommendations.append(CoachRecommendation(
                group='Boosting',
                recommended_models=['histgb_clf', 'rf'],
                why=[
                    f"Class imbalance ratio: {imbalance_ratio:.2f} - boosting/trees handle imbalance well",
                    "Use PR-AUC (Precision-Recall Area Under Curve) instead of accuracy",
                    "Consider class weighting or threshold tuning"
                ],
                when_not_to_use=[
                    "If dataset is very small (< 100 samples)"
                ],
                suggested_preprocessing=[],
                priority=2
            ))
    
    # High collinearity / multicollinearity
    max_corr = signals.collinearity_summary.get('max_corr', 0)
    if max_corr > 0.85:
        recommendations.append(CoachRecommendation(
            group='Linear',
            recommended_models=['ridge', 'elasticnet', 'lasso'],
            why=[
                f"Maximum correlation: {max_corr:.2f} - multicollinearity detected",
                "Regularized linear models (Ridge/Lasso/ElasticNet) stabilize coefficients",
                "Unregularized GLM coefficients may be unstable with high collinearity"
            ],
            when_not_to_use=[
                "If you need exact feature selection (Lasso may zero out features)"
            ],
            suggested_preprocessing=['standardize numeric features', 'consider PCA for dimensionality reduction'],
            priority=2
        ))
    
    # High dimensionality (p/n ratio)
    if p_n_ratio > 0.5:
        recommendations.append(CoachRecommendation(
            group='Linear',
            recommended_models=['ridge', 'lasso', 'elasticnet'],
            why=[
                f"Feature-to-sample ratio: {p_n_ratio:.2f} - high dimensionality",
                "Regularized models prevent overfitting in high-dimensional settings",
                "PCA (Principal Component Analysis) may help reduce dimensionality"
            ],
            when_not_to_use=[
                "If you need to preserve all original features"
            ],
            suggested_preprocessing=['standardize numeric features', 'consider PCA'],
            priority=2
        ))
    
    # Many missing values
    high_missing_count = len(signals.high_missing_cols)
    if high_missing_count > 0:
        max_missing = max(signals.missing_rate_by_col.values()) if signals.missing_rate_by_col else 0
        if max_missing > 0.1:
            recommendations.append(CoachRecommendation(
                group='Trees',
                recommended_models=['rf', 'extratrees_reg' if task_type == 'regression' else 'extratrees_clf'],
                why=[
                    f"{high_missing_count} columns with >5% missing, max: {max_missing:.1%}",
                    "Tree-based models handle missing values natively",
                    "Consider adding missingness indicator features"
                ],
                when_not_to_use=[
                    "If missingness is informative (add indicators for all models)"
                ],
                suggested_preprocessing=['add missingness indicators'],
                priority=3
            ))
    
    # Moderate dataset size + low feature count -> kNN
    if n_features <= 30 and 100 <= n_rows <= 10000:
        recommendations.append(CoachRecommendation(
            group='Distance',
            recommended_models=['knn_reg' if task_type == 'regression' else 'knn_clf'],
            why=[
                f"Moderate dataset ({n_rows:,} rows, {n_features} features) - kNN is a good baseline",
                "kNN is non-parametric and can capture local patterns"
            ],
            when_not_to_use=[
                "If dataset is very large (>10k samples) - kNN becomes slow",
                "If features are not scaled - kNN is sensitive to scaling"
            ],
            suggested_preprocessing=['standardize numeric features (required for kNN)'],
            priority=4
        ))
    
    # Use EDA insights if available
    if eda_insights:
        for insight in eda_insights:
            if insight.get('id') == 'target_outliers' and insight.get('category') == 'target_characteristics':
                # Already handled above, but can strengthen recommendation
                pass
            elif insight.get('id') == 'class_imbalance' and insight.get('category') == 'target_characteristics':
                # Already handled above
                pass
            elif insight.get('id') == 'collinearity' and insight.get('category') == 'feature_relationships':
                # Already handled above, but can strengthen
                pass
    
    # Check quick probe results if available
    if optional_results and 'quick_probe_baselines' in optional_results:
        probe_results = optional_results['quick_probe_baselines']
        # If RF/Boosting significantly outperform linear, recommend trees/boosting
        if 'findings' in probe_results:
            for finding in probe_results['findings']:
                if 'Random Forest' in finding or 'RF' in finding:
                    if 'outperforms' in finding.lower() or 'better' in finding.lower():
                        recommendations.append(CoachRecommendation(
                            group='Trees',
                            recommended_models=['rf', 'extratrees_reg' if task_type == 'regression' else 'extratrees_clf'],
                            why=[
                                "Quick probe suggests nonlinear patterns",
                                "Tree models capture interactions and nonlinearities automatically"
                            ],
                            when_not_to_use=[
                                "If interpretability is critical"
                            ],
                            suggested_preprocessing=[],
                            priority=3
                        ))
                        break
    
    # Always include neural network as advanced option
    if n_rows > 1000 and n_features > 5:
        recommendations.append(CoachRecommendation(
            group='Neural Net',
            recommended_models=['nn'],
            why=[
                f"Dataset size ({n_rows:,} rows) sufficient for neural networks",
                "Neural networks can capture complex nonlinear patterns"
            ],
            when_not_to_use=[
                "If dataset is very small (<500 samples)",
                "If interpretability is required"
            ],
            suggested_preprocessing=['standardize numeric features'],
            priority=5
        ))
    
    # Merge recommendations by group to avoid duplicates
    merged = _merge_recommendations_by_group(recommendations)
    
    # Sort by priority
    merged.sort(key=lambda x: x.priority)
    
    return merged


def _merge_recommendations_by_group(recommendations: List[CoachRecommendation]) -> List[CoachRecommendation]:
    """
    Merge recommendations with the same group into a single recommendation.
    Combines models, reasons, caveats, and preprocessing suggestions.
    Uses the lowest (best) priority from merged recommendations.
    """
    group_map: Dict[str, CoachRecommendation] = {}
    
    for rec in recommendations:
        if rec.group not in group_map:
            # First recommendation for this group - copy it
            group_map[rec.group] = CoachRecommendation(
                group=rec.group,
                recommended_models=list(rec.recommended_models),
                why=list(rec.why),
                when_not_to_use=list(rec.when_not_to_use),
                suggested_preprocessing=list(rec.suggested_preprocessing),
                priority=rec.priority,
                readiness_checks=list(rec.readiness_checks)
            )
        else:
            # Merge into existing recommendation
            existing = group_map[rec.group]
            
            # Add unique models
            for model in rec.recommended_models:
                if model not in existing.recommended_models:
                    existing.recommended_models.append(model)
            
            # Add unique reasons
            for reason in rec.why:
                if reason not in existing.why:
                    existing.why.append(reason)
            
            # Add unique caveats
            for caveat in rec.when_not_to_use:
                if caveat not in existing.when_not_to_use:
                    existing.when_not_to_use.append(caveat)
            
            # Add unique preprocessing suggestions
            for prep in rec.suggested_preprocessing:
                if prep not in existing.suggested_preprocessing:
                    existing.suggested_preprocessing.append(prep)
            
            # Add unique readiness checks
            for check in rec.readiness_checks:
                if check not in existing.readiness_checks:
                    existing.readiness_checks.append(check)
            
            # Use the lowest (best) priority
            existing.priority = min(existing.priority, rec.priority)
    
    return list(group_map.values())
