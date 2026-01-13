"""
EDA Analysis Actions - Runnable functions for EDA recommendations.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
import streamlit as st

from ml.eval import calculate_regression_metrics, calculate_classification_metrics
from ml.clinical_units import infer_unit


def plausibility_check(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """
    Check physiologic plausibility for common clinical columns with unit inference.
    
    Returns:
        Dict with 'findings', 'warnings', 'figures'
    """
    findings = []
    warnings = []
    figures = []
    
    # Get unit overrides from session state
    unit_overrides = session_state.get('unit_overrides', {})
    
    checked_cols = []
    out_of_range = []
    unit_inferences = []
    
    for col in df.columns:
        col_lower = col.lower()
        # Check if this matches a known clinical variable pattern
        is_clinical = any(
            pattern in col_lower 
            for pattern in ['weight', 'height', 'waist', 'glucose', 'cholesterol', 
                          'triglyceride', 'bp_sys', 'bp_di', 'bmi', 'hba1c', 'kcal']
        )
        
        if is_clinical and col in signals.numeric_cols:
            checked_cols.append(col)
            col_data = df[col].dropna()
            
            if len(col_data) > 0:
                # Infer unit (or use override)
                if col in unit_overrides:
                    inferred_unit_info = {
                        'inferred_unit': unit_overrides[col],
                        'canonical_unit': 'unknown',
                        'confidence': 'override',
                        'explanation': f'User override: {unit_overrides[col]}',
                        'conversion_factor': 1.0
                    }
                else:
                    inferred_unit_info = infer_unit(col, col_data)
                
                # Build unit inference row with threshold bands if available
                unit_row = {
                    'Column': col,
                    'Inferred Unit': inferred_unit_info.get('inferred_unit', 'Unknown'),
                    'Canonical Unit': inferred_unit_info.get('canonical_unit', 'N/A'),
                    'Confidence': inferred_unit_info.get('confidence', 'low'),
                    'Explanation': inferred_unit_info.get('explanation', '')
                }
                
                # Add fasting note if applicable
                if inferred_unit_info.get('fasting_note'):
                    unit_row['Note'] = 'Fasting assumption (reference ranges assume fasting state)'
                else:
                    unit_row['Note'] = ''
                
                unit_inferences.append(unit_row)
                
                # If we have a canonical unit and conversion, check ranges
                if inferred_unit_info.get('canonical_unit') and inferred_unit_info.get('conversion_factor'):
                    # Get plausible range from clinical_units
                    from ml.clinical_units import CLINICAL_VARIABLES
                    matched_var = None
                    for var_name in CLINICAL_VARIABLES.keys():
                        if var_name in col_lower:
                            matched_var = var_name
                            break
                    
                    if matched_var:
                        var_config = CLINICAL_VARIABLES[matched_var]
                        thresholds = var_config.get('thresholds', {}).get(inferred_unit_info['inferred_unit'])
                        fasting_note = var_config.get('fasting_note', False)
                        
                        # Find the hypothesis that matches inferred unit
                        for unit_name, conv_factor, (min_val, max_val) in var_config['hypotheses']:
                            if unit_name == inferred_unit_info['inferred_unit']:
                                # Convert to canonical and check
                                converted = col_data * conv_factor
                                
                                # Classify values into threshold bands if available
                                threshold_bands = {}
                                if thresholds:
                                    for band_name, (band_min, band_max) in thresholds.items():
                                        if band_max is None:
                                            # >= threshold
                                            count = (converted >= band_min).sum()
                                        else:
                                            count = ((converted >= band_min) & (converted < band_max)).sum()
                                        threshold_bands[band_name] = count
                                
                                # Check against overall plausible range
                                below_min = (converted < min_val).sum()
                                above_max = (converted > max_val).sum()
                                total_out = below_min + above_max
                                out_rate = total_out / len(col_data)
                                
                                # Build range description with threshold bands
                                range_desc = f"{min_val}-{max_val} {var_config['canonical_unit']}"
                                if thresholds:
                                    band_names = {
                                        'normal': 'Normal',
                                        'prediabetes': 'Prediabetes',
                                        'diabetes': 'Diabetes',
                                        'borderline_high': 'Borderline High',
                                        'high': 'High',
                                        'very_high': 'Very High'
                                    }
                                    band_summary = []
                                    for band_name, count in threshold_bands.items():
                                        pct = count / len(col_data)
                                        if pct > 0:
                                            band_summary.append(f"{band_names.get(band_name, band_name)}: {pct:.1%}")
                                    if band_summary:
                                        range_desc += f" ({', '.join(band_summary)})"
                                
                                if total_out > 0 or (thresholds and any(v > 0 for v in threshold_bands.values() if 'normal' not in str(v))):
                                    out_of_range.append({
                                        'Column': col,
                                        'Inferred Unit': inferred_unit_info['inferred_unit'],
                                        'Min (canonical)': f"{converted.min():.1f}",
                                        'Max (canonical)': f"{converted.max():.1f}",
                                        'Reference Range': range_desc,
                                        'Out of Range %': f"{out_rate:.1%}" if total_out > 0 else "0%"
                                    })
                                    if out_rate > 0.05:
                                        fasting_text = " (fasting assumption)" if fasting_note else ""
                                        warnings.append(
                                            f"{col}: {out_rate:.1%} values outside typical reference range "
                                            f"({range_desc}){fasting_text} after converting from {inferred_unit_info['inferred_unit']}"
                                        )
                                break
    
    findings.append(f"Checked {len(checked_cols)} columns with medical/nutritional patterns")
    
    if len(unit_inferences) > 0:
        unit_df = pd.DataFrame(unit_inferences)
        figures.append(('table', unit_df))
        findings.append(f"Inferred units for {len(unit_inferences)} clinical variables")
    
    if len(out_of_range) > 0:
        findings.append(f"Found {len(out_of_range)} columns with out-of-range values")
        out_df = pd.DataFrame(out_of_range)
        figures.append(('table', out_df))
    else:
        findings.append("All checked columns within plausible ranges")
    
    # Add unit sanity flags from signals
    if signals.unit_sanity_flags:
        warnings.extend(signals.unit_sanity_flags)
        findings.append(f"Found {len(signals.unit_sanity_flags)} potential unit mismatch flags")
    
    # Add note about unit overrides
    if unit_overrides:
        findings.append(f"Using {len(unit_overrides)} user-specified unit overrides")
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures
    }


def missingness_scan(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Analyze missingness patterns and association with target."""
    findings = []
    warnings = []
    figures = []
    
    if not target or target not in df.columns:
        return {
            'findings': ["Target not available for missingness analysis"],
            'warnings': [],
            'figures': []
        }
    
    # Missingness bar chart
    missing_df = pd.DataFrame({
        'Column': list(signals.missing_rate_by_col.keys()),
        'Missing Rate': list(signals.missing_rate_by_col.values())
    })
    missing_df = missing_df[missing_df['Missing Rate'] > 0].sort_values('Missing Rate', ascending=False)
    
    if len(missing_df) > 0:
        fig = px.bar(
            missing_df.head(20),
            x='Missing Rate',
            y='Column',
            orientation='h',
            title='Missingness by Column (Top 20)'
        )
        figures.append(('plotly', fig))
        findings.append(f"{len(missing_df)} columns have missing values")
    
    # Missingness vs target association
    if signals.task_type_final == 'regression':
        # Compare target mean for missing vs non-missing
        associations = []
        for col in signals.high_missing_cols[:10]:  # Limit to top 10
            if col in df.columns and col != target:
                missing_mask = df[col].isnull()
                if missing_mask.sum() > 0 and (~missing_mask).sum() > 0:
                    target_missing = df.loc[missing_mask, target].mean()
                    target_nonmissing = df.loc[~missing_mask, target].mean()
                    diff = abs(target_missing - target_nonmissing)
                    associations.append({
                        'Column': col,
                        'Target Mean (Missing)': target_missing,
                        'Target Mean (Non-Missing)': target_nonmissing,
                        'Difference': diff
                    })
        
        if associations:
            assoc_df = pd.DataFrame(associations).sort_values('Difference', ascending=False)
            figures.append(('table', assoc_df))
            findings.append("Missingness may be informative (associated with target)")
    elif signals.task_type_final == 'classification':
        # Compare class proportions
        associations = []
        for col in signals.high_missing_cols[:10]:
            if col in df.columns and col != target:
                missing_mask = df[col].isnull()
                if missing_mask.sum() > 0:
                    missing_class_prop = df.loc[missing_mask, target].value_counts(normalize=True)
                    nonmissing_class_prop = df.loc[~missing_mask, target].value_counts(normalize=True)
                    # Simple difference metric
                    if len(missing_class_prop) > 0 and len(nonmissing_class_prop) > 0:
                        max_diff = abs(missing_class_prop - nonmissing_class_prop).max()
                        associations.append({
                            'Column': col,
                            'Max Class Prop Difference': max_diff
                        })
        
        if associations:
            assoc_df = pd.DataFrame(associations).sort_values('Max Class Prop Difference', ascending=False)
            figures.append(('table', assoc_df))
            findings.append("Missingness may be informative (associated with class)")
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures
    }


def cohort_split_guidance(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Provide guidance on cohort structure and split strategy."""
    findings = []
    warnings = []
    figures = []
    
    findings.append(f"Cohort type: {signals.cohort_type_final}")
    findings.append(f"Entity ID column: {signals.entity_id_final or 'Not specified'}")
    
    if signals.entity_id_final and signals.entity_id_final in df.columns:
        entity_counts = df[signals.entity_id_final].value_counts()
        median_rows = entity_counts.median()
        mean_rows = entity_counts.mean()
        findings.append(f"Median rows per entity: {median_rows:.1f}")
        findings.append(f"Mean rows per entity: {mean_rows:.1f}")
        findings.append(f"Total unique entities: {len(entity_counts)}")
        
        # Distribution plot
        fig = px.histogram(
            x=entity_counts.values,
            nbins=20,
            title='Distribution of Rows per Entity',
            labels={'x': 'Rows per Entity', 'y': 'Count'}
        )
        figures.append(('plotly', fig))
        
        warnings.append("⚠️ Must use group-based splitting to prevent data leakage")
        warnings.append("Random splits will leak information across train/test")
    else:
        warnings.append("⚠️ Entity ID not specified - cannot use group-based splitting")
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures
    }


def target_profile(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Profile target distribution (regression or classification)."""
    findings = []
    warnings = []
    figures = []
    
    if not target or target not in df.columns:
        return {
            'findings': ["Target not available"],
            'warnings': [],
            'figures': []
        }
    
    target_series = df[target].dropna()
    
    if signals.task_type_final == 'regression':
        # Histogram
        fig1 = px.histogram(
            target_series,
            nbins=30,
            title=f'Target Distribution: {target}',
            labels={'value': target, 'count': 'Count'}
        )
        figures.append(('plotly', fig1))
        
        # Log histogram if all positive
        if (target_series > 0).all():
            log_target = np.log1p(target_series)
            fig2 = px.histogram(
                log_target,
                nbins=30,
                title=f'Log-Transformed Target Distribution: {target}',
                labels={'value': f'log({target})', 'count': 'Count'}
            )
            figures.append(('plotly', fig2))
            findings.append("Target is positive - log transform may help")
        
        # Outlier summary
        outlier_rate = signals.target_stats.get('outlier_rate', 0)
        skew = signals.target_stats.get('skew', 0)
        findings.append(f"Skewness: {skew:.2f}")
        findings.append(f"Outlier rate: {outlier_rate:.1%}")
        
        if abs(skew) > 1:
            warnings.append("High skewness - consider log transform or robust loss")
        if outlier_rate > 0.05:
            warnings.append(f"High outlier rate ({outlier_rate:.1%}) - consider robust loss")
    
    elif signals.task_type_final == 'classification':
        # Class counts
        class_counts = target_series.value_counts().sort_index()
        fig = px.bar(
            x=class_counts.index.astype(str),
            y=class_counts.values,
            title=f'Class Distribution: {target}',
            labels={'x': 'Class', 'y': 'Count'}
        )
        figures.append(('plotly', fig))
        
        # Baseline accuracy
        n_classes = len(class_counts)
        if n_classes > 0:
            majority_class_count = class_counts.max()
            baseline_acc = majority_class_count / len(target_series)
            findings.append(f"Classes: {n_classes}")
            findings.append(f"Baseline accuracy (majority class): {baseline_acc:.1%}")
            
            imbalance_ratio = signals.target_stats.get('class_imbalance_ratio', 1.0)
            if imbalance_ratio < 0.5:
                warnings.append(f"Class imbalance detected (ratio: {imbalance_ratio:.2f}) - consider class weighting")
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures
    }


def dose_response_trends(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Plot dose-response trends for top numeric features."""
    findings = []
    warnings = []
    figures = []
    
    if not target or target not in df.columns:
        return {
            'findings': ["Target not available"],
            'warnings': [],
            'figures': []
        }
    
    numeric_features = [f for f in features if f in signals.numeric_cols and f != target]
    if len(numeric_features) == 0:
        return {
            'findings': ["No numeric features available"],
            'warnings': [],
            'figures': []
        }
    
    # Select top k features by association
    k = min(5, len(numeric_features))
    
    if signals.task_type_final == 'regression':
        # Use correlation
        correlations = []
        for feat in numeric_features:
            corr = abs(df[feat].corr(df[target]))
            if not np.isnan(corr):
                correlations.append((feat, corr))
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in correlations[:k]]
    else:
        # Use mutual information (sample for speed)
        sample_size = min(1000, len(df))
        df_sample = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
        
        try:
            mi_scores = mutual_info_classif(
                df_sample[numeric_features],
                df_sample[target],
                random_state=42
            )
            feature_mi = list(zip(numeric_features, mi_scores))
            feature_mi.sort(key=lambda x: x[1], reverse=True)
            top_features = [f[0] for f in feature_mi[:k]]
        except:
            # Fallback to correlation
            correlations = []
            for feat in numeric_features:
                corr = abs(df[feat].corr(df[target]))
                if not np.isnan(corr):
                    correlations.append((feat, corr))
            correlations.sort(key=lambda x: x[1], reverse=True)
            top_features = [f[0] for f in correlations[:k]]
    
    # Plot binned trends
    for feat in top_features:
        if feat not in df.columns:
            continue
        
        # Create bins
        feat_data = df[feat].dropna()
        if len(feat_data) < 10:
            continue
        
        n_bins = min(10, len(feat_data) // 10)
        if n_bins < 3:
            continue
        
        bins = pd.qcut(feat_data, q=n_bins, duplicates='drop')
        bin_centers = [interval.mid for interval in bins.cat.categories if pd.notna(interval)]
        bin_labels = df.loc[feat_data.index, target].groupby(bins).mean()
        
        if len(bin_centers) == len(bin_labels):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=bin_centers,
                y=bin_labels.values,
                mode='lines+markers',
                name=feat
            ))
            fig.update_layout(
                title=f'Dose-Response: {feat} vs {target}',
                xaxis_title=feat,
                yaxis_title=f'Mean {target}'
            )
            figures.append(('plotly', fig))
    
    findings.append(f"Analyzed top {len(top_features)} features by association with target")
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures
    }


def collinearity_map(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Create correlation heatmap for numeric features."""
    findings = []
    warnings = []
    figures = []
    
    numeric_features = [f for f in features if f in signals.numeric_cols]
    if len(numeric_features) < 2:
        return {
            'findings': ["Need at least 2 numeric features for collinearity analysis"],
            'warnings': [],
            'figures': []
        }
    
    # Limit to top 30 by variance
    if len(numeric_features) > 30:
        variances = df[numeric_features].var().sort_values(ascending=False)
        numeric_features = variances.head(30).index.tolist()
        findings.append("Limited to top 30 features by variance")
    
    corr_matrix = df[numeric_features].corr().abs()
    
    fig = px.imshow(
        corr_matrix,
        title='Feature Correlation Heatmap',
        labels=dict(x="Feature", y="Feature", color="|Correlation|"),
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    figures.append(('plotly', fig))
    
    # Find high correlation pairs
    high_corr_pairs = signals.collinearity_summary.get('high_corr_pairs', [])
    if high_corr_pairs:
        findings.append(f"Found {len(high_corr_pairs)} highly correlated pairs (>0.85)")
        warnings.append("High collinearity may cause GLM coefficient instability")
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures
    }


def leakage_scan(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Scan for target leakage risks."""
    findings = []
    warnings = []
    figures = []
    
    if not target or target not in df.columns:
        return {
            'findings': ["Target not available"],
            'warnings': [],
            'figures': []
        }
    
    # Use leakage candidates from signals
    if signals.leakage_candidate_cols:
        leakage_df = pd.DataFrame({
            'Column': signals.leakage_candidate_cols,
            'Risk': 'High correlation with target'
        })
        figures.append(('table', leakage_df))
        findings.append(f"Found {len(signals.leakage_candidate_cols)} columns with >0.95 correlation to target")
        warnings.append("⚠️ These columns should be excluded from features to prevent leakage")
    else:
        findings.append("No obvious leakage candidates detected")
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures
    }


def interaction_analysis(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Analyze interactions with demographic variables."""
    findings = []
    warnings = []
    figures = []
    
    if not target or target not in df.columns:
        return {
            'findings': ["Target not available"],
            'warnings': [],
            'figures': []
        }
    
    # Find demographic columns
    demo_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in ['age', 'sex', 'gender', 'bmi']):
            if col in signals.numeric_cols or col in signals.categorical_cols:
                demo_cols.append(col)
    
    if len(demo_cols) == 0:
        return {
            'findings': ["No demographic columns (age/sex/gender/BMI) found"],
            'warnings': [],
            'figures': []
        }
    
    # For each demo column, show stratified trends for top numeric features
    numeric_features = [f for f in features if f in signals.numeric_cols]
    if len(numeric_features) == 0:
        return {
            'findings': ["No numeric features available for interaction analysis"],
            'warnings': [],
            'figures': []
        }
    
    # Select top feature by correlation/MI
    if signals.task_type_final == 'regression':
        correlations = [(f, abs(df[f].corr(df[target]))) for f in numeric_features if not np.isnan(df[f].corr(df[target]))]
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_feature = correlations[0][0] if correlations else None
    else:
        # Use first feature as fallback
        top_feature = numeric_features[0] if numeric_features else None
    
    if top_feature:
        for demo_col in demo_cols[:2]:  # Limit to 2 demo columns
            if demo_col in df.columns and top_feature in df.columns:
                if demo_col in signals.categorical_cols:
                    # Box plot by category
                    fig = px.box(
                        df,
                        x=demo_col,
                        y=target,
                        color=demo_col,
                        title=f'{target} by {demo_col} (stratified)'
                    )
                    figures.append(('plotly', fig))
                else:
                    # Bin demo column and plot
                    demo_binned = pd.qcut(df[demo_col].dropna(), q=3, duplicates='drop', labels=['Low', 'Mid', 'High'])
                    df_temp = df.copy()
                    df_temp['_demo_bin'] = demo_binned
                    fig = px.box(
                        df_temp,
                        x='_demo_bin',
                        y=target,
                        title=f'{target} by {demo_col} (tertiles)'
                    )
                    figures.append(('plotly', fig))
        
        findings.append(f"Analyzed interactions with {len(demo_cols)} demographic variables")
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures
    }


def outlier_influence(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Analyze outlier influence on regression."""
    findings = []
    warnings = []
    figures = []
    
    if signals.task_type_final != 'regression' or not target:
        return {
            'findings': ["Outlier analysis only available for regression tasks"],
            'warnings': [],
            'figures': []
        }
    
    target_series = df[target].dropna()
    if len(target_series) < 10:
        return {
            'findings': ["Insufficient data for outlier analysis"],
            'warnings': [],
            'figures': []
        }
    
    # IQR method
    Q1 = target_series.quantile(0.25)
    Q3 = target_series.quantile(0.75)
    IQR = Q3 - Q1
    
    if IQR > 0:
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (target_series < lower_bound) | (target_series > upper_bound)
        n_outliers = outliers.sum()
        
        if n_outliers > 0:
            # Show outlier locations
            fig = px.scatter(
                df,
                x=target,
                y=target,
                color=outliers,
                title=f'Outlier Detection: {target}',
                labels={'color': 'Outlier'}
            )
            figures.append(('plotly', fig))
            
            findings.append(f"Found {n_outliers} outliers ({n_outliers/len(target_series):.1%})")
            findings.append(f"Outlier range: <{lower_bound:.2f} or >{upper_bound:.2f}")
            warnings.append("High outlier rate may require robust loss (Huber) or winsorization")
        else:
            findings.append("No outliers detected using IQR method")
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures
    }


def quick_probe_baselines(
    df: pd.DataFrame,
    target: Optional[str],
    features: List[str],
    signals: Any,
    session_state: Any
) -> Dict[str, Any]:
    """Run quick baseline models (constant, simple GLM, shallow RF)."""
    findings = []
    warnings = []
    figures = []
    
    if not target or target not in df.columns:
        return {
            'findings': ["Target not available"],
            'warnings': [],
            'figures': []
        }
    
    if len(features) == 0:
        return {
            'findings': ["No features selected"],
            'warnings': [],
            'figures': []
        }
    
    # Prepare data
    X = df[features].select_dtypes(include=[np.number])
    y = df[target]
    
    # Remove rows with missing target
    valid_mask = ~(y.isnull() | X.isnull().any(axis=1))
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(X) < 10:
        return {
            'findings': ["Insufficient data for baseline models"],
            'warnings': [],
            'figures': []
        }
    
    # Simple train/test split (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    results = []
    
    if signals.task_type_final == 'regression':
        # Constant predictor (mean)
        constant_pred = np.full(len(y_test), y_train.mean())
        mae_const = mean_absolute_error(y_test, constant_pred)
        rmse_const = np.sqrt(mean_squared_error(y_test, constant_pred))
        r2_const = r2_score(y_test, constant_pred)
        results.append({
            'Model': 'Constant (Mean)',
            'MAE': f"{mae_const:.3f}",
            'RMSE': f"{rmse_const:.3f}",
            'R²': f"{r2_const:.3f}"
        })
        
        # Simple GLM
        try:
            glm = LinearRegression()
            glm.fit(X_train, y_train)
            y_pred_glm = glm.predict(X_test)
            mae_glm = mean_absolute_error(y_test, y_pred_glm)
            rmse_glm = np.sqrt(mean_squared_error(y_test, y_pred_glm))
            r2_glm = r2_score(y_test, y_pred_glm)
            results.append({
                'Model': 'GLM (OLS)',
                'MAE': f"{mae_glm:.3f}",
                'RMSE': f"{rmse_glm:.3f}",
                'R²': f"{r2_glm:.3f}"
            })
        except Exception as e:
            warnings.append(f"GLM failed: {str(e)}")
        
        # Shallow RF
        try:
            rf = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            mae_rf = mean_absolute_error(y_test, y_pred_rf)
            rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
            r2_rf = r2_score(y_test, y_pred_rf)
            results.append({
                'Model': 'RF (10 trees, depth=3)',
                'MAE': f"{mae_rf:.3f}",
                'RMSE': f"{rmse_rf:.3f}",
                'R²': f"{r2_rf:.3f}"
            })
        except Exception as e:
            warnings.append(f"RF failed: {str(e)}")
    
    else:  # classification
        # Constant predictor (majority class)
        majority_class = y_train.mode()[0] if len(y_train.mode()) > 0 else y_train.iloc[0]
        constant_pred = np.full(len(y_test), majority_class)
        acc_const = accuracy_score(y_test, constant_pred)
        f1_const = f1_score(y_test, constant_pred, average='weighted')
        results.append({
            'Model': 'Constant (Majority)',
            'Accuracy': f"{acc_const:.3f}",
            'F1 (weighted)': f"{f1_const:.3f}"
        })
        
        # Simple Logistic
        try:
            logreg = LogisticRegression(max_iter=500, random_state=42)
            logreg.fit(X_train, y_train)
            y_pred_log = logreg.predict(X_test)
            acc_log = accuracy_score(y_test, y_pred_log)
            f1_log = f1_score(y_test, y_pred_log, average='weighted')
            results.append({
                'Model': 'Logistic Regression',
                'Accuracy': f"{acc_log:.3f}",
                'F1 (weighted)': f"{f1_log:.3f}"
            })
        except Exception as e:
            warnings.append(f"Logistic regression failed: {str(e)}")
        
        # Shallow RF
        try:
            rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            acc_rf = accuracy_score(y_test, y_pred_rf)
            f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
            results.append({
                'Model': 'RF (10 trees, depth=3)',
                'Accuracy': f"{acc_rf:.3f}",
                'F1 (weighted)': f"{f1_rf:.3f}"
            })
        except Exception as e:
            warnings.append(f"RF failed: {str(e)}")
    
    if results:
        results_df = pd.DataFrame(results)
        figures.append(('table', results_df))
        findings.append(f"Ran {len(results)} baseline models")
        findings.append("These are quick probes only - not saved as trained models")
    
    return {
        'findings': findings,
        'warnings': warnings,
        'figures': figures
    }
