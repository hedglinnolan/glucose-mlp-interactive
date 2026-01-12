"""
Page 02: Exploratory Data Analysis
Shows summary stats, distributions, correlations, and target analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional

from utils.session_state import init_session_state, get_data, DataConfig
from data_processor import get_numeric_columns

init_session_state()

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")

df = get_data()
if df is None:
    st.warning("⚠️ Please upload data in the Upload & Audit page first")
    st.stop()

data_config: Optional[DataConfig] = st.session_state.get('data_config')
if data_config is None or not data_config.target_col:
    st.warning("⚠️ Please select target and features in the Upload & Audit page first")
    st.stop()

target_col = data_config.target_col
feature_cols = data_config.feature_cols

st.info(f"**Target:** {target_col} | **Features:** {len(feature_cols)}")

# Summary statistics
st.header("📈 Summary Statistics")
st.dataframe(df[feature_cols + [target_col]].describe(), use_container_width=True)

# Distribution plots
st.header("📊 Distributions")

# Target distribution
st.subheader(f"Target Distribution: {target_col}")
col1, col2 = st.columns(2)

with col1:
    fig_hist = px.histogram(df, x=target_col, nbins=30, title=f"Distribution of {target_col}")
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    fig_box = px.box(df, y=target_col, title=f"Box Plot of {target_col}")
    st.plotly_chart(fig_box, use_container_width=True)

# Classification: class balance
if data_config.task_type == 'classification':
    st.subheader("Class Balance")
    class_counts = df[target_col].value_counts().sort_index()
    fig_bar = px.bar(x=class_counts.index.astype(str), y=class_counts.values,
                     title="Class Distribution", labels={'x': 'Class', 'y': 'Count'})
    st.plotly_chart(fig_bar, use_container_width=True)
    st.info(f"Classes: {len(class_counts)} | Imbalance ratio: {class_counts.max()/class_counts.min():.2f}")

# Feature distributions (top 6)
st.subheader("Feature Distributions")
n_features_show = min(6, len(feature_cols))
cols_per_row = 3

for i in range(0, n_features_show, cols_per_row):
    cols = st.columns(cols_per_row)
    for j, col in enumerate(cols):
        if i + j < n_features_show:
            feat = feature_cols[i + j]
            with col:
                fig = px.histogram(df, x=feat, nbins=20, title=feat)
                st.plotly_chart(fig, use_container_width=True)

# Correlation heatmap
st.header("🔗 Correlation Analysis")
numeric_cols = get_numeric_columns(df)
corr_cols = [c for c in feature_cols + [target_col] if c in numeric_cols]

if len(corr_cols) > 1:
    corr_matrix = df[corr_cols].corr()
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap",
        color_continuous_scale='RdBu',
        zmin=-1, zmax=1
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Target vs feature correlations
    st.subheader("Target-Feature Correlations")
    target_corrs = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)
    fig_bar = px.bar(
        x=target_corrs.index,
        y=target_corrs.values,
        title=f"Correlation with {target_col}",
        labels={'x': 'Feature', 'y': 'Correlation'}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Target vs feature plots
st.header("🎯 Target vs Features")

# Regression: scatter plots
if data_config.task_type == 'regression':
    n_plots = min(6, len(feature_cols))
    cols_per_row = 3
    
    for i in range(0, n_plots, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < n_plots:
                feat = feature_cols[i + j]
                with col:
                    fig = px.scatter(df, x=feat, y=target_col, title=f"{target_col} vs {feat}")
                    st.plotly_chart(fig, use_container_width=True)

# Classification: box plots
else:
    n_plots = min(6, len(feature_cols))
    cols_per_row = 3
    
    for i in range(0, n_plots, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < n_plots:
                feat = feature_cols[i + j]
                with col:
                    fig = px.box(df, x=target_col, y=feat, title=f"{feat} by {target_col}")
                    st.plotly_chart(fig, use_container_width=True)

st.success("✅ EDA complete. Proceed to Preprocessing page.")
