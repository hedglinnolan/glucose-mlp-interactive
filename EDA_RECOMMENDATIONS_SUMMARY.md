# EDA Recommendation Cards System - Implementation Summary

## What Changed

### 1. Data Model (`ml/eda_recommender.py`) - NEW FILE
- **DatasetSignals** dataclass: Comprehensive dataset signals including:
  - Basic stats (n_rows, n_cols, column types)
  - Missingness patterns
  - Target statistics (skew, kurtosis, outlier rate, class balance)
  - Leakage flags and candidates
  - Collinearity summary
  - Unit sanity flags (medical/nutritional heuristics)
  
- **EDARecommendation** dataclass: Individual recommendation cards with:
  - id, title, priority, cost
  - why (triggered reasons)
  - what_you_learn, model_implications
  - run_action (function name)
  
- **compute_dataset_signals()**: Computes all signals from dataset
  - Fast: uses sampling for expensive operations
  - Includes medical/nutritional unit sanity checks
  - Detects leakage, collinearity, missingness patterns
  
- **recommend_eda()**: Generates 10+ recommendation types:
  - R1: Physiologic plausibility (always)
  - R2: Missingness mechanism scan
  - R3: Cohort structure guidance (longitudinal)
  - R4: Leakage risk scan
  - R5: Target distribution/class balance
  - R6: Dose-response trends
  - R7: Interaction analysis (age/sex/BMI)
  - R8: Collinearity map
  - R9: Outlier influence (regression)
  - R10: Quick probe baselines (always)

### 2. Analysis Functions (`ml/eda_actions.py`) - NEW FILE
Implements all runnable analyses:

- **plausibility_check()**: Range checks for clinical variables
- **missingness_scan()**: Missingness patterns and association with target
- **cohort_split_guidance()**: Longitudinal data split guidance
- **target_profile()**: Target distribution (regression/classification)
- **dose_response_trends()**: Binned trends for top features
- **collinearity_map()**: Correlation heatmap
- **leakage_scan()**: Target leakage risk assessment
- **interaction_analysis()**: Stratified trends by demographics
- **outlier_influence()**: Outlier detection and impact
- **quick_probe_baselines()**: Fast baseline models (constant, GLM, shallow RF)

All functions return: `{'findings': [], 'warnings': [], 'figures': []}`

### 3. EDA Page Integration (`pages/02_EDA.py`)
- Computes signals using final task/cohort types from session_state
- Generates recommendations based on signals
- Displays recommendation cards with:
  - Title, priority, cost badge
  - "Why recommended" expander
  - "What you'll learn" expander
  - "Model implications" expander
  - "Run" button per card
- Executes actions when "Run" clicked
- Stores results in `session_state.eda_results[rec_id]`
- Displays results (findings, warnings, figures) under each card
- Manual mode expander for running any analysis
- "Explain recommendations" expander showing dataset signals

## Files Modified/Added

1. **ml/eda_recommender.py** (NEW)
   - DatasetSignals dataclass
   - EDARecommendation dataclass
   - compute_dataset_signals() function
   - recommend_eda() function

2. **ml/eda_actions.py** (NEW)
   - 10 analysis action functions
   - All return standardized dict format

3. **pages/02_EDA.py**
   - Integrated recommendation cards UI
   - Signal computation and caching
   - Results storage and display
   - Manual mode

4. **utils/session_state.py** (already updated)
   - eda_results dictionary initialized in init_session_state()

## Manual Test Checklist

### Test 1: Regression with Outliers Dataset
1. Upload "Linear Regression with Outliers" built-in dataset
2. Select target and features
3. Go to EDA page
4. **Expected recommendations:**
   - R1: Physiologic plausibility (always)
   - R5: Target distribution & outliers (high outlier rate)
   - R9: Outlier influence analysis
   - R6: Dose-response trends
   - R10: Quick probe baselines
5. Click "Run" on R5 (Target Distribution)
6. **Expected:** Histogram, log histogram, outlier summary, skewness warning
7. Click "Run" on R9 (Outlier Influence)
8. **Expected:** Outlier scatter plot, outlier count, warnings about robust loss

### Test 2: Nonlinear Regression Dataset
1. Upload "Nonlinear Regression" built-in dataset
2. Go to EDA page
3. **Expected recommendations:**
   - R6: Dose-response trends (should show nonlinear patterns)
   - R10: Quick probe baselines
4. Click "Run" on R6
5. **Expected:** Binned trend plots showing nonlinear relationships
6. Click "Run" on R10
7. **Expected:** Baseline model table, RF should outperform GLM

### Test 3: Imbalanced Classification Dataset
1. Upload "Imbalanced Classification" built-in dataset
2. Go to EDA page
3. **Expected recommendations:**
   - R5: Class balance & baseline (should show imbalance)
   - R10: Quick probe baselines
4. Click "Run" on R5
5. **Expected:** Class distribution bar chart, baseline accuracy, imbalance warning
6. Click "Run" on R10
7. **Expected:** Baseline model table with accuracy/F1 metrics

### Test 4: Longitudinal Data
1. Upload/create dataset with repeated entity IDs
2. Ensure cohort detection identifies as longitudinal
3. Go to EDA page
4. **Expected recommendations:**
   - R3: Longitudinal data split guidance
5. Click "Run" on R3
6. **Expected:** Entity ID distribution plot, warnings about group-based splitting

### Test 5: Manual Mode
1. Go to EDA page
2. Expand "Manual Mode" expander
3. Select any analysis from dropdown
4. Click "Run Selected Analysis"
5. **Expected:** Results displayed below

### Test 6: Persistence
1. Run an analysis (e.g., R1)
2. Navigate to another page
3. Navigate back to EDA
4. **Expected:** Results still visible under the card

## Notes

- All recommendations are explainable: each shows WHY it was recommended with concrete signals
- Recommendations respect `task_type_final` and `cohort_type_final` from session_state
- Results persist in `session_state.eda_results` across page navigation
- Heavy computations are cached using `@st.cache_data`
- All actions catch exceptions and show errors gracefully
- Medical/nutritional unit sanity checks are lightweight heuristics (labeled as such)
