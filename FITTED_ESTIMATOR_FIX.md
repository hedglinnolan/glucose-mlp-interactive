# Fitted Estimator Status Fix

## Problem

Explainability functions were failing with:
- "Estimator not marked as fitted" for RF, GLM, Huber
- "'estimator' must be a fitted regressor or classifier" for NN partial dependence

## Root Cause

1. **Sklearn models don't use `is_fitted_`**: They use sklearn's `check_is_fitted()` which looks for fitted attributes like `coef_`, `feature_importances_`, etc.
2. **Our check was too strict**: We were checking `hasattr(estimator, 'is_fitted_')` which fails for sklearn models
3. **NN estimator needed proper initialization**: The sklearn wrapper needed to be properly marked as fitted

## Solution

1. **Created `ml/estimator_utils.py`** with `is_estimator_fitted()` helper:
   - Uses sklearn's `check_is_fitted()` first (works for sklearn models)
   - Falls back to `is_fitted_` attribute (for our custom wrappers)
   - Handles exceptions gracefully

2. **Updated explainability checks**:
   - All explainability functions now use `is_estimator_fitted()` instead of checking `is_fitted_` directly
   - Works for both sklearn models and custom wrappers

3. **Ensured proper storage**:
   - NN: Store sklearn-compatible wrapper, ensure it's marked as fitted
   - Others: Store sklearn model directly (already fitted after `model.fit()`)

## Files Changed

1. `ml/estimator_utils.py` - NEW: Helper function for checking fitted status
2. `pages/05_Explainability.py` - Use `is_estimator_fitted()` for all checks
3. `pages/04_Train_and_Compare.py` - Ensure NN estimator is properly initialized

## Test Checklist

1. ✅ Train RF, GLM, Huber models
2. ✅ Go to Explainability page
3. ✅ Click "Calculate Permutation Importance"
   - **Expected:** All models (RF, GLM, Huber) succeed without "not marked as fitted" errors
4. ✅ Click "Calculate Partial Dependence"
   - **Expected:** All models succeed, including NN
5. ✅ Enable SHAP and calculate
   - **Expected:** All models succeed without "not marked as fitted" errors
