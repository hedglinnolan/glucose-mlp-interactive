# Bug Fixes Summary

## Issues Fixed (A-G)

### A) State Reset Bugs - Target/Features/Task Selections

**Problem:** Selecting target resets features; changing task resets target/features.

**Fix:**
- Added stable `key=` parameters to all widgets (target_selectbox, features_multiselect, task_type_radio, datetime_selectbox)
- Created `utils/reconcile.py` with `reconcile_target_features()` helper to enforce rules without resetting
- Modified Upload page to rehydrate widgets from `session_state.data_config` on every load
- Widgets now initialize from existing config if available, otherwise use defaults
- Target changes now only remove invalid features (target from features list), preserving valid selections

**Files Changed:**
- `pages/01_Upload_and_Audit.py` - Added keys, reconciliation logic, state rehydration
- `utils/reconcile.py` - NEW: Helper function for target/feature reconciliation
- `utils/session_state.py` - Added `data_source` tracking

---

### B) Generated Data Persistence

**Problem:** Generated example data gets wiped on target selection.

**Fix:**
- Generated data now properly stored in `session_state.raw_data` via `set_data()`
- Added confirmation checkbox "Replace existing dataset" (default unchecked) before overwriting
- After generation, page reruns to update UI with new data
- Data source tracked in `session_state.data_source`

**Files Changed:**
- `pages/01_Upload_and_Audit.py` - Added confirmation pattern, proper state storage
- `utils/session_state.py` - Added `data_source` field

---

### C) Cross-Page Navigation Reset

**Problem:** Upload page resets after visiting EDA and coming back.

**Fix:**
- Upload page now reads ALL widget values from `session_state.data_config` (single source of truth)
- No reinitialization of defaults if state exists
- Widgets use `index=` parameter to set initial selection from existing config
- All selections persist across page navigation

**Files Changed:**
- `pages/01_Upload_and_Audit.py` - Complete rehydration from session_state

---

### D) Plot Clarity - Prediction vs Actual

**Problem:** Dashed red line labeled "Perfect Predictions" is unclear.

**Fix:**
- Renamed legend to "y = x reference (perfect agreement)"
- Added caption explaining the reference line
- For classification tasks, hide Prediction vs Actual plot and show note instead

**Files Changed:**
- `visualizations.py` - Updated legend label
- `pages/04_Train_and_Compare.py` - Added caption, conditional display for classification

---

### E) Permutation Importance Crashes for NN

**Problem:** sklearn's `permutation_importance` requires sklearn-compatible estimator, but NN returns PyTorch model.

**Fix:**
- Created `SklearnCompatibleNN` wrapper class (BaseEstimator + RegressorMixin)
- NN wrapper's `get_model()` now returns sklearn-compatible wrapper instead of raw PyTorch model
- Wrapper implements `fit()`, `predict()`, `get_params()`, `set_params()` for sklearn compatibility

**Files Changed:**
- `models/nn_whuber.py` - Added `SklearnCompatibleNN` class, updated `get_model()` method

---

### F) Partial Dependence Crashes + st.columns Error

**Problem:** 
- PD crashes because it receives PyTorch model instead of sklearn estimator
- `st.columns()` receives invalid spec when `len(pd_data) == 0`

**Fix:**
- PD now uses `model.get_model()` which returns sklearn-compatible wrapper (fixes NN)
- Fixed `st.columns()` to always use positive integer: `max(1, min(3, len(pd_data)))`
- PD now uses original feature names (pre-transform) to avoid confusion with one-hot expansions
- Added error handling with expander showing detailed errors instead of crashing
- Handles sparse matrices safely (converts to dense only when needed)

**Files Changed:**
- `pages/05_Explainability.py` - Fixed st.columns spec, improved error handling, use sklearn-compatible models

---

### G) Report Export - Missing Tabulate Dependency

**Problem:** Report Export crashes with `ImportError: Missing optional dependency 'tabulate'`.

**Fix:**
- Added fallback markdown table generation when `tabulate` is missing
- Manual table creation using string formatting
- Report still generates and downloads successfully without tabulate

**Files Changed:**
- `pages/06_Report_Export.py` - Added try/except with fallback table generation

---

## Manual Test Checklist

### Test A: State Persistence
1. ✅ Upload CSV → Select target → Select features → Change target
   - **Expected:** Features persist, only target removed from features if it was selected
2. ✅ Select task type → Change target
   - **Expected:** Task type persists, target/features remain
3. ✅ Navigate Upload → EDA → Upload
   - **Expected:** All selections remain intact

### Test B: Generated Data Persistence
1. ✅ Generate built-in dataset → Select target
   - **Expected:** Data persists, selections work
2. ✅ Generate dataset when data already exists
   - **Expected:** Checkbox appears, data only replaced if checked

### Test C: Cross-Page Navigation
1. ✅ Upload → Configure → EDA → Back to Upload
   - **Expected:** All widgets show previous selections

### Test D: Plot Clarity
1. ✅ Train regression model → View "Predictions vs Actual"
   - **Expected:** Legend shows "y = x reference (perfect agreement)", caption explains reference line
2. ✅ Train classification model → View diagnostics
   - **Expected:** No "Predictions vs Actual" plot, note shown instead

### Test E: Permutation Importance (NN)
1. ✅ Train NN model → Calculate Permutation Importance
   - **Expected:** No crash, importance calculated successfully

### Test F: Partial Dependence
1. ✅ Train NN model → Calculate Partial Dependence
   - **Expected:** No crash, PD plots displayed
2. ✅ Train model with categorical features → Calculate PD
   - **Expected:** Uses original feature names, handles sparse matrices

### Test G: Report Export
1. ✅ Complete workflow → Export Report (without tabulate installed)
   - **Expected:** Report generates successfully, markdown tables formatted correctly

---

## Summary of Files Modified

1. `pages/01_Upload_and_Audit.py` - State persistence, reconciliation, confirmation patterns
2. `pages/04_Train_and_Compare.py` - Plot clarity improvements
3. `pages/05_Explainability.py` - Sklearn compatibility, st.columns fix, error handling
4. `pages/06_Report_Export.py` - Tabulate fallback
5. `models/nn_whuber.py` - Sklearn-compatible wrapper
6. `visualizations.py` - Plot legend improvements
7. `utils/reconcile.py` - NEW: Target/feature reconciliation helper
8. `utils/session_state.py` - Added data_source tracking

All fixes are minimal and surgical, maintaining existing architecture.
