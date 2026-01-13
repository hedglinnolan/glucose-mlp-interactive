# Smoke Test Checklist

## Test 1: Basic Navigation & State Persistence
1. Start app: `.\run.ps1` (Windows) or `./run.sh` (macOS/Linux)
2. **Expected**: App loads without errors, all pages accessible
3. Upload CSV or generate built-in dataset
4. Select target and features on Upload page
5. Navigate to EDA page
6. Navigate back to Upload page
7. **Expected**: Target and features still selected, no reset
8. Set task type override
9. Navigate to EDA and back
10. **Expected**: Task type override persists

## Test 2: Dataset Replacement
1. Upload/generate first dataset
2. Select target and features
3. Upload/generate different dataset (check "Replace existing dataset")
4. **Expected**: Invalid columns removed from selections, valid selections preserved
5. **Expected**: Overrides persist if compatible

## Test 3: Train & Compare Workflow
1. Complete Upload → EDA → Preprocess
2. Go to Train & Compare
3. Adjust split percentages
4. Click "Prepare Splits"
5. **Expected**: Splits prepared successfully, no `.values` error
6. Select models (NN, RF, GLM)
7. Adjust hyperparameters
8. Click "Train Models"
9. **Expected**: Models train without crashing entire page if one fails
10. **Expected**: Training errors shown in expander, not blocking

## Test 4: Explainability
1. After training models, go to Explainability page
2. Click "Calculate Permutation Importance"
3. **Expected**: Works for all trained models
4. Click "Calculate Partial Dependence"
5. **Expected**: Works for all trained models
6. Enable SHAP, adjust sample sizes
7. Click "Calculate SHAP Values"
8. **Expected**: SHAP computes with progress feedback

## Test 5: Report Export
1. After completing workflow, go to Report Export
2. Click "Download Report"
3. **Expected**: Report downloads without crashing
4. **Expected**: Report includes all key information

## Test 6: Longitudinal Data
1. Upload/create dataset with repeated entity IDs
2. **Expected**: Entity ID detected is actually an ID (not triglycerides/glucose)
3. Go to Train & Compare
4. **Expected**: Group split used only if valid ID detected
5. **Expected**: Warning if ID invalid, falls back to standard split

## Test 7: State Debug
1. On any page, expand "Advanced / State Debug"
2. **Expected**: Shows current state values
3. Navigate to another page and back
4. **Expected**: State values persist correctly

## Test 8: Error Handling
1. Train a model that might fail (e.g., insufficient data)
2. **Expected**: Error shown in expander, page doesn't crash
3. Other models can still be trained
4. **Expected**: Error details visible for debugging
