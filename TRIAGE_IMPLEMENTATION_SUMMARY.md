# Task Type and Cohort Structure Detection - Implementation Summary

## What Changed

### 1. Session State Schema (`utils/session_state.py`)
- Added `TaskTypeDetection` dataclass with:
  - `detected`: Auto-detected task type
  - `confidence`: Detection confidence level
  - `reasons`: List of explanation strings
  - `override_enabled`: Whether override is active
  - `override_value`: Manual override value
  - `final`: Computed property returning override if enabled, else detected

- Added `CohortStructureDetection` dataclass with:
  - `detected`: Auto-detected cohort type (cross_sectional/longitudinal)
  - `confidence`: Detection confidence level
  - `reasons`: List of explanation strings
  - `override_enabled`: Whether override is active
  - `override_value`: Manual override value
  - `entity_id_candidates`: List of candidate entity ID columns
  - `entity_id_detected`: Best detected entity ID column
  - `entity_id_override_enabled`: Whether entity ID override is active
  - `entity_id_override_value`: Manual entity ID selection
  - `entity_id_final`: Computed property for final entity ID
  - `time_column_candidates`: List of candidate time/datetime columns
  - `final`: Computed property returning override if enabled, else detected

- Updated `init_session_state()` to initialize these detection objects

### 2. Detection Logic (`ml/triage.py`) - NEW FILE
- `detect_task_type(df, target)`: Detects regression vs classification
  - Checks dtype (object/category/bool → classification)
  - Checks unique value count and ratio
  - Handles binary (0/1) numeric as classification
  - Returns detected type, confidence, and reasons

- `detect_cohort_structure(df)`: Detects cross-sectional vs longitudinal
  - Pattern matching for entity ID columns (patient, subject, id, etc.)
  - Pattern matching for time columns (date, time, visit, etc.)
  - Analyzes median rows per entity ID
  - Checks for duplicate rows suggesting repeated measures
  - Returns detected type, confidence, reasons, entity ID candidates, and time column candidates

### 3. Upload & Audit Page (`pages/01_Upload_and_Audit.py`)
- Runs cohort structure detection when data is loaded
- Runs task type detection when target is selected/changed
- Displays auto-detection results in a panel:
  - Shows detected task type with confidence and reasons
  - Shows detected cohort structure with confidence and reasons
  - Shows detected entity ID column
- Provides manual override controls:
  - Checkbox + radio for task type override
  - Checkbox + radio for cohort structure override
  - Checkbox + selectbox for entity ID column override
- All widgets have stable keys for persistence
- Updates final values based on overrides
- Stores final task type in `data_config.task_type` for backward compatibility

### 4. Train & Compare Page (`pages/04_Train_and_Compare.py`)
- Reads final values from detection objects:
  - `task_type_final` from `task_type_detection.final`
  - `cohort_type_final` from `cohort_structure_detection.final`
  - `entity_id_final` from `cohort_structure_detection.entity_id_final`
- Uses `task_type_final` throughout (replaces `data_config.task_type`)
- Implements group-based splitting for longitudinal data:
  - Uses `GroupShuffleSplit` when `cohort_type_final == 'longitudinal'` and `entity_id_final` exists
  - Shows info message with group counts
  - Warns if longitudinal but no entity ID
- Falls back to time-based split if longitudinal but no entity ID
- All model selection and training uses `task_type_final`

## Files Modified

1. **utils/session_state.py**
   - Added `TaskTypeDetection` and `CohortStructureDetection` dataclasses
   - Updated `init_session_state()` to initialize detection objects

2. **ml/triage.py** (NEW)
   - `detect_task_type()` function
   - `detect_cohort_structure()` function

3. **pages/01_Upload_and_Audit.py**
   - Integrated detection logic
   - Added auto-detection display panel
   - Added manual override controls
   - Updated to use final values

4. **pages/04_Train_and_Compare.py**
   - Updated to read final values from detection objects
   - Implemented group-based splitting for longitudinal data
   - Replaced all `data_config.task_type` references with `task_type_final`
   - Added imports for `GroupShuffleSplit`

## Manual Verification Checklist

### Test 1: Regression Dataset Detection
1. Upload a regression-style dataset (continuous numeric target)
2. Select target column
3. **Expected**: Auto-detection shows "regression" with high/med confidence
4. Override to "classification"
5. **Expected**: Final task type is "classification"
6. Navigate to Train & Compare
7. **Expected**: Only classification models available, stratification enabled

### Test 2: Classification Dataset Detection
1. Upload a classification dataset (binary or low-cardinality target)
2. Select target column
3. **Expected**: Auto-detection shows "classification" with appropriate confidence
4. Check detection reasons (should explain why classification was detected)
5. Navigate to Train & Compare
6. **Expected**: Classification models available, stratification enabled

### Test 3: Longitudinal Data Detection
1. Upload a dataset with repeated patient/subject IDs (multiple rows per entity)
2. **Expected**: Cohort structure detection shows "longitudinal" with entity ID detected
3. Check detection reasons (should mention median rows per entity)
4. Navigate to Train & Compare
5. **Expected**: Info message about group-based splitting, splits prepared with groups
6. **Expected**: Group counts shown in info message

### Test 4: Override Persistence
1. Set task type override
2. Set cohort structure override
3. Navigate to another page and back
4. **Expected**: Overrides persist, widgets show correct values
5. **Expected**: Final values remain consistent across pages

### Test 5: Cross-Sectional Default
1. Upload a dataset without repeated entity IDs
2. **Expected**: Cohort structure detection shows "cross_sectional"
3. Navigate to Train & Compare
4. **Expected**: Standard random split (no group-based splitting)

## Notes

- Task type detection already existed but was basic; now uses comprehensive heuristics
- Cohort structure detection is new functionality
- All detection results are stored in session state and persist across page navigation
- Overrides take precedence over auto-detection
- Group-based splitting prevents data leakage in longitudinal data
- Backward compatibility maintained: `data_config.task_type` is still set for existing code
