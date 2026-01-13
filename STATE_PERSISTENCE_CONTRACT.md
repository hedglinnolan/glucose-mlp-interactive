# State Persistence Contract

## Single Source of Truth
All application state lives in `st.session_state` and is initialized via `utils/session_state.init_session_state()`.

## State Reset Rules

### What TRIGGERS a reset (intentional):
1. **Dataset replacement**: When user uploads new CSV or generates new built-in dataset AND confirms "Replace existing dataset"
   - Action: `reconcile_state_with_df()` is called to drop invalid columns from features/target/entity_id
   - Preserved: Task/cohort overrides, unit overrides, model configs (unless incompatible)

2. **Column invalidation**: When selected target/features/entity_id no longer exist in current DataFrame
   - Action: Invalid columns removed from selections
   - Preserved: Valid selections remain

3. **Task type incompatibility**: When task type changes and makes a model invalid (e.g., Huber for classification)
   - Action: Model deselected with warning
   - Preserved: Other models remain selected

### What NEVER resets (must persist):
1. **User selections**: Target, features, task type override, cohort override, entity ID override
2. **Model hyperparameters**: All model configs persist across navigation
3. **Split settings**: Train/val/test percentages, CV settings
4. **Unit overrides**: User-specified unit overrides for clinical variables
5. **Trained models**: Once trained, models remain in session_state
6. **EDA results**: Analysis results persist in `eda_results` dict

## Widget Key Naming Convention
All widgets use stable, deterministic keys with page prefix:
- Upload page: `upload_*` (e.g., `upload_target_selectbox`, `upload_task_override_checkbox`)
- EDA page: `eda_*` (e.g., `eda_manual_action_select`)
- Preprocess page: `preprocess_*` (e.g., `preprocess_numeric_imputation`)
- Train page: `train_*` (e.g., `train_split_train_pct`, `train_model_nn`)
- Explainability page: `explain_*` (e.g., `explain_perm_importance_button`)
- Report page: `report_*` (e.g., `report_download_button`)

## State Reconciliation
When DataFrame changes:
1. Call `reconcile_state_with_df(df, session_state)`
2. This function:
   - Drops invalid columns from feature list
   - Clears target if missing
   - Clears entity_id if missing
   - Preserves valid selections
   - Does NOT reset overrides unless data is incompatible

## Page Initialization Pattern
Every page must:
1. Call `init_session_state()` at the top
2. Read from `session_state` for widget defaults (not hardcoded)
3. Use stable `key=` for all widgets
4. Store user changes back to `session_state`

## Navigation Behavior
- Upload → EDA → Upload: All selections preserved
- Any page → Any page: State persists unless explicitly reset by user action
- Reruns: Widgets rehydrate from `session_state` using their `key=`
