# Widget Keys Audit - Complete List

## Page 01: Upload & Audit
- `upload_dataset_select` - Built-in dataset selector
- `replace_dataset_checkbox` - Replace dataset confirmation (built-in)
- `generate_dataset_button` - Generate built-in dataset button
- `csv_uploader` - CSV file uploader
- `replace_upload_checkbox` - Replace dataset confirmation (upload)
- `upload_target_selectbox` - Target variable selector
- `upload_features_multiselect` - Feature variables multiselect
- `upload_datetime_selectbox` - Datetime column selector
- `upload_task_override_checkbox` - Task type override checkbox
- `upload_task_override_radio` - Task type override radio
- `upload_cohort_override_checkbox` - Cohort structure override checkbox
- `upload_cohort_override_radio` - Cohort structure override radio
- `upload_entity_id_override_checkbox` - Entity ID override checkbox
- `upload_entity_id_override_selectbox` - Entity ID override selectbox

## Page 02: EDA
- `show_all_recommendations` - Show all recommendations checkbox
- `run_{rec.id}` - Run button for each recommendation (dynamic)
- `eda_manual_action_select` - Manual mode action selector
- `eda_manual_run_button` - Manual mode run button

## Page 03: Preprocess
- `preprocess_numeric_imputation` - Numeric imputation strategy
- `preprocess_numeric_scaling` - Numeric scaling strategy
- `preprocess_numeric_log_transform` - Numeric log transform checkbox
- `preprocess_categorical_imputation` - Categorical imputation strategy
- `preprocess_categorical_encoding` - Categorical encoding strategy
- `preprocess_build_button` - Build pipeline button
- `preprocess_rebuild_button` - Rebuild pipeline button

## Page 04: Train & Compare
- `train_use_time_split` - Time-based split checkbox
- `train_split_train_pct` - Train percentage slider
- `train_split_val_pct` - Validation percentage slider
- `train_split_test_pct` - Test percentage slider
- `train_use_cv` - Cross-validation checkbox
- `train_cv_folds` - CV folds slider
- `train_prepare_splits_button` - Prepare splits button
- `train_model_nn` - Neural Network checkbox
- `nn_epochs` - NN epochs number input
- `nn_batch` - NN batch size number input
- `nn_lr` - NN learning rate number input
- `nn_wd` - NN weight decay number input
- `nn_patience` - NN early stopping patience number input
- `nn_dropout` - NN dropout number input
- `train_model_rf` - Random Forest checkbox
- `rf_n_est` - RF n_estimators number input
- `rf_depth` - RF max_depth number input
- `rf_leaf` - RF min_samples_leaf number input
- `train_model_glm` - GLM checkbox
- `train_model_huber` - Huber GLM checkbox
- `huber_eps` - Huber epsilon number input
- `huber_alpha` - Huber alpha number input
- `train_models_button` - Train models button

## Page 05: Explainability
- `explain_perm_importance_button` - Permutation importance button
- `explain_pd_button` - Partial dependence button
- `explain_shap_enable` - SHAP enable checkbox
- `explain_shap_background_size` - SHAP background size slider
- `explain_shap_eval_size` - SHAP evaluation size slider
- `explain_shap_button` - SHAP calculate button
- `shap_class_{name}` - SHAP class selector (dynamic, per model)

## Page 06: Report Export
- `report_download_button` - Download report button

## Global Settings (Sidebar)
- Random seed control (in Train & Compare sidebar) - uses `random_seed` in session_state
