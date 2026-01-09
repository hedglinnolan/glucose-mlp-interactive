# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-01-09

### Added
- Initial release of Interactive Regression Model Trainer
- Support for 4 model types:
  - Neural Network (2-layer MLP with Weighted Huber loss)
  - Random Forest (500 trees)
  - GLM OLS (Ordinary Least Squares)
  - GLM Huber (Robust regression)
- CSV file upload with drag-and-drop
- Feature and target selection interface
- Real-time training progress display
- Side-by-side model comparison
- Interactive visualizations (Plotly):
  - Training history plots
  - Predictions vs Actual scatter plots
  - Residual analysis plots
- Results download as CSV
- Comprehensive error handling and logging
- Setup scripts for Windows (PowerShell) and Unix/macOS (Bash)
- Example dataset for testing
- Complete documentation (README, QUICKSTART, DEPLOYMENT, CONTRIBUTING)

### Features
- Automatic data validation
- Feature standardization
- Train/Val/Test split (70/15/15)
- Early stopping for Neural Network
- Learning rate scheduling
- Model comparison table with best model highlighting

### Technical
- Python 3.8+ support
- Type hints throughout
- Comprehensive docstrings
- Error handling with user-friendly messages
- Logging system for debugging
- Streamlit configuration file
