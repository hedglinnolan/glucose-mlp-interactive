"""
Utilities for checking estimator fitted status across sklearn and custom models.
"""
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


def is_estimator_fitted(estimator):
    """
    Check if an estimator is fitted (works for sklearn models and custom wrappers).
    
    Args:
        estimator: sklearn estimator or custom wrapper
        
    Returns:
        bool: True if fitted, False otherwise
    """
    # For sklearn models, use sklearn's validation (this is the standard way)
    try:
        check_is_fitted(estimator)
        return True
    except NotFittedError:
        # sklearn says it's not fitted
        # Check for custom is_fitted_ attribute (for our custom wrappers)
        if hasattr(estimator, 'is_fitted_'):
            return estimator.is_fitted_
        return False
    except (AttributeError, ValueError, TypeError):
        # Some sklearn models might not pass check_is_fitted but are still fitted
        # Check for common fitted attributes
        if hasattr(estimator, 'coef_') or hasattr(estimator, 'feature_importances_') or hasattr(estimator, 'n_features_in_'):
            return True
        # Check for custom is_fitted_ attribute
        if hasattr(estimator, 'is_fitted_'):
            return estimator.is_fitted_
        return False
    except Exception:
        # If check fails for other reasons, try to infer from common fitted attributes
        if hasattr(estimator, 'coef_') or hasattr(estimator, 'feature_importances_') or hasattr(estimator, 'n_features_in_'):
            return True
        if hasattr(estimator, 'is_fitted_'):
            return estimator.is_fitted_
        return False
