"""
Clinical unit inference and conversion utilities.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Literal


# Clinical variable patterns and unit hypotheses
CLINICAL_VARIABLES = {
    'weight': {
        'canonical_unit': 'kg',
        'hypotheses': [
            ('kg', 1.0, (30, 200)),  # (unit_name, conversion_factor, plausible_range_in_canonical)
            ('lb', 0.453592, (66, 440))  # lb to kg
        ]
    },
    'height': {
        'canonical_unit': 'cm',
        'hypotheses': [
            ('cm', 1.0, (100, 220)),
            ('inches', 2.54, (39, 87)),  # inches to cm
            ('meters', 100.0, (1.0, 2.2))  # meters to cm
        ]
    },
    'waist': {
        'canonical_unit': 'cm',
        'hypotheses': [
            ('cm', 1.0, (50, 150)),
            ('inches', 2.54, (20, 59))
        ]
    },
    'glucose': {
        'canonical_unit': 'mmol/L',
        'hypotheses': [
            ('mmol/L', 1.0, (3.5, 25.0)),
            ('mg/dL', 0.0555, (70, 500))  # mg/dL to mmol/L: divide by 18
        ]
    },
    'cholesterol': {
        'canonical_unit': 'mmol/L',
        'hypotheses': [
            ('mmol/L', 1.0, (2.0, 10.0)),
            ('mg/dL', 0.0259, (80, 400))  # mg/dL to mmol/L: divide by 38.67
        ]
    },
    'triglyceride': {
        'canonical_unit': 'mmol/L',
        'hypotheses': [
            ('mmol/L', 1.0, (0.5, 5.0)),
            ('mg/dL', 0.0113, (50, 500))  # mg/dL to mmol/L: divide by 88.57
        ]
    },
    'bp_sys': {
        'canonical_unit': 'mmHg',
        'hypotheses': [
            ('mmHg', 1.0, (80, 200))
        ]
    },
    'bp_di': {
        'canonical_unit': 'mmHg',
        'hypotheses': [
            ('mmHg', 1.0, (40, 120))
        ]
    },
    'bmi': {
        'canonical_unit': 'kg/m²',
        'hypotheses': [
            ('kg/m²', 1.0, (15, 50))
        ]
    },
    'hba1c': {
        'canonical_unit': '%',
        'hypotheses': [
            ('%', 1.0, (4.0, 15.0))
        ]
    },
    'kcal': {
        'canonical_unit': 'kcal',
        'hypotheses': [
            ('kcal', 1.0, (500, 5000)),
            ('kJ', 0.239, (2100, 21000))  # kJ to kcal
        ]
    }
}


def infer_unit(
    col_name: str,
    values: pd.Series
) -> Dict[str, any]:
    """
    Infer the most likely unit for a clinical variable.
    
    Args:
        col_name: Column name
        values: Series of values
        
    Returns:
        Dict with:
            - inferred_unit: str or None
            - canonical_unit: str or None
            - confidence: 'high'|'med'|'low'|None
            - explanation: str
            - conversion_factor: float or None
    """
    col_lower = col_name.lower()
    
    # Find matching clinical variable
    matched_var = None
    for var_name, var_config in CLINICAL_VARIABLES.items():
        if var_name in col_lower:
            matched_var = var_name
            break
    
    if not matched_var:
        return {
            'inferred_unit': None,
            'canonical_unit': None,
            'confidence': None,
            'explanation': 'No reference range available for this variable',
            'conversion_factor': None
        }
    
    var_config = CLINICAL_VARIABLES[matched_var]
    canonical_unit = var_config['canonical_unit']
    hypotheses = var_config['hypotheses']
    
    # Remove NaN values
    clean_values = values.dropna()
    if len(clean_values) == 0:
        return {
            'inferred_unit': None,
            'canonical_unit': canonical_unit,
            'confidence': None,
            'explanation': 'No valid values to analyze',
            'conversion_factor': None
        }
    
    # Test each hypothesis
    best_hypothesis = None
    best_score = -1
    best_fit_pct = 0
    
    for unit_name, conversion_factor, (min_val, max_val) in hypotheses:
        # Convert to canonical units
        converted = clean_values * conversion_factor
        
        # Compute fit score: % within range + penalty for extreme outliers
        within_range = ((converted >= min_val) & (converted <= max_val)).sum()
        fit_pct = within_range / len(converted)
        
        # Penalty for extreme outliers (beyond 2x range)
        range_width = max_val - min_val
        extreme_low = (converted < min_val - 2 * range_width).sum()
        extreme_high = (converted > max_val + 2 * range_width).sum()
        extreme_penalty = (extreme_low + extreme_high) / len(converted) * 0.5
        
        score = fit_pct - extreme_penalty
        
        if score > best_score:
            best_score = score
            best_hypothesis = (unit_name, conversion_factor, min_val, max_val)
            best_fit_pct = fit_pct
    
    if best_hypothesis is None:
        return {
            'inferred_unit': None,
            'canonical_unit': canonical_unit,
            'confidence': 'low',
            'explanation': 'Could not determine unit with confidence',
            'conversion_factor': None
        }
    
    unit_name, conversion_factor, min_val, max_val = best_hypothesis
    
    # Determine confidence
    if best_fit_pct >= 0.9:
        confidence = 'high'
    elif best_fit_pct >= 0.7:
        confidence = 'med'
    else:
        confidence = 'low'
    
    explanation = f"{best_fit_pct:.0%} of values within plausible {canonical_unit} range after converting {unit_name}→{canonical_unit}"
    
    return {
        'inferred_unit': unit_name,
        'canonical_unit': canonical_unit,
        'confidence': confidence,
        'explanation': explanation,
        'conversion_factor': conversion_factor
    }
