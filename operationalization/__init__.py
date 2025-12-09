"""
Phase 6: Operationalization & Outputs Module

This module provides production-ready components for operational AMR surveillance:
- Automated antibiogram generator
- Early-warning alert system
- Empiric therapy recommender
- REST API for model scoring
- Batch scoring pipeline
- Model registry and versioning

Author: AMR Surveillance Team
"""

from .antibiogram_generator import AntibiogramGenerator
from .early_warning import EarlyWarningSystem
from .therapy_recommender import EmpiricTherapyRecommender

__all__ = [
    'AntibiogramGenerator',
    'EarlyWarningSystem',
    'EmpiricTherapyRecommender',
]
