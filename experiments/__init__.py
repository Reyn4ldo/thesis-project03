"""
Experiments module for supervised learning tasks.

This module provides a framework for running supervised learning experiments
with multiple algorithms across various prediction tasks.
"""

from .base_experiment import BaseExperiment
from .esbl_classifier import ESBLClassifierExperiment
from .multilabel_prediction import MultilabelPredictionExperiment
from .species_classifier import SpeciesClassifierExperiment
from .mar_regression import MARRegressionExperiment

__all__ = [
    'BaseExperiment',
    'ESBLClassifierExperiment',
    'MultilabelPredictionExperiment',
    'SpeciesClassifierExperiment',
    'MARRegressionExperiment'
]
