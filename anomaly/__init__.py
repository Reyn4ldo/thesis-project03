"""
Anomaly detection module for identifying rare, extreme, or inconsistent isolates.

This module provides tools for:
- Unsupervised outlier detection (Isolation Forest, LOF, DBSCAN)
- Rule-based consistency checks for MIC vs interpretive calls
- Multivariate distance-based detection (Mahalanobis distance)
- Anomaly scoring and triage
"""

from .outlier_detectors import OutlierDetector
from .consistency_checker import ConsistencyChecker
from .anomaly_scorer import AnomalyScorer

__all__ = [
    'OutlierDetector',
    'ConsistencyChecker',
    'AnomalyScorer'
]
