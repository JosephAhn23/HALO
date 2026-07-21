"""
dataset_engineering/
----------------------
Production dataset engineering for LLM pipelines:

  - DatasetVersion   : DVC-backed versioning with lineage tracking
  - QualityChecker   : schema validation, dedup, drift detection
  - SyntheticGen     : LLM-powered synthetic QA pair generation
  - FeatureStore     : lightweight feature registry with versioning
"""

from src.dataset_engineering.feature_store import (
    FeatureDefinition,
    FeatureSnapshot,
    FeatureSpec,  # alias for FeatureDefinition
    FeatureStore,
    FeatureVector,  # alias for FeatureSnapshot
)
from src.dataset_engineering.quality import QualityChecker, QualityIssue, QualityReport
from src.dataset_engineering.synthetic import SyntheticDataset, SyntheticQA, SyntheticQAGenerator
from src.dataset_engineering.versioning import DatasetLineage, DatasetVersion

# Backwards-compatible aliases
DataQualityChecker = QualityChecker
SyntheticDataGenerator = SyntheticQAGenerator

__all__ = [
    "DatasetVersion",
    "DatasetLineage",
    "QualityChecker",
    "QualityReport",
    "QualityIssue",
    "SyntheticQAGenerator",
    "SyntheticDataset",
    "SyntheticQA",
    "FeatureStore",
    "FeatureDefinition",
    "FeatureSnapshot",
    "FeatureSpec",
    "FeatureVector",
    # aliases
    "DataQualityChecker",
    "SyntheticDataGenerator",
]
