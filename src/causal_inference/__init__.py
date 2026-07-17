"""
causal_inference/
-----------------
Causal analysis for retrieval quality:
  - Uplift modeling (does reranking *cause* better RAGAS scores?)
  - ATE / CATE estimation via DoWhy + EconML
  - Counterfactual retrieval evaluation
"""

from src.causal_inference.uplift import UpliftEstimator, UpliftConfig, UpliftResults
from src.causal_inference.retrieval_effect import RetrievalCausalAnalyzer, CausalEffect
from src.causal_inference.counterfactual import CounterfactualEvaluator, CounterfactualResult

__all__ = [
    "UpliftEstimator",
    "UpliftConfig",
    "UpliftResults",
    "RetrievalCausalAnalyzer",
    "CausalEffect",
    "CounterfactualEvaluator",
    "CounterfactualResult",
]
