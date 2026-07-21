def __getattr__(name):
    """Lazy imports to avoid loading mlflow/torch at import time."""
    _lazy = {
        "Pipeline": ("src.agents.orchestrator", "Pipeline"),
        "run_pipeline": ("src.agents.orchestrator", "run_pipeline"),
        "get_pipeline": ("src.agents.orchestrator", "get_pipeline"),
        "RetrieverAgent": ("src.agents.retriever", "RetrieverAgent"),
        "RetrievedChunk": ("src.agents.retriever", "RetrievedChunk"),
        "RerankerAgent": ("src.agents.reranker", "RerankerAgent"),
        "CrossEncoderReranker": ("src.agents.reranker", "CrossEncoderReranker"),
        "SynthesizerAgent": ("src.agents.synthesizer", "SynthesizerAgent"),
        "Retriever": ("src.agents.protocols", "Retriever"),
        "Reranker": ("src.agents.protocols", "Reranker"),
        "Synthesizer": ("src.agents.protocols", "Synthesizer"),
    }
    if name in _lazy:
        import importlib

        module_name, attr = _lazy[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    raise AttributeError(f"module 'agents' has no attribute {name!r}")


__all__ = [
    "Pipeline",
    "run_pipeline",
    "get_pipeline",
    "RetrieverAgent",
    "RetrievedChunk",
    "RerankerAgent",
    "CrossEncoderReranker",
    "SynthesizerAgent",
    "Retriever",
    "Reranker",
    "Synthesizer",
]
