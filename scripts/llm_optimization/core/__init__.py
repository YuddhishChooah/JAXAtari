"""
Core helpers for the active LeGPS thesis pipeline.
"""

from .optimizers import (
    CMAESOptimizer,
    BayesianOptimizer,
    RandomSearchOptimizer,
    get_optimizer,
    params_to_vector,
    vector_to_params,
)

from .evaluator import (
    ParallelEvaluator,
    create_evaluator,
    EvaluationMetrics,
)

from .llm_client import (
    LLMClient,
    LLMConfig,
    create_llm_client,
)

from .logger import (
    Logger,
    PongLogger,
    FreewayLogger,
    AsterixLogger,
    AblationConfig,
    create_logger,
    get_all_metric_descriptions,
    format_metrics_for_llm,
)

__all__ = [
    # Optimizers
    "CMAESOptimizer",
    "BayesianOptimizer",
    "RandomSearchOptimizer",
    "get_optimizer",
    "params_to_vector",
    "vector_to_params",
    # Evaluator
    "ParallelEvaluator",
    "create_evaluator",
    "EvaluationMetrics",
    # LLM Client
    "LLMClient",
    "LLMConfig",
    "create_llm_client",
    # Logger
    "Logger",
    "PongLogger",
    "FreewayLogger",
    "AsterixLogger",
    "AblationConfig",
    "create_logger",
    "get_all_metric_descriptions",
    "format_metrics_for_llm",
]
