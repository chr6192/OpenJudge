# LangSmith Integration Guide

## Overview

LangSmith is an LLM application development and monitoring platform provided by LangChain. It adopts a "contract-based" integration model, treating evaluators as part of the experimental pipeline.

## Integration Principles

LangSmith deeply integrates into the runtime flow, requiring evaluators to be packaged as Python Callables that accept specific inputs and return specific outputs. OpenJudge can be easily integrated with LangSmith by wrapping OpenJudge graders as LangSmith-compatible evaluators.

## Quick Start: Integrating with Individual Graders

### 1. Install Dependencies

To begin integrating OpenJudge with LangSmith, first install the required dependencies:

```bash
pip install langsmith
```

### 2. Create OpenJudge Evaluator Wrapper

The first step is to create a wrapper function that converts OpenJudge graders into LangSmith-compatible evaluators. This wrapper handles the conversion of data formats and result types between the two systems.

```python
from typing import Callable, Dict, Any, Union
from open_judge.graders.base_grader import BaseGrader
from open_judge.graders.schema import GraderResult, GraderScore, GraderRank, GraderError

def create_langsmith_evaluator(grader: BaseGrader) -> Callable:
    """
    Create a LangSmith-compatible evaluator from an OpenJudge grader.

    This function wraps an OpenJudge grader to make it compatible with LangSmith's
    evaluation interface. It handles data transformation and result formatting.

    Args:
        grader: An OpenJudge grader instance

    Returns:
        A LangSmith-compatible evaluator function
    """
    async def langsmith_evaluator(run, example) -> Dict[str, Union[float, str]]:
        """
        LangSmith evaluator function.

        This function extracts data from LangSmith's run and example objects,
        passes it to the OpenJudge grader, and formats the result for LangSmith.

        Args:
            run: The run object from LangSmith containing outputs
            example: The example object from LangSmith containing inputs

        Returns:
            A dictionary containing the evaluation results with score and reasoning
        """
        try:
            # Extract input data from the example
            inputs = example.inputs
            # Extract output data from the run
            outputs = run.outputs or {}

            # Combine inputs and outputs for grader evaluation
            # This creates a single dictionary with all available data
            evaluation_data = {**inputs, **outputs}

            # Execute OpenJudge evaluation
            # This calls the grader's aevaluate method with the combined data
            result: GraderResult = await grader.aevaluate(**evaluation_data)

            # Convert OpenJudge result to LangSmith format
            # Different result types need different handling
            if isinstance(result, GraderScore):
                return {
                    "score": result.score,
                    "reasoning": getattr(result, 'reason', '')
                }
            elif isinstance(result, GraderRank):
                return {
                    "score": getattr(result, 'rank', 0),
                    "reasoning": getattr(result, 'reason', '')
                }
            elif isinstance(result, GraderError):
                return {
                    "score": 0.0,
                    "reasoning": f"Error: {result.error}"
                }
            else:
                return {
                    "score": 0.0,
                    "reasoning": "Unknown result type"
                }
        except Exception as e:
            # Handle any unexpected errors during evaluation
            return {
                "score": 0.0,
                "reasoning": f"Evaluation failed: {str(e)}"
            }

    return langsmith_evaluator

# Example usage
from open_judge.graders.text.similarity import SimilarityGrader

# Create OpenJudge grader for cosine similarity
similarity_grader = SimilarityGrader(algorithm="cosine")
```

### 3. Use OpenJudge Evaluators in LangSmith

After creating the wrapper, you can use OpenJudge graders in LangSmith evaluations. This example shows how to set up and run an evaluation with multiple graders.

```python
from langsmith import Client
from langsmith.evaluation import evaluate
from open_judge.graders.text.relevance_grader import RelevanceGrader

# Initialize LangSmith client
# Make sure you have set the LANGSMITH_API_KEY environment variable
client = Client()

# Create multiple OpenJudge evaluators
# This demonstrates how to use different types of graders
graders = {
    "similarity": SimilarityGrader(algorithm="cosine"),
    "relevance": RelevanceGrader()
}

# Convert to LangSmith evaluators
# Each OpenJudge grader is wrapped to be compatible with LangSmith
langsmith_evaluators = {
    name: create_langsmith_evaluator(grader)
    for name, grader in graders.items()
}

# Run evaluation
# This executes the evaluation using LangSmith's evaluation framework
results = evaluate(
    <your_target_task>,  # Your LLM application or chain
    data=<your_dataset_name_or_id>,  # Dataset in LangSmith
    evaluators=list(langsmith_evaluators.values()),
    experiment_prefix="open_judge-evaluation"
)
```

## Advanced Usage: Integrating with GradingRunner

### Batch Evaluation with Multiple Graders

For more complex scenarios involving multiple graders, OpenJudge's GradingRunner provides efficient batch processing capabilities. This approach offers better performance and resource management compared to individual grader evaluation.

```python
from open_judge.runner.grading_runner import GradingRunner
from open_judge.graders.text.similarity import SimilarityGrader
from open_judge.graders.text.relevance_grader import RelevanceGrader

class LangSmithBatchEvaluator:
    """Batch evaluator that combines multiple OpenJudge graders"""

    def __init__(self):
        """
        Initialize the batch evaluator with a GradingRunner.

        The GradingRunner handles concurrent execution of multiple graders
        and provides built-in batching capabilities.
        """
        # Configure the runner with multiple graders
        self.runner = GradingRunner(
            grader_configs={
                "similarity": SimilarityGrader(algorithm="cosine"),
                "relevance": RelevanceGrader(),
                # Add more graders as needed
            },
            max_concurrency=10  # Control concurrency to manage resource usage
        )

    async def __call__(self, run, example) -> Dict[str, Dict[str, float]]:
        """
        LangSmith batch evaluator function.

        This function prepares data for batch processing, executes the evaluation
        using GradingRunner, and formats the results for LangSmith.

        Args:
            run: The run object from LangSmith
            example: The example object from LangSmith

        Returns:
            A dictionary containing results from all graders
        """
        try:
            # Prepare data for batch processing
            # Even though we're processing one item, we need to wrap it in a list
            inputs = example.inputs
            outputs = run.outputs or {}
            evaluation_data = [{**inputs, **outputs}]

            # Execute batch evaluation using OpenJudge runner
            # The runner handles concurrent execution of all configured graders
            batch_results = await self.runner.arun(evaluation_data)

            # Convert results to LangSmith format
            # Process results from each grader
            formatted_results = {}
            for grader_name, grader_results in batch_results.items():
                if grader_results:  # Check if results exist
                    result = grader_results[0]  # We only have one sample
                    if isinstance(result, GraderScore):
                        formatted_results[grader_name] = {
                            "score": result.score,
                            "reasoning": getattr(result, 'reason', '')
                        }
                    elif isinstance(result, GraderRank):
                        formatted_results[grader_name] = {
                            "score": getattr(result, 'rank', 0),
                            "reasoning": getattr(result, 'reason', '')
                        }
                    elif isinstance(result, GraderError):
                        formatted_results[grader_name] = {
                            "score": 0.0,
                            "reasoning": f"Error: {result.error}"
                        }
                    else:
                        formatted_results[grader_name] = {
                            "score": 0.0,
                            "reasoning": "Unknown result type"
                        }

            return formatted_results

        except Exception as e:
            # Handle any errors during batch evaluation
            return {
                "error": {
                    "score": 0.0,
                    "reasoning": f"Batch evaluation failed: {str(e)}"
                }
            }

# Usage
# Create an instance of the batch evaluator
batch_evaluator = LangSmithBatchEvaluator()

# Run evaluation with batch evaluator
# This uses the batch evaluator instead of individual wrappers
results = evaluate(
    <your_target_task>,
    data=<your_dataset_name_or_id>,
    evaluators=[batch_evaluator],  # Single batch evaluator handles multiple graders
    experiment_prefix="open_judge-batch-evaluation"
)
```

### Working with Aggregated Results

OpenJudge's GradingRunner also supports result aggregation, allowing you to combine multiple evaluation metrics into composite scores. This is particularly useful when you want to create overall quality measures from individual metrics.

```python
from open_judge.runner.grading_runner import GradingRunner
from open_judge.analyzer.aggregator.weighted_sum_aggregator import WeightedSumAggregator
from open_judge.graders.text.similarity import SimilarityGrader
from open_judge.graders.text.relevance_grader import RelevanceGrader

class AggregatedLangSmithEvaluator:
    """Evaluator that uses aggregated results from OpenJudge runner"""

    def __init__(self):
        """
        Initialize the evaluator with graders and aggregators.

        Aggregators automatically combine results from multiple graders
        according to specified rules (e.g., weighted sums).
        """
        # Configure runner with graders and aggregators
        self.runner = GradingRunner(
            grader_configs={
                "similarity": SimilarityGrader(algorithm="cosine"),
                "relevance": RelevanceGrader(),
            },
            max_concurrency=10,
            # Add aggregators for result combination
            aggregators=[
                WeightedSumAggregator(weights={"similarity": 0.6, "relevance": 0.4})
            ]
        )

    async def __call__(self, run, example) -> Dict[str, Dict[str, float]]:
        """
        Execute evaluation with both individual and aggregated results.

        This function demonstrates how to work with both raw grader results
        and aggregated scores produced by OpenJudge's built-in aggregators.
        """
        try:
            # Prepare data for evaluation
            inputs = example.inputs
            outputs = run.outputs or {}
            evaluation_data = [{**inputs, **outputs}]

            # Execute evaluation with aggregation
            # The runner automatically applies aggregators to the results
            batch_results = await self.runner.arun(evaluation_data)

            # Format individual results
            # These are the raw results from each grader
            formatted_results = {}
            for grader_name, grader_results in batch_results.items():
                if grader_results:
                    result = grader_results[0]
                    if isinstance(result, (GraderScore, GraderRank)):
                        formatted_results[grader_name] = {
                            "score": getattr(result, 'score', getattr(result, 'rank', 0)),
                            "reasoning": getattr(result, 'reason', '')
                        }
                    elif isinstance(result, GraderError):
                        formatted_results[grader_name] = {
                            "score": 0.0,
                            "reasoning": f"Error: {result.error}"
                        }

            # Format aggregated results
            # Aggregated results are prefixed with "aggregated_" in the results
            for key in batch_results.keys():
                if key.startswith("aggregated_"):
                    # Extract the aggregator name from the key
                    agg_name = key[len("aggregated_"):]
                    if batch_results[key]:
                        agg_result = batch_results[key][0]
                        if isinstance(agg_result, (GraderScore, GraderRank)):
                            formatted_results[agg_name] = {
                                "score": getattr(agg_result, 'score', getattr(agg_result, 'rank', 0)),
                                "reasoning": getattr(agg_result, 'reason', '')
                            }
                        elif isinstance(agg_result, GraderError):
                            formatted_results[agg_name] = {
                                "score": 0.0,
                                "reasoning": f"Aggregation error: {agg_result.error}"
                            }

            return formatted_results

        except Exception as e:
            # Handle errors in aggregated evaluation
            return {
                "error": {
                    "score": 0.0,
                    "reasoning": f"Aggregated evaluation failed: {str(e)}"
                }
            }
```

## Tips

-  **Grader Error Handling**: Always check for `GraderError` results when using graders. These indicate evaluation failures that should be handled gracefully rather than accessing non-existent attributes.

-  **Grader Result Types**: Understand the different result types (`GraderScore`, `GraderRank`, `GraderError`) that graders can return and handle each appropriately in your integration code.

-  **Runner Concurrency**: Use the `max_concurrency` parameter in `GradingRunner` to control how many graders run simultaneously, preventing resource exhaustion.

-  **Batch Processing with Runner**: For evaluating multiple data points, use `GradingRunner.arun()` to process them in batches rather than calling graders individually for better performance.

-  **Runner Aggregators**: Take advantage of built-in aggregators in `GradingRunner` rather than implementing your own result aggregation logic.

-  **Async Grader Methods**: Always use the async methods (e.g., `aevaluate`) when working with graders to ensure proper asynchronous execution.

## Related Resources

- [OpenJudge Runner Documentation](../running_graders/run_tasks.md)
- [LangSmith Official Documentation](https://docs.smith.langchain.com/)