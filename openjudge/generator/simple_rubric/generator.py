# -*- coding: utf-8 -*-
"""Simple rubrics generator implementation.

This module implements a task-description-based approach to generating
evaluation rubrics. It creates LLMGrader instances with rubrics generated
from task descriptions and sample queries.

This is a simpler alternative to the iterative_rubric module, which learns
rubrics from preference data through an iterative refinement process.

Usage:
    >>> from openjudge.generator.simple_rubric import SimpleRubricsGenerator, SimpleRubricsGeneratorConfig
    >>> from openjudge.models.openai_chat_model import OpenAIChatModel
    >>>
    >>> config = SimpleRubricsGeneratorConfig(
    ...     grader_name="Medical QA Grader",
    ...     model=OpenAIChatModel(model="gpt-4o-mini"),
    ...     task_description="Medical question answering system",
    ...     scenario="Healthcare professionals seeking quick answers"
    ... )
    >>> generator = SimpleRubricsGenerator(config)
    >>> grader = await generator.generate(dataset=[], sample_queries=["What are the symptoms of flu?"])
"""

from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

from openjudge.generator.iterative_rubric.query_rubric_generator import (
    LISTWISE_EVALUATION_TEMPLATE,
    POINTWISE_EVALUATION_TEMPLATE,
)
from openjudge.generator.llm_grader_generator import (
    LLMGraderGenerator,
    LLMGraderGeneratorConfig,
)
from openjudge.generator.simple_rubric.rubric_generator import (
    RubricGenerationConfig,
    TaskBasedRubricGenerator,
)
from openjudge.graders.llm_grader import LLMGrader
from openjudge.graders.schema import GraderMode
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.prompt_template import LanguageEnum


@dataclass
class SimpleRubricsGeneratorConfig(LLMGraderGeneratorConfig):
    """Configuration for simple rubrics generator.

    This configuration extends LLMGraderGeneratorConfig with parameters
    specific to task-description-based rubric generation.

    Attributes:
        task_description: Description of the task for evaluation.
                         Should describe what kind of queries and responses are expected.
        scenario: Optional usage scenario for context.
                 Helps the generator understand the evaluation context.
        language: Language for prompts (ZH or EN). Defaults to EN.
        default_rubrics: Fallback rubrics if generation fails.
                        These are used when LLM generation fails.
        max_retries: Maximum number of retry attempts for LLM calls. Defaults to 3.
        min_score: Minimum score for pointwise evaluation. Defaults to 0.
        max_score: Maximum score for pointwise evaluation. Defaults to 1.

    Inherited from LLMGraderGeneratorConfig:
        grader_name: Human-readable name for the generated grader.
        model: Language model to use for generation.
        grader_mode: Mode for the generated grader (POINTWISE or LISTWISE).
        custom_evaluation_prompt: Custom template for evaluation.
    """

    # Task description parameters
    task_description: str = ""
    scenario: Optional[str] = None
    language: LanguageEnum = LanguageEnum.EN

    # Fallback configuration
    default_rubrics: List[str] = field(
        default_factory=lambda: [
            "Accuracy: Whether the response is factually correct",
            "Relevance: Whether the response addresses the query",
            "Completeness: Whether the response is comprehensive",
        ]
    )

    # Generation parameters
    max_retries: int = 3

    # Pointwise-specific parameters
    min_score: int = 0
    max_score: int = 1

    def __post_init__(self):
        """Process model configuration if provided as dict."""
        if isinstance(self.model, dict):
            self.model = OpenAIChatModel(**self.model)


class SimpleRubricsGenerator(LLMGraderGenerator):
    """Generator for creating LLM-based graders with task-description-based rubrics.

    This generator implements a simple approach to rubric generation:
    1. Takes a task description and optional sample queries
    2. Uses an LLM to generate relevant evaluation criteria
    3. Creates an LLMGrader configured with these rubrics

    This is suitable for scenarios where:
    - You have a clear task description
    - You don't have labeled preference data for rubric learning
    - You want a quick way to set up evaluation

    For more sophisticated rubric generation from preference data,
    see the iterative_rubric module.

    Example:
        >>> config = SimpleRubricsGeneratorConfig(
        ...     grader_name="Medical QA Grader",
        ...     model=OpenAIChatModel(model="gpt-4o-mini"),
        ...     task_description="Medical question answering system",
        ...     scenario="Healthcare professionals seeking quick answers"
        ... )
        >>> generator = SimpleRubricsGenerator(config)
        >>> grader = await generator.generate(
        ...     dataset=[],
        ...     sample_queries=["What are the symptoms of flu?"]
        ... )
        >>> # Now use the grader to evaluate responses
        >>> result = await grader.aevaluate(query="...", response="...")
    """

    def __init__(self, config: SimpleRubricsGeneratorConfig) -> None:
        """Initialize the simple rubrics generator.

        Args:
            config: Configuration for rubric generation. Includes:
                - grader_name: Name for the generated grader
                - model: Language model for generation and evaluation
                - task_description: Description of the evaluation task
                - scenario: Optional usage scenario
                - language: Language for prompts (ZH or EN)
                - grader_mode: POINTWISE or LISTWISE
                - default_rubrics: Fallback rubrics if generation fails
        """
        super().__init__(config)
        self.config: SimpleRubricsGeneratorConfig = config

        # Initialize the rubric generator
        rubric_config = RubricGenerationConfig(
            task_description=config.task_description,
            scenario=config.scenario,
            language=config.language,
            default_rubrics=config.default_rubrics,
            max_retries=config.max_retries,
        )
        self._rubric_generator = TaskBasedRubricGenerator(
            config=rubric_config,
            model=config.model,
        )

    async def generate(
        self,
        dataset: List[dict],
        sample_queries: Optional[List[str]] = None,
        **kwargs,
    ) -> LLMGrader:
        """Generate an LLMGrader with rubrics from task description.

        This method generates evaluation rubrics based on the task description
        and creates an LLMGrader instance configured with these rubrics.

        Args:
            dataset: List of data dictionaries. For this generator, the dataset
                    is optional and only used to extract sample queries if
                    sample_queries is not provided.
            sample_queries: Optional list of sample queries for context.
                           If not provided, queries may be extracted from dataset.
            **kwargs: Additional arguments (currently unused).

        Returns:
            LLMGrader: Configured grader instance with generated rubrics.
        """
        # Extract sample queries from dataset if not provided
        if sample_queries is None and dataset:
            sample_queries = [d.get("query", "") for d in dataset[:5] if d.get("query")]

        # Generate rubrics
        rubrics = await self._generate_rubrics(dataset, sample_queries=sample_queries, **kwargs)

        # Prepare grader kwargs
        grader_kwargs = {
            "name": self.config.grader_name,
            "model": self.config.model,
            "mode": self.config.grader_mode,
            "rubrics": rubrics,
            "language": self.config.language,
        }

        # Add min_score and max_score only for pointwise mode
        if self.config.grader_mode == GraderMode.POINTWISE:
            grader_kwargs["min_score"] = self.config.min_score
            grader_kwargs["max_score"] = self.config.max_score

        # Add template: use custom if provided, otherwise use default based on mode
        if self.config.custom_evaluation_prompt is not None:
            grader_kwargs["template"] = self.config.custom_evaluation_prompt
        else:
            # Use default evaluation template based on grader mode
            if self.config.grader_mode == GraderMode.POINTWISE:
                grader_kwargs["template"] = POINTWISE_EVALUATION_TEMPLATE
            else:
                grader_kwargs["template"] = LISTWISE_EVALUATION_TEMPLATE

        return LLMGrader(**grader_kwargs)

    async def _generate_rubrics(
        self,
        dataset: List[dict],
        sample_queries: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """Generate rubrics from task description.

        This method uses the TaskBasedRubricGenerator to create rubrics
        based on the task description and sample queries.

        Args:
            dataset: List of data dictionaries (used for extracting sample queries
                    if sample_queries is not provided).
            sample_queries: Optional list of sample queries for context.
            **kwargs: Additional arguments (currently unused).

        Returns:
            str: Formatted string containing evaluation rubrics.
        """
        # Generate rubrics as list
        rubrics_list = await self._rubric_generator.generate(
            sample_queries=sample_queries,
        )

        # Format rubrics into a string
        formatted_rubrics = "\n\n".join(
            [f"{i + 1}. {rubric}" for i, rubric in enumerate(rubrics_list)]
        )

        logger.info(f"Generated {len(rubrics_list)} rubrics from task description")

        return formatted_rubrics

