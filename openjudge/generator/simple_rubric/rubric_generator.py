# -*- coding: utf-8 -*-
"""Task-based rubric generator for automatic evaluation criteria generation.

This module provides functionality to automatically generate evaluation rubrics
based on task descriptions, enabling zero-shot evaluation pipelines.

The generator uses an LLM to analyze the task description and sample queries
to produce relevant evaluation criteria without requiring labeled training data.

Classes:
    RubricGenerationConfig: Configuration for rubric generation.
    TaskBasedRubricGenerator: Generator for evaluation rubrics.
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed

from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# =============================================================================
# Prompt Templates
# =============================================================================

RUBRIC_GENERATION_PROMPT_EN = """# Task
Generate evaluation rubrics for pairwise comparison of model responses.

## Task Description
{task_description}

## Scenario
{scenario}

## Sample Queries (for context)
{sample_queries}

## Requirements
- Generate 3-5 clear evaluation criteria for comparing two responses
- Each criterion should be objective and measurable
- Criteria should be relevant to the task and scenario
- Focus on aspects that distinguish good responses from poor ones

## Output Format
Return a JSON object with:
- rubrics: list of evaluation criteria strings
- reason: brief explanation of why these criteria are important

Example:
{{
    "rubrics": [
        "Accuracy: Whether the response contains correct and factual information",
        "Completeness: Whether the response fully addresses the query",
        "Clarity: Whether the response is well-organized and easy to understand"
    ],
    "reason": "These criteria capture the key aspects for evaluating..."
}}
"""

RUBRIC_GENERATION_PROMPT_ZH = """# 任务
为模型回答的成对比较生成评估标准。

## 任务描述
{task_description}

## 使用场景
{scenario}

## 示例查询（用于上下文理解）
{sample_queries}

## 要求
- 生成3-5个清晰的评估标准用于比较两个回答
- 每个标准应该客观且可测量
- 标准应与任务和场景相关
- 聚焦于能够区分好回答和差回答的方面

## 输出格式
返回一个JSON对象，包含：
- rubrics: 评估标准字符串列表
- reason: 简要解释为什么这些标准是重要的

示例：
{{
    "rubrics": [
        "准确性：回答是否包含正确和真实的信息",
        "完整性：回答是否完整地解决了问题",
        "清晰度：回答是否组织良好、易于理解"
    ],
    "reason": "这些标准捕捉了评估的关键方面..."
}}
"""

RUBRIC_GENERATION_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content="You are an expert at designing evaluation criteria for AI systems.",
            ),
            ChatMessage(role="user", content=RUBRIC_GENERATION_PROMPT_EN),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="system",
                content="你是一位设计AI系统评估标准的专家。",
            ),
            ChatMessage(role="user", content=RUBRIC_GENERATION_PROMPT_ZH),
        ],
    },
)


# =============================================================================
# Output Schema
# =============================================================================


class RubricGenerationOutput(BaseModel):
    """Output schema for rubric generation."""

    rubrics: List[str] = Field(..., description="List of evaluation rubrics")
    reason: str = Field(default="", description="Reasoning for these rubrics")


# =============================================================================
# Configuration
# =============================================================================


class RubricGenerationConfig(BaseModel):
    """Configuration for rubric generation.

    Attributes:
        task_description: Description of the task for evaluation.
                         Should describe what kind of queries and responses are expected.
        scenario: Optional usage scenario for context.
                 Helps the generator understand the evaluation context.
        language: Language for prompts (ZH or EN). Defaults to EN.
        default_rubrics: Fallback rubrics if generation fails.
                        These are used when LLM generation fails.
        max_retries: Maximum number of retry attempts for LLM calls. Defaults to 3.
    """

    task_description: str = Field(..., description="Task description")
    scenario: Optional[str] = Field(default=None, description="Usage scenario")
    language: LanguageEnum = Field(default=LanguageEnum.EN, description="Language for prompts")
    default_rubrics: List[str] = Field(
        default=[
            "Accuracy: Whether the response is factually correct",
            "Relevance: Whether the response addresses the query",
            "Completeness: Whether the response is comprehensive",
        ],
        description="Fallback rubrics if generation fails",
    )
    max_retries: int = Field(default=3, description="Maximum retry attempts")


# =============================================================================
# TaskBasedRubricGenerator
# =============================================================================


class TaskBasedRubricGenerator:
    """Generate evaluation rubrics based on task description.

    This generator creates evaluation rubrics that can be used for pairwise
    comparison or other evaluation scenarios. It uses an LLM to generate
    task-specific criteria based on the provided task description.

    This is a core utility class that generates rubrics as a list of strings.
    For creating complete LLMGrader instances, use SimpleRubricsGenerator instead.

    Attributes:
        config: Rubric generation configuration
        model: Language model for generation

    Example:
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.generator.simple_rubric import TaskBasedRubricGenerator, RubricGenerationConfig
        >>>
        >>> config = RubricGenerationConfig(
        ...     task_description="Medical question answering system",
        ...     scenario="Healthcare professionals seeking quick answers"
        ... )
        >>> model = OpenAIChatModel(model="gpt-4o-mini")
        >>> generator = TaskBasedRubricGenerator(config=config, model=model)
        >>> rubrics = await generator.generate(sample_queries=["What are the symptoms of flu?"])
    """

    def __init__(
        self,
        config: RubricGenerationConfig,
        model: BaseChatModel,
    ):
        """Initialize TaskBasedRubricGenerator.

        Args:
            config: Rubric generation configuration
            model: Language model for generating rubrics
        """
        self.config = config
        self.model = model

    @classmethod
    def from_dict(
        cls,
        config_dict: Dict[str, Any],
        model: Optional[BaseChatModel] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> "TaskBasedRubricGenerator":
        """Create TaskBasedRubricGenerator from dictionary configuration.

        Args:
            config_dict: Configuration dictionary with task_description, scenario, etc.
            model: Pre-initialized model (optional)
            model_config: Model configuration dict if model not provided

        Returns:
            TaskBasedRubricGenerator instance
        """
        config = RubricGenerationConfig(**config_dict)

        if model is None:
            if model_config is None:
                raise ValueError("Either model or model_config must be provided")
            model = OpenAIChatModel(**model_config)

        return cls(config=config, model=model)

    async def generate(
        self,
        sample_queries: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate evaluation rubrics.

        Args:
            sample_queries: Optional sample queries for context.
                           These help the LLM understand what kind of
                           queries will be evaluated.

        Returns:
            List of rubric strings
        """

        @retry(stop=stop_after_attempt(self.config.max_retries), wait=wait_fixed(1.0))
        async def _generate() -> List[str]:
            queries_text = "None provided"
            if sample_queries:
                queries_text = "\n".join(f"- {q}" for q in sample_queries[:5])

            messages = RUBRIC_GENERATION_TEMPLATE.format(
                task_description=self.config.task_description,
                scenario=self.config.scenario or "General usage",
                sample_queries=queries_text,
                language=self.config.language,
            )

            response = await self.model.achat(
                messages=list(messages),
                structured_model=RubricGenerationOutput,
            )

            if not response.parsed or "rubrics" not in response.parsed:
                raise ValueError("Failed to parse rubric generation response")

            return response.parsed["rubrics"]

        try:
            rubrics = await _generate()
            logger.info(f"Generated {len(rubrics)} evaluation rubrics")
            for i, rubric in enumerate(rubrics, 1):
                logger.debug(f"  {i}. {rubric}")
            return rubrics
        except Exception as e:
            logger.error(f"Rubric generation failed: {e}")
            # Return default rubrics as fallback
            logger.warning("Using default rubrics as fallback")
            return self.config.default_rubrics

