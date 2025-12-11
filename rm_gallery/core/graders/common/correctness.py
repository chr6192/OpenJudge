# -*- coding: utf-8 -*-
"""
Correctness Grader

Evaluates whether model response matches the provided reference response (correct response),
including factual consistency, information coverage, and appropriate alignment.
"""

import textwrap
from typing import Optional

from loguru import logger

from rm_gallery.core.graders.base_grader import GraderMode, GraderScore
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models.base_chat_model import BaseChatModel
from rm_gallery.core.models.schema.message import ChatMessage
from rm_gallery.core.models.schema.prompt_template import LanguageEnum, PromptTemplate

# English Prompt
CORRECTNESS_PROMPT_EN = """
You are a professional data annotator responsible for evaluating whether the model response matches the provided correct response (reference response). Your task is to score according to the following criteria:

<Scoring Criteria>
A response that perfectly matches the reference response should:
- Maintain factual consistency with all information in the reference response.
- Include key points from the reference response that are relevant to the query.
- Match the style, tone, and format of the reference response when appropriate.
- Not contradict, misrepresent, or distort information from the reference response.
- Properly ground claims in the reference response without fabricating details.
- Use reference response information accurately without taking it out of context.
- Balance reference response adherence with responding appropriately to the specific query.

Points should be deducted for:
- Factual contradictions with the reference response.
- Omitting critical information present in the reference response.
- Misrepresenting or distorting reference response information.
- Adding claims not supported by the reference response when grounding is required.
- Significantly departing from reference response style/format when matching is expected.
- Taking reference response information out of context.
- Over-relying on reference response when original synthesis is needed.
</Scoring Criteria>

<Guidance>
- Carefully read the reference response to understand its key facts, style, and content.
- Compare each claim in the response against the reference response.
- Check if the response appropriately balances using reference response info vs. answering the specific question.
- Evaluate whether the response matches the reference response's level of detail and confidence.
- Consider if the response properly attributes or grounds information in the reference response.
- Assess whether the response adds appropriate synthesis or only paraphrases.
</Guidance>

<Reminder>
The goal is to evaluate correctness against reference response, not general quality. A well-written response that contradicts the reference response should score low. A simple response that accurately reflects and properly uses the reference response should score high. Consider both accuracy and appropriate application of the reference response.
</Reminder>

{context_section}

<query>
{query}
</query>

<response>
{response}
</response>

{reference_section}

# Output Instructions
Provide your evaluation in the following structured JSON format:
{{
    "score": <integer between 1 and 5, where 5 means perfect match with reference response and 1 means complete deviation from reference response>,
    "reason": "<brief explanation for the assigned score, specifically mentioning how the response aligns with or deviates from the reference response>"
}}

Scoring Scale:
- 5: Perfect match with reference response in all aspects
- 4: Strong match with minor stylistic differences
- 3: Partially matches reference response but with notable deviations
- 2: Significant departures or misrepresentation of reference response
- 1: Completely ignores or contradicts reference response

JSON:
"""

# Chinese Prompt
CORRECTNESS_PROMPT_ZH = """
你是一名专业的数据标注员，负责评估模型输出是否与提供的正确回复（reference response）一致。你的任务是根据以下标准进行评分：

<评分标准>
完美匹配 reference response 的回答应该：
- 与 reference response 中的所有信息保持事实一致性。
- 包含 reference response 中与输入问题相关的关键点。
- 在适当时与 reference response 的风格、语气和格式相匹配。
- 不与 reference response 中的信息矛盾、歪曲或扭曲。
- 在 reference response 中正确地支撑声明，而不捏造细节。
- 准确使用 reference response 信息，不脱离上下文。
- 在遵循 reference response 和适当回答特定输入之间取得平衡。

以下情况应扣分：
- 与 reference response 的事实矛盾。
- 遗漏 reference response 中存在的关键信息。
- 歪曲或扭曲 reference response 信息。
- 在需要支撑时添加 reference response 不支持的声明。
- 在预期匹配时明显偏离 reference response 风格/格式。
- 脱离上下文使用 reference response 信息。
- 在需要原创综合时过度依赖 reference response。
</评分标准>

<指导>
- 仔细阅读 reference response 以理解其关键事实、风格和内容。
- 将输出中的每个声明与 reference response 进行比较。
- 检查输出是否适当地平衡使用 reference response 信息和回答特定问题。
- 评估输出是否与 reference response 的细节水平和可信度相匹配。
- 考虑输出是否正确地在 reference response 中归因或支撑信息。
- 评估输出是否添加了适当的综合还是仅仅转述。
</指导>

<提醒>
目标是评估与 reference response 的正确性，而不是一般质量。一个写得很好但与 reference response 矛盾的回答应该得分低。一个简单但准确反映并正确使用 reference response 的回答应该得分高。同时考虑准确性和 reference response 的适当应用。
</提醒>

{context_section}

<query>
{query}
</query>

<response>
{response}
</response>

{reference_section}

# 输出指令
请按以下结构化 JSON 格式提供你的评估：
{{
    "score": <1到5之间的整数，其中5表示完美匹配 reference response，1表示完全偏离 reference response>,
    "reason": "<对所给分数的简要解释，特别提到输出如何与 reference response 一致或偏离>"
}}

评分标尺：
- 5: 在所有方面都完美匹配 reference response
- 4: 强烈匹配，仅有轻微的风格差异
- 3: 部分匹配 reference response，但有明显偏离
- 2: 明显偏离或误述 reference response 内容
- 1: 完全忽略或与 reference response 矛盾

JSON:
"""

# Build default template from prompts
DEFAULT_CORRECTNESS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(CORRECTNESS_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(CORRECTNESS_PROMPT_ZH),
            ),
        ],
    },
)


class CorrectnessGrader(LLMGrader):
    """
    Correctness Grader

    Purpose:
        Evaluates how well model outputs match the provided correct response (reference response),
        ensuring factual accuracy, proper information coverage, and appropriate alignment
        with the expected correct answer.

    What it evaluates:
        - Factual Consistency: Output aligns with facts in reference response
        - Information Coverage: Key points from reference response are appropriately included
        - Style/Format Matching: Output matches reference response style when required
        - Proper Attribution: reference response information used without misrepresentation
        - Grounding Quality: Output properly grounded in reference response
        - Omission Detection: Identifies when critical information is missing

    When to use:
        - Evaluating model responses against known correct answers
        - Assessing factual accuracy in Q&A systems
        - Validating outputs in knowledge-based tasks
        - Comparing generated content against gold standard responses
        - Quality assurance for information retrieval systems
        - Educational content evaluation

    Scoring:
        - 5: Perfect match with reference response in all aspects
        - 4: Strong match with minor stylistic differences
        - 3: Partially matches reference response but with notable deviations
        - 2: Significant departures or misrepresentation of reference response
        - 1: Completely ignores or contradicts reference response

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        threshold: Minimum score [0, 1] to pass (default: 0.7)
        template: Custom evaluation template (default: DEFAULT_CORRECTNESS_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)

    Returns:
        GraderScore object with:
            - score: Score [1, 5] where 5 = perfect match, 1 = complete deviation
            - reason: Explanation of how well output matches reference response
            - metadata: Threshold and evaluation details

    Example:
        >>> from rm_gallery.core.model.openai_llm import OpenAIChatModel
        >>> from rm_gallery.core.llm_judge import CorrectnessGrader
        >>>
        >>> # Initialize grader
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-max")
        >>> grader = CorrectnessGrader(model=model, threshold=0.7)
        >>>
        >>> # Good match
        >>> result = await grader.aevaluate(
        ...     query="When was the product launched?",
        ...     response="The product launched in Q1 2023 in Europe, capturing 50% market share.",
        ...     reference_response="Product launched Q1 2023 in Europe with 50% market share."
        ... )
        >>> print(result.score)  # 5 - accurate to reference response
        >>>
        >>> # Poor match
        >>> result = await grader.aevaluate(
        ...     query="When and where was the product launched?",
        ...     response="The product was launched in early 2023 in European markets.",
        ...     reference_response="The product was launched in Q1 2023 in Europe."
        ... )
        >>> print(result.score)  # 2 - deviates from reference response
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        threshold: float = 0.7,
        template: Optional[PromptTemplate] = DEFAULT_CORRECTNESS_TEMPLATE,
        language: LanguageEnum = LanguageEnum.EN,
    ):
        """
        Initialize CorrectnessGrader

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            threshold: Success threshold [0, 1] (default: 0.7)
            template: PromptTemplate for evaluation prompts (default: DEFAULT_CORRECTNESS_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
        """
        super().__init__(
            name="correctness",
            mode=GraderMode.POINTWISE,
            description="Evaluate whether response matches the provided reference response",
            model=model,
            template=template,
            language=language,
        )
        self.threshold = threshold

    async def aevaluate(
        self,
        query: str,
        response: str,
        reference_response: str = "",
        context: str = "",
    ) -> GraderScore:
        """
        Evaluate correctness of response against reference response

        Args:
            query: Original user query or question
            response: Model response to evaluate
            reference_response: Correct response to compare against. Defaults to empty string.
            context: Additional context or background information. Defaults to empty string.

        Returns:
            GraderScore: Score with correctness value [1, 5]
                        where 5 means perfect match with reference response,
                        1 means complete deviation from reference response

        Example:
            >>> result = await grader.aevaluate(
            ...     query="What is the capital of France?",
            ...     response="Paris is the capital of France.",
            ...     reference_response="The capital of France is Paris, with a population of 2.2M.",
            ...     context="Geography quiz question"
            ... )
        """
        # Prepare context section
        context_section = ""
        if context:
            if self.language == LanguageEnum.ZH:
                context_section = f"""<context>
{context}
</context>"""
            else:
                context_section = f"""<context>
{context}
</context>"""

        # Prepare reference response section based on language
        reference_section = ""
        if reference_response:
            if self.language == LanguageEnum.ZH:
                reference_section = f"""以下是正确的回复供你参考：
<reference>
{reference_response}
</reference>"""
            else:
                reference_section = f"""The following is the correct response for your reference:
<reference>
{reference_response}
</reference>"""

        try:
            result = await super().aevaluate(
                query=query,
                response=response,
                context_section=context_section,
                reference_section=reference_section,
            )
            score = result.score
            reason = result.reason

        except Exception as e:
            logger.error(f"Error evaluating correctness: {e}")
            score = 0.0
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "threshold": self.threshold,
        }

        return GraderScore(
            name=self.name,
            score=score,
            reason=reason,
            metadata=metadata,
        )


__all__ = ["CorrectnessGrader", "DEFAULT_CORRECTNESS_TEMPLATE"]
