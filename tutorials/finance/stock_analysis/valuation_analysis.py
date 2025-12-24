# -*- coding: utf-8 -*-
"""
Valuation Analysis Grader for Finance Domain

Evaluates the quality of valuation analysis by comparing two responses based on
conclusion clarity, completeness, and logical rigor.
"""

import textwrap
from typing import Any, Optional

from loguru import logger

from open_judge.graders.base_grader import GraderMode, GraderRank
from open_judge.graders.llm_grader import LLMGrader
from open_judge.models.base_chat_model import BaseChatModel
from open_judge.models.schema.oai.message import ChatMessage
from open_judge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# Chinese Prompt (Primary language for finance domain)
VALUATION_ANALYSIS_PROMPT_ZH = """
你是一个专业的金融评估专家，你需要评估金融助手回答中的估值部分。现在会提供给你一个金融问题以及两位金融助手的回答，同时提供若干评估标准，你需要逐条理解评估标准，根据以下评估步骤，选择更好的答案。

<评估步骤>
1. 先判断回答中有无估值分析，有估值分析的回答优于没有估值分析的回答；如果两个回答都没有估值分析，则根据自己准则判断。
2. 针对每个评估标准选择更好的回答，并给出你的理由；
3. 综合评估每个评估标准下的结果，然后选出综合最优的答案，并给出理由；
</评估步骤>

<评估标准>
评价标准1. 估值结论清晰：评估回答中必须有估值结论，例如偏高/偏低等

评价标准2. 估值完整性：评估回答中包含多个维度的估值分析，包含但不限于：
a. 当前估值与历史估值对比
b. 与同行业可比公司对比
c. 与行业估值中枢对比

评价标准3. 估值逻辑严谨性：评估回答中估值结论有完整的逻辑链，不能只有结论没有论据
</评估标准>

<金融问题>
{query}
</金融问题>

<回答1>
{answer_1}
</回答1>

<回答2>
{answer_2}
</回答2>

# 评分指令
请根据上述评估标准，比较两个回答的质量。
- 如果回答1更好：rank = [1, 2]
- 如果回答2更好：rank = [2, 1]

请按以下结构化 JSON 格式提供你的评估：
{{
    "rank": <[1, 2] 或 [2, 1]>,
    "reason": "<详细解释你的评估理由，包括在各个评估标准下的表现对比>"
}}

JSON:
"""

# English Prompt
VALUATION_ANALYSIS_PROMPT_EN = """
You are a professional financial evaluation expert. You need to assess the valuation section in financial assistants' responses. You will be provided with a financial question and responses from two financial assistants, along with several evaluation criteria. You need to understand each criterion and select the better answer according to the following evaluation steps.

<Evaluation Steps>
1. First, determine whether there is valuation analysis in the responses. A response with valuation analysis is better than one without; if both responses lack valuation analysis, judge based on your own criteria.
2. For each evaluation criterion, select the better response and provide your reasoning;
3. Comprehensively evaluate the results under each criterion, then select the overall best answer and provide reasoning;
</Evaluation Steps>

<Evaluation Criteria>
Criterion 1. Clear Valuation Conclusion: The response must have a valuation conclusion, such as overvalued/undervalued, etc.

Criterion 2. Valuation Completeness: The response includes multi-dimensional valuation analysis, including but not limited to:
a. Current valuation compared to historical valuation
b. Comparison with peer companies in the same industry
c. Comparison with industry valuation benchmark

Criterion 3. Logical Rigor of Valuation: The valuation conclusion has a complete logical chain, cannot have only conclusions without evidence
</Evaluation Criteria>

<Financial Question>
{query}
</Financial Question>

<Answer 1>
{answer_1}
</Answer 1>

<Answer 2>
{answer_2}
</Answer 2>

# Scoring Instructions
Based on the above evaluation criteria, compare the quality of the two answers.
- If Answer 1 is better: rank = [1, 2]
- If Answer 2 is better: rank = [2, 1]

Please provide your evaluation in the following structured JSON format:
{{
    "rank": <[1, 2] or [2, 1]>,
    "reason": "<detailed explanation of your evaluation reasoning, including performance comparison under each evaluation criterion>"
}}

JSON:
"""

# Build default template from prompts
DEFAULT_VALUATION_ANALYSIS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(VALUATION_ANALYSIS_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(VALUATION_ANALYSIS_PROMPT_ZH),
            ),
        ],
    },
)


class ValuationAnalysisGrader(LLMGrader):
    """
    Valuation Analysis Grader for Finance Domain

    Evaluates the quality of valuation analysis by comparing two responses based on
    conclusion clarity (clear valuation judgment), completeness (historical, peer, industry
    comparisons), and logical rigor (evidence-based reasoning).

    Evaluation Criteria:
    1. Clear Conclusion: Must have clear valuation conclusion (overvalued/undervalued)
    2. Completeness: Multi-dimensional comparisons (historical, peers, industry)
    3. Logical Rigor: Complete logical chain with evidence

    Attributes:
        name: Grader name
        model: BaseChatModel instance for evaluation
        template: Evaluation template
        language: Language for evaluation prompts (default: LanguageEnum.ZH for Chinese)

    Example:
        >>> from open_judge.models.openai_chat_model import OpenAIChatModel
        >>> from open_judge.models.schema.prompt_template import LanguageEnum
        >>>
        >>> model = OpenAIChatModel(
        ...     api_key="your-key",
        ...     model="qwen-max",
        ...     generate_kwargs={"temperature": 0.1}
        ... )
        >>>
        >>> grader = ValuationAnalysisGrader(
        ...     model=model,
        ...     language=LanguageEnum.ZH
        ... )
        >>>
        >>> result = await grader.aevaluate(
        ...     query="宁德时代估值是否合理？",
        ...     answer_1="估值偏高。",
        ...     answer_2="结论：估值偏高。PE 60倍，高于历史均值45倍，也高于比亚迪的50倍..."
        ... )
        >>> print(f"Rank: {result.rank}")  # [2, 1] if answer_2 is better
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_VALUATION_ANALYSIS_TEMPLATE,
        language: LanguageEnum = LanguageEnum.ZH,
    ):
        """
        Initialize ValuationAnalysisGrader.

        Args:
            model: The chat model to use for evaluation
            template: The prompt template for valuation analysis evaluation
            language: The language for the evaluation prompt (default: Chinese)
        """
        super().__init__(
            name="valuation_analysis",
            mode=GraderMode.LISTWISE,
            description="Evaluate valuation analysis quality by comparing two responses",
            model=model,
            template=template,
            language=language,
        )
        self.template = template if template is not None else DEFAULT_VALUATION_ANALYSIS_TEMPLATE

    async def aevaluate(
        self,
        query: str,
        answer_1: str,
        answer_2: str,
        **kwargs: Any,
    ) -> GraderRank:
        """
        Evaluate two valuation analysis responses

        Args:
            query: The financial question or prompt
            answer_1: First response to evaluate
            answer_2: Second response to evaluate
            **kwargs: Additional arguments

        Returns:
            GraderRank: Rank result with [1, 2] if answer_1 is better, [2, 1] if answer_2 is better
        """
        try:
            result = await super().aevaluate(
                query=query,
                answer_1=answer_1,
                answer_2=answer_2,
            )

            rank = result.rank
            reason = result.reason

            # Validate rank format
            if not isinstance(rank, list) or len(rank) != 2:
                logger.warning(f"Invalid rank format: {rank}, defaulting to [1, 2]")
                rank = [1, 2]

            # Ensure rank is either [1, 2] or [2, 1]
            if set(rank) != {1, 2}:
                logger.warning(f"Invalid rank values: {rank}, defaulting to [1, 2]")
                rank = [1, 2]

        except Exception as e:
            logger.error(f"Error evaluating valuation analysis: {e}")
            rank = [1, 2]
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "evaluation_type": "valuation_analysis",
            "criteria": ["conclusion_clarity", "completeness", "logical_rigor"],
        }

        return GraderRank(
            name=self.name,
            rank=rank,
            reason=reason,
            metadata=metadata,
        )
