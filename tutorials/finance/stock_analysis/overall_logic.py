# -*- coding: utf-8 -*-
"""
Overall Logic Grader for Finance Domain

Evaluates the overall logical structure and coherence of financial analysis responses
by comparing two responses based on clarity, completeness, and consistency.
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
OVERALL_LOGIC_PROMPT_ZH = """
你是一个专业的金融评估专家，你需要评估金融助手回答的整体逻辑。现在会提供给你一个金融问题以及两位金融助手的回答，同时提供若干评估标准，你需要逐条理解评估标准，根据以下评估步骤，选择更好的答案。

<评估步骤>
1. 针对每个评估标准选择更好的回答，并给出你的理由；
2. 综合评估每个评估标准下的结果，然后选出综合最优的答案，并给出理由；
</评估步骤>

<评估标准>
评价标准1. 整体逻辑是否清晰：评估答案整体逻辑是否清晰，包括但不限于：
a. 是否采用"总-分-总"的格式，即开头说出结论，中间是各个子论点，结尾给出总结；

评价标准2. 整体逻辑是否完备：评估答案整体逻辑相对金融问题是否完备，包括但不限于：
a. 子论点是否足够推导出结论；
b. 整体逻辑是否全面回答了金融问题;

评价标准3. 整体逻辑是否自洽：评估答案整体逻辑是否自洽，包括但不限于：
a. 各子论点应和结论有逻辑关系，非孤立罗列，确保逻辑自洽；
b. 各子论点内容没有重叠，从不同角度论证结论；
c. 各子论点内容没有矛盾或者冲突
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
OVERALL_LOGIC_PROMPT_EN = """
You are a professional financial evaluation expert. You need to assess the overall logic of financial assistants' responses. You will be provided with a financial question and responses from two financial assistants, along with several evaluation criteria. You need to understand each criterion and select the better answer according to the following evaluation steps.

<Evaluation Steps>
1. For each evaluation criterion, select the better response and provide your reasoning;
2. Comprehensively evaluate the results under each criterion, then select the overall best answer and provide reasoning;
</Evaluation Steps>

<Evaluation Criteria>
Criterion 1. Whether the Overall Logic is Clear: Assess whether the overall logic of the answer is clear, including but not limited to:
a. Whether it adopts a "summary-analysis-conclusion" format, i.e., stating the conclusion at the beginning, sub-arguments in the middle, and a summary at the end;

Criterion 2. Whether the Overall Logic is Complete: Assess whether the overall logic is complete relative to the financial question, including but not limited to:
a. Whether the sub-arguments are sufficient to derive the conclusion;
b. Whether the overall logic comprehensively answers the financial question;

Criterion 3. Whether the Overall Logic is Consistent: Assess whether the overall logic is self-consistent, including but not limited to:
a. Each sub-argument should have a logical relationship with the conclusion, not isolated listing, ensuring logical consistency;
b. Sub-arguments have no overlap in content, arguing the conclusion from different angles;
c. Sub-arguments have no contradictions or conflicts in content
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
DEFAULT_OVERALL_LOGIC_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(OVERALL_LOGIC_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(OVERALL_LOGIC_PROMPT_ZH),
            ),
        ],
    },
)


class OverallLogicGrader(LLMGrader):
    """
    Overall Logic Grader for Finance Domain

    Evaluates the overall logical structure and coherence of financial analysis responses
    by comparing two responses based on clarity (structure), completeness (coverage),
    and consistency (coherence).

    Evaluation Criteria:
    1. Clarity: Clear structure with summary-analysis-conclusion format
    2. Completeness: Sufficient sub-arguments, comprehensive answer
    3. Consistency: Logical relationships, no overlap, no contradictions

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
        >>> grader = OverallLogicGrader(
        ...     model=model,
        ...     language=LanguageEnum.ZH
        ... )
        >>>
        >>> result = await grader.aevaluate(
        ...     query="分析宁德时代的投资价值",
        ...     answer_1="宁德时代值得投资，因为...",
        ...     answer_2="结论：建议买入。理由：1)基本面...2)估值..."
        ... )
        >>> print(f"Rank: {result.rank}")  # [2, 1] if answer_2 is better
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_OVERALL_LOGIC_TEMPLATE,
        language: LanguageEnum = LanguageEnum.ZH,
    ):
        """
        Initialize OverallLogicGrader.

        Args:
            model: The chat model to use for evaluation
            template: The prompt template for overall logic evaluation
            language: The language for the evaluation prompt (default: Chinese)
        """
        super().__init__(
            name="overall_logic",
            mode=GraderMode.LISTWISE,
            description="Evaluate overall logic and structure quality by comparing two responses",
            model=model,
            template=template,
            language=language,
        )
        self.template = template if template is not None else DEFAULT_OVERALL_LOGIC_TEMPLATE

    async def aevaluate(
        self,
        query: str,
        answer_1: str,
        answer_2: str,
        **kwargs: Any,
    ) -> GraderRank:
        """
        Evaluate two responses for overall logic quality

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
            logger.error(f"Error evaluating overall logic: {e}")
            rank = [1, 2]
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "evaluation_type": "overall_logic",
            "criteria": ["clarity", "completeness", "consistency"],
        }

        return GraderRank(
            name=self.name,
            rank=rank,
            reason=reason,
            metadata=metadata,
        )
