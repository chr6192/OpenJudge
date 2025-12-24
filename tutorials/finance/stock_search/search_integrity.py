# -*- coding: utf-8 -*-
"""
Search Integrity Grader for Finance Domain

Evaluates the integrity and completeness of stock search results by comparing
two responses based on coverage and information completeness.
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
SEARCH_INTEGRITY_PROMPT_ZH = """
你是一个专业的金融评估专家，你需要评估金融助手回答股票检索问题的专业性。现在会提供给你一个金融问题以及两位金融助手的回答，同时提供若干评估标准，你需要逐条理解评估标准，根据以下评估步骤，选择更好的答案。

<评估步骤>
1. 针对每个评估标准选择更好的回答，并给出你的理由;
2. 综合评估每个评估标准下的结果，然后选出综合最优的答案，并给出理由；
</评估步骤>

<评估标准>
评价标准1. 检索完整性：评估答案中检索到的股票信息的完整性，包括但不限于：
a. 覆盖全面：在确保准确性的前提下，应尽可能覆盖符合条件的主要公司，而非只列举单个或少数示例。
b. 若信息不全，应明确说明缺失部分并提示可能的来源或查询建议。
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
SEARCH_INTEGRITY_PROMPT_EN = """
You are a professional financial evaluation expert. You need to assess the professionalism of financial assistants' responses to stock search questions. You will be provided with a financial question and responses from two financial assistants, along with several evaluation criteria. You need to understand each criterion and select the better answer according to the following evaluation steps.

<Evaluation Steps>
1. For each evaluation criterion, select the better response and provide your reasoning;
2. Comprehensively evaluate the results under each criterion, then select the overall best answer and provide reasoning;
</Evaluation Steps>

<Evaluation Criteria>
Criterion 1. Search Integrity: Assess the integrity of stock information retrieved in the answer, including but not limited to:
a. Comprehensive coverage: While ensuring accuracy, should cover as many qualifying major companies as possible, rather than listing only one or a few examples.
b. If information is incomplete, should clearly state missing parts and suggest possible sources or query recommendations.
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
DEFAULT_SEARCH_INTEGRITY_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(SEARCH_INTEGRITY_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(SEARCH_INTEGRITY_PROMPT_ZH),
            ),
        ],
    },
)


class SearchIntegrityGrader(LLMGrader):
    """
    Search Integrity Grader for Finance Domain

    Evaluates the integrity and completeness of stock search results by comparing
    two responses based on comprehensive coverage and information completeness.

    Evaluation Criteria:
    1. Comprehensive coverage: Cover major qualifying companies, not just examples
    2. Information completeness: State missing information and suggest sources

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
        >>> grader = SearchIntegrityGrader(
        ...     model=model,
        ...     language=LanguageEnum.ZH
        ... )
        >>>
        >>> result = await grader.aevaluate(
        ...     query="检索新能源汽车行业的上市公司",
        ...     answer_1="比亚迪是新能源汽车龙头。",
        ...     answer_2="新能源汽车上市公司包括：比亚迪、宁德时代、理想汽车..."
        ... )
        >>> print(f"Rank: {result.rank}")  # [2, 1] if answer_2 is better
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_SEARCH_INTEGRITY_TEMPLATE,
        language: LanguageEnum = LanguageEnum.ZH,
    ):
        """
        Initialize SearchIntegrityGrader.

        Args:
            model: The chat model to use for evaluation
            template: The prompt template for search integrity evaluation
            language: The language for the evaluation prompt (default: Chinese)
        """
        super().__init__(
            name="search_integrity",
            mode=GraderMode.LISTWISE,
            description="Evaluate stock search integrity and completeness by comparing two responses",
            model=model,
            template=template,
            language=language,
        )
        self.template = template if template is not None else DEFAULT_SEARCH_INTEGRITY_TEMPLATE

    async def aevaluate(
        self,
        query: str,
        answer_1: str,
        answer_2: str,
        **kwargs: Any,
    ) -> GraderRank:
        """
        Evaluate two stock search responses for integrity

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
            logger.error(f"Error evaluating search integrity: {e}")
            rank = [1, 2]
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "evaluation_type": "search_integrity",
            "criteria": ["comprehensive_coverage", "information_completeness"],
        }

        return GraderRank(
            name=self.name,
            rank=rank,
            reason=reason,
            metadata=metadata,
        )
