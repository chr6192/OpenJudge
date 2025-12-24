# -*- coding: utf-8 -*-
"""
Search Relevance Grader for Finance Domain

Evaluates the relevance of stock search results by comparing two responses
based on alignment with query requirements and precision matching.
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
SEARCH_RELEVANCE_PROMPT_ZH = """
你是一个专业的金融评估专家，你需要评估金融助手回答股票检索问题的专业性。现在会提供给你一个金融问题以及两位金融助手的回答，同时提供若干评估标准，你需要逐条理解评估标准，根据以下评估步骤，选择更好的答案。

<评估步骤>
1. 针对每个评估标准选择更好的回答，并给出你的理由;
2. 综合评估每个评估标准下的结果，然后选出综合最优的答案，并给出理由；
</评估步骤>

<评估标准>
评价标准1. 检索相关性：评估答案中检索到的股票和问题的相关性，包括但不限于：
a. 回答应严格围绕问题核心要求（行业范围、地域条件、业务特征等），不引入无关行业或公司。
b. 精准匹配：检索结果需与问题限定的地域、行业、业务特性精准匹配，例如"越南有产能布局的国内上市出口公司"需同时满足"国内上市 + 出口 + 越南产能"条件。
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
SEARCH_RELEVANCE_PROMPT_EN = """
You are a professional financial evaluation expert. You need to assess the professionalism of financial assistants' responses to stock search questions. You will be provided with a financial question and responses from two financial assistants, along with several evaluation criteria. You need to understand each criterion and select the better answer according to the following evaluation steps.

<Evaluation Steps>
1. For each evaluation criterion, select the better response and provide your reasoning;
2. Comprehensively evaluate the results under each criterion, then select the overall best answer and provide reasoning;
</Evaluation Steps>

<Evaluation Criteria>
Criterion 1. Search Relevance: Assess the relevance of stocks retrieved in the answer to the question, including but not limited to:
a. Responses should strictly focus on the core requirements of the question (industry scope, geographic conditions, business characteristics, etc.), without introducing irrelevant industries or companies.
b. Precise matching: Search results must precisely match the geographic, industry, and business characteristics specified in the question. For example, "domestically listed export companies with production capacity in Vietnam" must simultaneously meet "domestic listing + export + Vietnam capacity" conditions.
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
DEFAULT_SEARCH_RELEVANCE_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(SEARCH_RELEVANCE_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(SEARCH_RELEVANCE_PROMPT_ZH),
            ),
        ],
    },
)


class SearchRelevanceGrader(LLMGrader):
    """
    Search Relevance Grader for Finance Domain

    Evaluates the relevance of stock search results by comparing two responses
    based on alignment with core requirements and precise matching of conditions.

    Evaluation Criteria:
    1. Focus on core requirements: Industry scope, geography, business characteristics
    2. Precise matching: All specified conditions must be met simultaneously

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
        >>> grader = SearchRelevanceGrader(
        ...     model=model,
        ...     language=LanguageEnum.ZH
        ... )
        >>>
        >>> result = await grader.aevaluate(
        ...     query="检索在越南有产能布局的国内上市出口公司",
        ...     answer_1="阿里巴巴是国内上市公司。",
        ...     answer_2="申洲国际：国内上市，纺织出口，越南有产能；立讯精密..."
        ... )
        >>> print(f"Rank: {result.rank}")  # [2, 1] if answer_2 is better
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_SEARCH_RELEVANCE_TEMPLATE,
        language: LanguageEnum = LanguageEnum.ZH,
    ):
        """
        Initialize SearchRelevanceGrader.

        Args:
            model: The chat model to use for evaluation
            template: The prompt template for search relevance evaluation
            language: The language for the evaluation prompt (default: Chinese)
        """
        super().__init__(
            name="search_relevance",
            mode=GraderMode.LISTWISE,
            description="Evaluate stock search relevance by comparing two responses",
            model=model,
            template=template,
            language=language,
        )
        self.template = template if template is not None else DEFAULT_SEARCH_RELEVANCE_TEMPLATE

    async def aevaluate(
        self,
        query: str,
        answer_1: str,
        answer_2: str,
        **kwargs: Any,
    ) -> GraderRank:
        """
        Evaluate two stock search responses for relevance

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
            logger.error(f"Error evaluating search relevance: {e}")
            rank = [1, 2]
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "evaluation_type": "search_relevance",
            "criteria": ["core_alignment", "precise_matching"],
        }

        return GraderRank(
            name=self.name,
            rank=rank,
            reason=reason,
            metadata=metadata,
        )
