# -*- coding: utf-8 -*-
"""
Fundamental Analysis Grader for Finance Domain

Evaluates the quality of fundamental analysis by comparing two responses based on
completeness, depth, and logical reasoning.
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
FUNDAMENTAL_ANALYSIS_PROMPT_ZH = """
你是一个专业的金融评估专家，你需要评估金融助手回答的基本面情况(包括业务分析、所在行业分析、财务分析等)。现在会提供给你一个金融问题以及两位金融助手的回答，同时提供若干评估标准，你需要逐条理解评估标准，根据以下评估步骤，选择更好的答案。

<评估步骤>
1. 针对每个评估标准选择更好的回答，并给出你的理由;
2. 综合评估每个评估标准下的结果，然后选出综合最优的答案，并给出理由；
</评估步骤>

<评估标准>
评价标准1. 基本面分析的完整性：评估答案中的基本面分析是否全面，包括但不限于:
a. 公司的业务分析，包括但不限于：主营业务分析(公司核心业务、产品、服务)、收入构成分析(收入构成以及各板块收入的稳定性/增长性)、商业模式解读(盈利模式/成本结构/议价能力);
b. 行业赛道与竞争格局分析，包括但不限于：行业周期与趋势(揭示行业所处周期阶段、供给需求格局及政策环境影响机制)、行业竞争格局(识别主要竞争对手并做对比分析)、个股竞争力分析(考虑行业竞争格局中个股的竞争优势);
c. 财务健康情况，包括但不限于：盈利能力分析(需要分析关键指标(毛利率、净利率、ROE、ROIC等)的变化趋势)、资产负债结构：需要分析负债率、短期偿债能力（流动比率、速动比率）、资本结构稳定性;

评价标准2. 基本面分析的深度：评估答案中是否包含公司的特质,包括但不限于:
a. 公司关键业务驱动因素：公司经营表现背后的核心驱动因素(技术优势、渠道、品牌等);
b. 公司在本行业的竞争优势，包括但不限于：成本优势、技术优势;
c. 公司的成长性分析;
d. 其他能够体现公司特质的其他内容;

评价标准3. 基本面分析逻辑性：评估答案的
a. 论据支撑：每个结论需通过论据(数字/描述等)明确推导，解释变量间作用机制
b. 各子论点应和金融问题有逻辑关系，非孤立罗列，确保逻辑自洽
c. 各子论点应和结论有逻辑关系，非孤立罗列，确保逻辑自洽
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
FUNDAMENTAL_ANALYSIS_PROMPT_EN = """
You are a professional financial evaluation expert. You need to assess the fundamental analysis (including business analysis, industry analysis, financial analysis, etc.) in financial assistants' responses. You will be provided with a financial question and responses from two financial assistants, along with several evaluation criteria. You need to understand each criterion and select the better answer according to the following evaluation steps.

<Evaluation Steps>
1. For each evaluation criterion, select the better response and provide your reasoning;
2. Comprehensively evaluate the results under each criterion, then select the overall best answer and provide reasoning;
</Evaluation Steps>

<Evaluation Criteria>
Criterion 1. Completeness of Fundamental Analysis: Assess whether the fundamental analysis in the answer is comprehensive, including but not limited to:
a. Company business analysis, including but not limited to: core business analysis (company's core business, products, services), revenue composition analysis (revenue composition and stability/growth of each segment), business model interpretation (profit model/cost structure/pricing power);
b. Industry landscape and competitive analysis, including but not limited to: industry cycle and trends (reveal industry cycle stage, supply-demand landscape, and policy environment impact mechanisms), industry competitive landscape (identify major competitors and comparative analysis), individual stock competitiveness analysis (consider competitive advantages within the industry landscape);
c. Financial health status, including but not limited to: profitability analysis (analyze trends of key indicators such as gross margin, net margin, ROE, ROIC, etc.), asset-liability structure: analyze debt ratio, short-term solvency (current ratio, quick ratio), capital structure stability;

Criterion 2. Depth of Fundamental Analysis: Assess whether the answer includes company-specific characteristics, including but not limited to:
a. Key business drivers: core driving factors behind the company's operational performance (technological advantages, channels, brand, etc.);
b. Company's competitive advantages in the industry, including but not limited to: cost advantages, technological advantages;
c. Company's growth analysis;
d. Other content that can reflect company characteristics;

Criterion 3. Logical Consistency of Fundamental Analysis: Assess the answer's
a. Evidence support: Each conclusion needs to be clearly derived through evidence (numbers/descriptions, etc.), explaining the mechanism of action between variables
b. Each sub-argument should have a logical relationship with the financial question, not isolated listing, ensuring logical consistency
c. Each sub-argument should have a logical relationship with the conclusion, not isolated listing, ensuring logical consistency
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
DEFAULT_FUNDAMENTAL_ANALYSIS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(FUNDAMENTAL_ANALYSIS_PROMPT_EN),
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="user",
                content=textwrap.dedent(FUNDAMENTAL_ANALYSIS_PROMPT_ZH),
            ),
        ],
    },
)


class FundamentalAnalysisGrader(LLMGrader):
    """
    Fundamental Analysis Grader for Finance Domain

    Evaluates the quality of fundamental analysis by comparing two responses based on
    completeness (business, industry, financial), depth (key drivers, competitive advantages,
    growth), and logical consistency.

    Evaluation Criteria:
    1. Completeness: Business analysis, industry/competition, financial health
    2. Depth: Key business drivers, competitive advantages, growth analysis, company specifics
    3. Logical Consistency: Evidence support, logical relationships

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
        >>> grader = FundamentalAnalysisGrader(
        ...     model=model,
        ...     language=LanguageEnum.ZH
        ... )
        >>>
        >>> result = await grader.aevaluate(
        ...     query="分析宁德时代的基本面",
        ...     answer_1="宁德时代是动力电池龙头。",
        ...     answer_2="宁德时代主营动力电池，ROE 25%，市占率33%..."
        ... )
        >>> print(f"Rank: {result.rank}")  # [2, 1] if answer_2 is better
    """

    def __init__(
        self,
        model: BaseChatModel | dict,
        template: Optional[PromptTemplate] = DEFAULT_FUNDAMENTAL_ANALYSIS_TEMPLATE,
        language: LanguageEnum = LanguageEnum.ZH,
    ):
        """
        Initialize FundamentalAnalysisGrader.

        Args:
            model: The chat model to use for evaluation
            template: The prompt template for fundamental analysis evaluation
            language: The language for the evaluation prompt (default: Chinese)
        """
        super().__init__(
            name="fundamental_analysis",
            mode=GraderMode.LISTWISE,
            description="Evaluate fundamental analysis quality by comparing two responses",
            model=model,
            template=template,
            language=language,
        )
        self.template = template if template is not None else DEFAULT_FUNDAMENTAL_ANALYSIS_TEMPLATE

    async def aevaluate(
        self,
        query: str,
        answer_1: str,
        answer_2: str,
        **kwargs: Any,
    ) -> GraderRank:
        """
        Evaluate two fundamental analysis responses

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
            logger.error(f"Error evaluating fundamental analysis: {e}")
            rank = [1, 2]
            reason = f"Evaluation error: {str(e)}"

        # Prepare metadata
        metadata = {
            "evaluation_type": "fundamental_analysis",
            "criteria": ["completeness", "depth", "logical_consistency"],
        }

        return GraderRank(
            name=self.name,
            rank=rank,
            reason=reason,
            metadata=metadata,
        )
