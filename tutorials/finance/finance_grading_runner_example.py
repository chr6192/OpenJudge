# -*- coding: utf-8 -*-
"""
Finance Grading Runner Example

This example demonstrates how to use GradingRunner with multiple finance graders
to evaluate financial analysis responses across different dimensions.

The example shows:
1. Setting up multiple finance graders (event analysis, fundamental analysis, risk analysis)
2. Preparing evaluation dataset with query and multiple answers
3. Running batch evaluation with GradingRunner
4. Analyzing results with various metrics
"""

import asyncio
import os

from loguru import logger

from open_judge.models.openai_chat_model import OpenAIChatModel
from open_judge.models.schema.prompt_template import LanguageEnum
from open_judge.runner.grading_runner import GradingRunner

# Import finance graders
from tutorials.finance.event_interpretation import EventAnalysisGrader
from tutorials.finance.stock_analysis import (
    FundamentalAnalysisGrader,
    StockRiskAnalysisGrader,
    ValuationAnalysisGrader,
)
from tutorials.finance.stock_search import SearchIntegrityGrader, SearchRelevanceGrader

# Sample dataset for evaluation
SAMPLE_DATASET = [
    {
        "query": "分析宁德时代2024年Q3财报，评估其投资价值",
        "answer_1": """
宁德时代是动力电池行业的龙头企业，市场份额领先。公司财务状况良好，
建议关注。
        """.strip(),
        "answer_2": """
【投资建议：谨慎增持】

一、基本面分析
1. 业务分析
   - 主营业务：动力电池系统、储能系统，2024Q3动力电池收入占比78%
   - 收入构成：国内60%、海外40%，海外收入增速45% YoY
   - 商业模式：规模效应显著，毛利率22%，高于行业平均18%

2. 行业竞争格局
   - 全球市占率：33%，排名第一（比亚迪25%，LG 15%）
   - 竞争优势：技术领先（麒麟电池能量密度255Wh/kg）、客户粘性强
   - 行业趋势：新能源车渗透率提升，2024年全球预计1400万辆（+25%）

3. 财务健康
   - 盈利能力：ROE 18%（同比-3pct），净利率8.5%
   - 资产负债：负债率58%，流动比率1.8，财务结构稳健
   - 现金流：经营现金流净额320亿元，资本开支280亿元

二、风险分析
1. 政策风险：补贴退坡，但影响可控（占收入<5%）
2. 行业风险：原材料价格波动（碳酸锂价格下跌30%→毛利率+2pct）
3. 竞争风险：比亚迪自供比例提升，宁德时代份额承压
4. 公司风险：产能扩张过快，利用率从85%降至78%

三、估值分析
【结论：估值合理偏高】
- 当前估值：PE 35倍，PB 4.5倍
- 历史对比：PE历史中位数30倍，当前处于60%分位
- 同业对比：比亚迪PE 28倍，LG PE 25倍，宁德时代溢价40%
- 合理性：考虑技术优势和全球化进展，溢价合理，但空间有限

四、整体逻辑
开头给出结论→基本面（业务+行业+财务）→风险分析→估值分析→总结
逻辑完备，各子论点支撑结论，自洽无矛盾。
        """.strip(),
        "expected_better": 2,  # answer_2 is expected to be better
        "category": "comprehensive_analysis",
    },
    {
        "query": "2024年美联储加息对中国股市有何影响？",
        "answer_1": """
美联储加息会导致资本外流，对中国股市有负面影响，建议谨慎投资。
        """.strip(),
        "answer_2": """
【事件识别】2024年美联储加息事件

【事件分析】
一、事件背景
2024年美联储维持高利率政策（5.25%-5.5%），主要应对通胀（CPI 3.2%）。

二、传导机制
美联储加息 → 中美利差扩大（当前-150bp）→ 资本外流压力 → 人民币贬值
→ 外资减持A股 → 估值承压

三、影响评估
1. 市场影响：
   - 北向资金流出：2024年累计流出580亿元
   - 估值压制：沪深300 PE从12倍降至11倍
   - 行业分化：出口链（电子、机械）受益人民币贬值；地产、金融承压

2. 政策应对：
   - 央行降准0.5%，释放流动性1.2万亿
   - 汇率政策：保持弹性，避免大幅贬值

四、投资建议
1. 防御策略：关注高股息（银行、公用事业）
2. 进攻策略：布局出口链（家电、新能源车）
3. 风险提示：若美联储继续加息，资本外流压力加大

【时间节点】2024年9月美联储议息会议维持利率不变
        """.strip(),
        "expected_better": 2,
        "category": "event_analysis",
    },
    {
        "query": "检索在越南有产能布局的中国上市纺织出口企业",
        "answer_1": """
申洲国际是主要的纺织出口企业，在越南有工厂。
        """.strip(),
        "answer_2": """
【检索结果】在越南有产能布局的中国上市纺织出口企业

一、符合条件的企业列表
1. 申洲国际 (02313.HK)
   - 产能布局：越南工厂产能占比35%（2023年数据）
   - 业务特征：运动服装代工，客户包括Nike、Adidas
   - 出口占比：95%以上

2. 华利集团 (300979.SZ)
   - 产能布局：越南工厂产能占比60%
   - 业务特征：运动鞋代工，主要客户Nike、Puma
   - 出口占比：98%

3. 天虹纺织 (02678.HK)
   - 产能布局：越南工厂产能占比25%
   - 业务特征：面料生产及染整
   - 出口占比：80%

4. 鲁泰纺织 (000726.SZ)
   - 产能布局：越南工厂建设中，预计2025年投产
   - 业务特征：高档衬衫面料
   - 出口占比：70%

二、检索说明
- 数据来源：公司年报、公告（截至2024年3月）
- 覆盖范围：A股、港股主要纺织出口企业
- 缺失信息：部分企业未披露越南产能详细数据

三、投资逻辑
越南产能布局主要应对：
1. 关税优惠：RCEP、EVFTA协议
2. 成本优势：人工成本较中国低30-40%
3. 订单转移：欧美客户要求供应链多元化

【时效说明】数据截至2024年Q1，建议关注最新产能投产进展
        """.strip(),
        "expected_better": 2,
        "category": "stock_search",
    },
]


async def run_finance_grading_example():
    """
    Run finance grading example with multiple graders.

    This example demonstrates:
    1. Multi-dimensional evaluation (fundamental, risk, valuation, search)
    2. Batch processing with GradingRunner
    3. Result analysis and comparison
    """
    logger.info("Starting Finance Grading Runner Example")

    # Initialize the chat model
    model = OpenAIChatModel(
        api_key=os.getenv("OPENAI_API_KEY"),  # Replace with actual key
        base_url=os.getenv("OPENAI_BASE_URL"),
        model="qwen3-max-preview",
    )

    # Initialize graders
    logger.info("Initializing finance graders...")

    event_analysis_grader = EventAnalysisGrader(
        model=model,
        language=LanguageEnum.ZH,
    )

    fundamental_analysis_grader = FundamentalAnalysisGrader(
        model=model,
        language=LanguageEnum.ZH,
    )

    risk_analysis_grader = StockRiskAnalysisGrader(
        model=model,
        language=LanguageEnum.ZH,
    )

    valuation_analysis_grader = ValuationAnalysisGrader(
        model=model,
        language=LanguageEnum.ZH,
    )

    search_integrity_grader = SearchIntegrityGrader(
        model=model,
        language=LanguageEnum.ZH,
    )

    search_relevance_grader = SearchRelevanceGrader(
        model=model,
        language=LanguageEnum.ZH,
    )

    # Configure graders for the runner
    grader_configs = {
        "event_analysis": event_analysis_grader,
        "fundamental_analysis": fundamental_analysis_grader,
        "risk_analysis": risk_analysis_grader,
        "valuation_analysis": valuation_analysis_grader,
        "search_integrity": search_integrity_grader,
        "search_relevance": search_relevance_grader,
    }

    # Initialize GradingRunner
    logger.info(f"Setting up GradingRunner with {len(grader_configs)} graders")
    runner = GradingRunner(grader_configs=grader_configs)

    # Run evaluation
    logger.info(f"Running evaluation on {len(SAMPLE_DATASET)} samples...")
    results = await runner.arun(SAMPLE_DATASET)

    # Analyze results
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)

    # results structure: {"grader_name": [result_for_sample1, result_for_sample2, ...]}
    # Reorganize to iterate by sample
    num_samples = len(SAMPLE_DATASET)
    for idx in range(num_samples):
        record = SAMPLE_DATASET[idx]
        logger.info(f"\n{'─'*80}")
        logger.info(f"Sample {idx+1}: {record['category']}")
        logger.info(f"{'─'*80}")
        logger.info(f"Query: {record['query'][:80]}...")
        logger.info(f"Expected Better Answer: {record['expected_better']}")
        logger.info("")

        # Analyze each grader's result for this sample
        for grader_name, grader_results in results.items():
            grader_result = grader_results[idx]  # Get result for this sample
            rank = grader_result.rank
            reason = grader_result.reason

            # Determine which answer was selected as better
            better_answer = 1 if rank[0] == 1 else 2
            matches_expected = "✓" if better_answer == record["expected_better"] else "✗"

            logger.info(f"  [{grader_name}] {matches_expected}")
            logger.info(f"    Rank: {rank} (Answer {better_answer} is better)")
            logger.info(f"    Reason: {reason[:150]}...")
            logger.info("")

    # Calculate agreement statistics
    logger.info("\n" + "=" * 80)
    logger.info("AGREEMENT STATISTICS")
    logger.info("=" * 80)

    grader_names = list(results.keys())
    total_samples = len(SAMPLE_DATASET)

    for grader_name in grader_names:
        correct = 0
        grader_results = results[grader_name]
        for idx, record in enumerate(SAMPLE_DATASET):
            rank = grader_results[idx].rank
            better_answer = 1 if rank[0] == 1 else 2
            if better_answer == record["expected_better"]:
                correct += 1

        accuracy = correct / total_samples * 100
        logger.info(f"  {grader_name}: {correct}/{total_samples} correct ({accuracy:.1f}%)")

    return results


def main():
    """Main entry point."""
    # Run comprehensive evaluation
    asyncio.run(run_finance_grading_example())


if __name__ == "__main__":
    main()
