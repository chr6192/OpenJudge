# -*- coding: utf-8 -*-
"""
Finance Domain Graders

This package provides specialized graders for evaluating financial analysis
and research content. All graders follow the new OpenJudgeframework architecture.

Modules:
    - event_interpretation: Event identification and analysis graders
    - industry_research: Industry characteristics, risk, and comparison graders
    - macro_analysis: Macroeconomic analysis and concept explanation graders
    - stock_analysis: Stock fundamental, valuation, risk, and logic graders
    - stock_search: Stock search integrity, relevance, and timeliness graders
"""

from tutorials.finance.event_interpretation import (
    EventAnalysisGrader,
    EventIdentificationGrader,
)
from tutorials.finance.industry_research import (
    CharacteristicsAnalysisGrader,
    RiskAnalysisGrader,
    UnderlyingComparisonGrader,
)
from tutorials.finance.macro_analysis import (
    ConceptExplanationGrader,
    MacroAnalysisGrader,
)
from tutorials.finance.stock_analysis import (
    FundamentalAnalysisGrader,
    OverallLogicGrader,
    StockRiskAnalysisGrader,
    ValuationAnalysisGrader,
)
from tutorials.finance.stock_search import (
    SearchIntegrityGrader,
    SearchRelevanceGrader,
    SearchTimelinessGrader,
)

__all__ = [
    # Event Interpretation
    "EventAnalysisGrader",
    "EventIdentificationGrader",
    # Industry Research
    "CharacteristicsAnalysisGrader",
    "RiskAnalysisGrader",
    "UnderlyingComparisonGrader",
    # Macro Analysis
    "ConceptExplanationGrader",
    "MacroAnalysisGrader",
    # Stock Analysis
    "FundamentalAnalysisGrader",
    "OverallLogicGrader",
    "StockRiskAnalysisGrader",
    "ValuationAnalysisGrader",
    # Stock Search
    "SearchIntegrityGrader",
    "SearchRelevanceGrader",
    "SearchTimelinessGrader",
]
