# -*- coding: utf-8 -*-
"""Test Evol-Instruct query evolution feature."""

import asyncio

from loguru import logger

from cookbooks.zero_shot_evaluation.schema import load_config
from cookbooks.zero_shot_evaluation.query_generator import QueryGenerator


async def test_evol_instruct():
    """Test Evol-Instruct query evolution."""
    config = load_config("data/examples/config.yaml")
    
    logger.info("=" * 60)
    logger.info("Evol-Instruct Test")
    logger.info("=" * 60)
    logger.info(f"Target queries: {config.query_generation.num_queries}")
    logger.info(f"Evolution enabled: {config.query_generation.enable_evolution}")
    logger.info(f"Evolution rounds: {config.query_generation.evolution_rounds}")
    logger.info(f"Complexity levels: {config.query_generation.complexity_levels}")
    logger.info("=" * 60)
    
    generator = QueryGenerator(
        judge_endpoint=config.judge_endpoint,
        task_config=config.task,
        query_config=config.query_generation,
    )
    
    logger.info("Starting query generation with Evol-Instruct...")
    queries = await generator.generate()
    
    logger.info("=" * 60)
    logger.info(f"Generated {len(queries)} queries:")
    logger.info("=" * 60)
    
    # 分类统计
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    
    for i, q in enumerate(queries, 1):
        difficulty = q.difficulty or "unknown"
        if difficulty in difficulty_counts:
            difficulty_counts[difficulty] += 1
        
        print(f"\n[{i}] {q.query}")
        print(f"    Category: {q.category or 'N/A'}")
        print(f"    Difficulty: {difficulty}")
    
    logger.info("=" * 60)
    logger.info("Difficulty Distribution:")
    for level, count in difficulty_counts.items():
        logger.info(f"  {level}: {count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_evol_instruct())

