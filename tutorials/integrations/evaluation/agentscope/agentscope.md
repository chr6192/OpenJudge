# AgentScope Integration

This document explains how to integrate OpenJudgewith AgentScope through the [OpenJudgeMetric](file:///Users/zhuohua/workspace/OOpenJudgeutorials/integrations/agentscope/agentscope.py#L23-L106) wrapper class.

## Overview

The integration enables OpenJudgegraders to be used as AgentScope metrics. This allows you to leverage OOpenJudge extensive collection of evaluation methods within the AgentScope framework.

## Core Component

### OpenJudgeMetric Class

The [OpenJudgeMetric](file:///Users/zhuohua/workspace/OpenJudgetutorials/integrations/agentscope/agentscope.py#L23-L106) is a wrapper that bridges OOpenJudge grading system with AgentScope's metric system.

```python
from open_judge.graders.base_grader import Grader
from tutorials.integrations.agentscope.agentscope import OpenJudgeMetric

# Create or obtain an OpenJudgegrader
grader = YourCustomGrader()

# Wrap it as an AgentScope metric
metric = OpenJudgeMetric(grader)
```

## Implementation Requirements

To use [OpenJudgeMetric](file:///Users/zhuohua/workspace/OpenJudgetutorials/integrations/agentscope/agentscope.py#L23-L106), you must implement two abstract methods in a subclass:

1. `_convert_solution_to_dict`: Converts AgentScope solutions to OpenJudgeinput format
2. `_convert_grader_result_to_metric_result`: Converts OpenJudgeoutputs to AgentScope format

Example implementation:
```python
class CustomOpenJudgeMetric(OpenJudgeMetric):
    async def _convert_solution_to_dict(self, solution):
        # Your conversion logic here
        return {"query": solution.query, "response": solution.response}

    async def _convert_grader_result_to_metric_result(self, grader_result):
        # Your conversion logic here
        return {"score": grader_result.score, "reason": grader_result.reason}
```

## Usage

Once implemented, you can use the metric in AgentScope workflows:

```python
# Evaluate a solution
result = await metric(solution)
```

This executes the OpenJudgegrader internally and returns the result in AgentScope's expected format.