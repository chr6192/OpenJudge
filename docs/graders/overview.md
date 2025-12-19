# Built-in Graders Overview

RM-Gallery provides **50+ pre-built graders** for evaluating AI responses across quality dimensions, agent behaviors, formats, and modalities. Use these graders out-of-the-box or customize them for your specific evaluation needs.

---

## Why Use Built-in Graders?

Pre-built graders accelerate AI evaluation by providing:

- **Production-Ready Quality**: Battle-tested evaluation logic with benchmark validation
- **Comprehensive Coverage**: From basic relevance to complex agent reasoning
- **Flexible Integration**: Compatible with any LLM framework and evaluation pipeline
- **Consistent Criteria**: Standardized scoring across your applications
- **Dual Evaluation Modes**: Support both pointwise scoring and listwise ranking

---

## Grader Categories

RM-Gallery organizes graders by evaluation focus:

| Category | Implementation | Best For | Key Graders |
|----------|---------------|----------|-------------|
| [Common](general.md) | LLM-Based | Quality assessment | Relevance, Hallucination, Harmfulness, Correctness |
| [Text](text.md) | Code-Based | Text similarity & matching | Similarity, String Match, Number Accuracy |
| [Code & Math](code-math.md) | Code-Based | Technical accuracy | Code Execution, Syntax Check, Math Verify |
| [Format](format.md) | Code-Based | Structure validation | JSON Validator, Reasoning Format, Length Penalty |
| [Multimodal](multimodal.md) | LLM-Based | Vision & image tasks | Image Coherence, Text-to-Image, Image Editing |

---

## Grader Architecture

All graders share a common evaluation interface:

```
┌─────────────────────────────────────────────────┐
│  Input Data                                     │
│  ├─ Query (optional)                           │
│  ├─ Response (required)                        │
│  └─ Context/Ground Truth (optional)            │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Grader (BaseGrader)                           │
│  ├─ Mode: POINTWISE | LISTWISE                 │
│  └─ Implementation Type:                       │
│     ├─ LLMGrader: Uses LLM for evaluation      │
│     ├─ FunctionGrader: Custom function logic   │
│     └─ Code-Based: Algorithm/rule-based        │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  Evaluation Result                              │
│  ├─ GraderScore: score + reason + metadata     │
│  └─ GraderRank: rank + reason + metadata       │
└─────────────────────────────────────────────────┘
```

> **Note:** All graders inherit from `BaseGrader` and support asynchronous evaluation via `aevaluate()`.

---

## Grader Types

RM-Gallery provides three implementation approaches:

| Type | Implementation | Best For | Example Graders |
|------|---------------|----------|-----------------|
| **LLM-Based** | Uses LLM as judge (default: qwen3-32b) | Subjective quality, agent reasoning, multimodal | `RelevanceGrader`, `HallucinationGrader`, `ToolCallAccuracyGrader` |
| **Code-Based** | Pure algorithms (no LLM) | Fast, deterministic, zero-cost evaluation | `CodeExecutionGrader`, `SimilarityGrader`, `JsonValidatorGrader` |
| **Function-Based** | Custom user functions | Domain-specific logic | `FunctionGrader` with custom logic |

> **Tip:** Use **Code-Based** graders (Text, Code, Format, Math) for fast, cost-free evaluation. Use **LLM-Based** graders (Common, Agent, Multimodal) for nuanced quality assessment.

---

## Quick Start

### 1. Choose a Grader

Select based on your evaluation goal:

```python
from rm_gallery.core.graders.common import RelevanceGrader, HallucinationGrader
from rm_gallery.core.graders.text import SimilarityGrader
from rm_gallery.core.graders.code import CodeExecutionGrader
from rm_gallery.core.models import OpenAIChatModel

# LLM-based grader (requires model)
model = OpenAIChatModel(model="qwen3-32b")
grader = RelevanceGrader(model=model)

# Code-based grader (no model needed)
grader = SimilarityGrader(algorithm="bleu")
grader = CodeExecutionGrader(continuous=True)
```

### 2. Run Evaluation

Evaluate responses with appropriate inputs:

```python
# Async evaluation (recommended)
result = await grader.aevaluate(
    query="What is machine learning?",
    response="Machine learning is a subset of AI..."
)

print(result.score)   # Numerical score
print(result.reason)  # Explanation
print(result.metadata)  # Additional info
```

---

## Evaluation Modes

RM-Gallery supports two evaluation modes:

### Pointwise Mode (Default)

Evaluate individual samples independently:

```python
from rm_gallery.core.graders.common import RelevanceGrader
from rm_gallery.core.graders.schema import GraderMode
from rm_gallery.core.models import OpenAIChatModel

model = OpenAIChatModel(model="qwen3-32b")
grader = RelevanceGrader(model=model)  # mode=GraderMode.POINTWISE by default

result = await grader.aevaluate(
    query="Explain photosynthesis",
    response="Photosynthesis is the process..."
)
# Returns: GraderScore(score=4.5, reason="...")
```

### Listwise Mode

Rank multiple responses relative to each other:

```python
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.graders.schema import GraderMode
from rm_gallery.core.models import OpenAIChatModel

# Create a custom listwise grader
model = OpenAIChatModel(model="qwen3-32b")
grader = LLMGrader(
    model=model,
    name="ranker",
    mode=GraderMode.LISTWISE,
    template="Rank these responses: {query}\n{responses}"
)

result = await grader.aevaluate(
    query="Explain AI",
    responses=["Response A", "Response B", "Response C"]
)
# Returns: GraderRank(rank=[1, 3, 2], reason="...")
```

---

## Batch Evaluation

Process multiple samples efficiently with `GradingRunner`:

```python
from rm_gallery.core.runner import GradingRunner, GraderConfig
from rm_gallery.core.graders.common import RelevanceGrader
from rm_gallery.core.models import OpenAIChatModel

model = OpenAIChatModel(model="qwen3-32b")
runner = GradingRunner(
    grader_configs={"relevance": GraderConfig(grader=RelevanceGrader(model=model))}
)
results = await runner.arun(
    dataset=[
        {"query": "Q1", "response": "A1"},
        {"query": "Q2", "response": "A2"},
        {"query": "Q3", "response": "A3"}
    ]
)

# Access results by grader name
for i, result in enumerate(results["relevance"]):
    print(f"Sample {i}: {result.score}")
```

### Multi-Grader Evaluation

Combine multiple graders for comprehensive assessment:

```python
from rm_gallery.core.runner import GradingRunner, GraderConfig
from rm_gallery.core.graders.common import RelevanceGrader, HallucinationGrader
from rm_gallery.core.models import OpenAIChatModel

model = OpenAIChatModel(model="qwen3-32b")
runner = GradingRunner(
    grader_configs={
        "relevance": GraderConfig(grader=RelevanceGrader(model=model)),
        "hallucination": GraderConfig(grader=HallucinationGrader(model=model))
    }
)

results = await runner.arun([{"query": "...", "response": "..."}])
# Returns: {"relevance": [result1, ...], "hallucination": [result1, ...]}
```

---

## Performance Benchmarks

All graders are validated against human-annotated datasets:

| Metric | Description | Target |
|--------|-------------|--------|
| **Preference Accuracy** | Alignment with human preferences | > 80% |
| **Format Compliance** | Valid output structure | 100% |
| **Avg Score Diff** | Separation between good/bad responses | > 2.0 |

> **Tip:** See individual grader pages for detailed benchmark results by model.

---

## Result Schema

### GraderScore (Pointwise)

```python
class GraderScore:
    name: str       # Grader identifier
    score: float    # Numerical score
    reason: str     # Explanation
    metadata: dict  # Additional info
```

### GraderRank (Listwise)

```python
class GraderRank:
    name: str       # Grader identifier
    rank: List[int] # Ranking (e.g., [1, 3, 2])
    reason: str     # Explanation
    metadata: dict  # Additional info
```

---

## Next Steps

**Explore graders by category:**

- **[Common Graders](general.md)** — LLM-based quality assessment (Relevance, Hallucination, Harmfulness)
- **[Text Graders](text.md)** — Fast, algorithm-based text evaluation (Similarity, String Match, Number Accuracy)
- **[Code & Math Graders](code-math.md)** — Code execution and mathematical verification
- **[Format Graders](format.md)** — Output structure validation (JSON, Length, Repetition)
- **[Multimodal Graders](multimodal.md)** — Vision and image evaluation

**Learn more:**

- **[Create Custom Graders](../building-graders/custom-graders.md)** — Build domain-specific evaluators
- **[Train a Grader](../building-graders/training/overview.md)** — Train custom reward models
