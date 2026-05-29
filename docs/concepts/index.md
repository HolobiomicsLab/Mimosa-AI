# Concepts

How Mimosa-AI works under the hood.

| Page | What it covers |
| ---- | -------------- |
| [Architecture](architecture.md) | The five layers and how they fit together. |
| [Evolution engine](evolution-engine.md) | Depth-first recursion, QD selection, stagnation-driven mutation scope. |
| [Evaluation pipeline](evaluation-pipeline.md) | The multi-source per-claim verifier — the **judge that drives evolution**. Not the ScienceAgentBench / PaperBench grader. |
| [Tool discovery & MCP](tools-and-mcp.md) | How Mimosa finds and wires MCP tools — Toolomics, ToolHive, or any reachable MCP endpoint. |
| [Toolomics](toolomics.md) | The companion platform for MCP packaging, workspace isolation, and multi-instance deployment. |
| [Scientific grounding](grounding.md) | The role of Perspicacité in workflow synthesis and soft-claim verdicts. |

If you're building on top of Mimosa, also read the [Developer guide](../DEVELOPER_GUIDE.md)
— it links code paths to the concepts described here.

!!! note "Two evaluations, one word"
    "Evaluation" in Mimosa means two different things:

    - The **judge / verifier** drives workflow evolution (see
      [Evaluation pipeline](evaluation-pipeline.md)).
    - **Benchmark graders** (ScienceAgentBench, PaperBench) score the
      *output* against ground-truth files provided by the benchmark
      authors. See [Evaluation overview](../evaluation/index.md).
