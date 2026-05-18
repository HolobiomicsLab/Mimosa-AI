"""AST-derived behavioural features for the QD descriptor.

The descriptor must be orthogonal to fitness so the archive can
separate "different ways of being good" from "different ways of being
mediocre". These features read the workflow genotype source only —
never reward, cost, or iteration count.
"""

import ast

_AGENT_CTORS: frozenset[str] = frozenset(
    {"SmolAgentFactory", "CodeAgent", "ToolCallingAgent", "MultiStepAgent"}
)
_EDGE_METHODS: frozenset[str] = frozenset({"add_edge", "add_conditional_edges"})

_SCALES: tuple[float, float, float, float] = (10.0, 10.0, 10.0, 5000.0)
DESCRIPTOR_DIM: int = 4


def extract_code_features(code: str | None) -> list[float]:
    """Return a normalised 4-vector parsed from the workflow source.

    Axes: ``[n_agents, n_edges, n_branches, prompt_chars]`` each
    divided by a fixed scale so contributions to k-NN Euclidean
    distance are comparable. Missing or unparseable code yields a
    zero vector (treated by the gate as low-novelty).
    """
    if not code:
        return [0.0] * DESCRIPTOR_DIM
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [0.0] * DESCRIPTOR_DIM

    n_agents = 0
    n_edges = 0
    n_branches = 0
    prompt_chars = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in _AGENT_CTORS:
                n_agents += 1
            elif isinstance(fn, ast.Attribute) and fn.attr in _EDGE_METHODS:
                n_edges += 1
        elif isinstance(node, (ast.If, ast.For, ast.While)):
            n_branches += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            prompt_chars += len(node.value)

    raw = (n_agents, n_edges, n_branches, prompt_chars)
    return [r / s for r, s in zip(raw, _SCALES)]
