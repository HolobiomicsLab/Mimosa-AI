# Hierarchical Optimization Framework for Self-Evolving Multi-Agent Systems

A theoretical formulation for understanding self-evolving multi-agent workflows as nested optimization over solution space partitions.

---

## 1. Problem Setup

### 1.1 Core Variables

| Symbol | Definition |
|--------|------------|
| $T$ | Task specification (the goal we want to achieve) |
| $\mathcal{S}$ | Full solution space (all possible outputs/trajectories) |
| $s^* \in \mathcal{S}$ | Optimal solution for task $T$ |

### 1.2 Agent-Level Variables

| Symbol | Definition |
|--------|------------|
| $a_i$ | Individual agent with role specification and capabilities |
| $R_i \subseteq \mathcal{S}$ | Subregion of solution space that agent $a_i$ operates within |
| $s_i \in R_i$ | Local solution produced by agent $a_i$ |
| $T_i$ | Subtask assigned to agent $a_i$ (derived from $T$ and upstream outputs) |
| $\mathbf{c}_i$ | Local context available to agent $a_i$ (bounded, agent-specific) |

### 1.3 Workflow-Level Variables

| Symbol | Definition |
|--------|------------|
| $\mathcal{G} = (V, E)$ | Workflow topology as a directed graph |
| $V = \{a_1, \ldots, a_k\}$ | Set of agents (nodes) |
| $E \subseteq V \times V$ | Information flow edges between agents |
| $\mathcal{W}$ | Space of all possible workflow topologies |
| $\phi: R_1 \times \cdots \times R_k \to \mathcal{S}$ | Composition function combining local solutions |

---

## 2. Level 1: Token-Level Optimization (Single Agent)

### 2.1 Generation Process

A single agent generates output through iterative optimization:

$$\hat{s}_{t+1} = \arg\max_{s} \, P(s \mid \mathbf{c}_t, \theta)$$

**Where:**
- $\mathbf{c}_t$ — accumulated context at step $t$ (prior tokens, observations, tool outputs)
- $\theta$ — model parameters (fixed during execution)
- $P(s \mid \mathbf{c}_t, \theta)$ — probability assigned to solution $s$ given context

### 2.2 Trajectory Drift

As generation proceeds, the agent's effective objective shifts away from the original task:

$$\underbrace{\arg\max_s \, P(s \mid T, \theta)}_{\text{Intended: solve task } T} \quad \xrightarrow{t \to \infty} \quad \underbrace{\arg\max_s \, P(s \mid \mathbf{c}_t, \theta)}_{\text{Actual: maintain consistency with } \mathbf{c}_t}$$

**Drift Mechanism:**

The context $\mathbf{c}_t$ becomes a corrupted proxy for $T$ because:

1. Suboptimal intermediate steps inject noise into $\mathbf{c}_t$
2. Context window constraints force compression/truncation
3. The model optimizes for *local coherence* rather than *global task completion*

**Formal Drift Measure (KL Divergence):**

$$D_{\text{drift}}(t) = D_{KL}\Big( P(s \mid \mathbf{c}_t, \theta) \,\|\, P(s \mid T, \theta) \Big)$$

This divergence increases monotonically with $t$ in the absence of external correction.

---

## 3. Level 2: Workflow-Level Optimization (Multi-Agent)

### 3.1 Problem Partitioning

A workflow $\mathcal{G}$ partitions the global problem into constrained subproblems. Each agent $a_i$ solves:

$$s_i^* = \arg\max_{s \in R_i} \, Q_i(s \mid T_i, \mathbf{c}_i)$$

**Where:**
- $T_i$ — subtask derived from $T$ and outputs of upstream agents
- $R_i \subseteq \mathcal{S}$ — constrained solution subspace (defines what $a_i$ can output)
- $\mathbf{c}_i$ — local context (bounded by agent interface)
- $Q_i$ — agent-specific objective function within its role

### 3.2 Why Partitioning Mitigates Drift

| Property | Effect on Drift |
|----------|-----------------|
| Smaller $R_i$ | Fewer steps to local convergence → less drift accumulation |
| Bounded $\mathbf{c}_i$ | Prevents unbounded context corruption |
| Cross-agent verification | Catches errors before propagation |
| Reset at boundaries | Each agent starts with clean context from upstream |

### 3.3 Solution Composition

The global solution emerges through composition of local solutions:

$$\hat{s} = \phi(s_1^*, s_2^*, \ldots, s_k^*; \mathcal{G})$$

**Where:**
- $\phi$ — composition function defined by workflow topology
- $\mathcal{G}$ — determines the order and connections between agents

### 3.4 The Topology Problem

The quality of $\hat{s}$ depends critically on how $\mathcal{G}$ partitions the problem:

| Failure Mode | Cause |
|--------------|-------|
| Incomplete coverage | $\bigcup_i R_i \not\supseteq$ relevant regions of $\mathcal{S}$ |
| Information bottleneck | Edges $E$ lose critical information between agents |
| Unnecessary serialization | Sequential dependencies where parallelism is possible |
| Role misalignment | Agent capabilities don't match subtask requirements |

---

## 4. Level 3: Meta-Optimization (Self-Evolution)

### 4.1 Optimization Objective

The meta-orchestrator searches over workflow space to find the optimal topology:

$$\mathcal{G}^* = \arg\max_{\mathcal{G} \in \mathcal{W}} \, J(\mathcal{G})$$

**Where the evaluation function $J(\mathcal{G})$ is:**

$$J(\mathcal{G}) = \mathbb{E}_{\tau \sim \mathcal{G}} \left[ \underbrace{\text{Perf}(\tau, T)}_{\text{task performance}} - \lambda \underbrace{\text{Cost}(\tau)}_{\text{execution cost}} \right]$$

**Variables:**
- $\tau$ — execution trajectory (full sequence of agent actions and outputs)
- $\tau \sim \mathcal{G}$ — trajectory sampled by running workflow $\mathcal{G}$ on task $T$
- $\text{Perf}(\tau, T)$ — measure of how well $\tau$ achieves task $T$
- $\text{Cost}(\tau)$ — computational/time/token cost of execution
- $\lambda \geq 0$ — cost-performance trade-off parameter

### 4.2 Search Process

Since $J(\mathcal{G})$ is non-differentiable (workflow changes are discrete structural modifications), meta-optimization proceeds empirically:

```
Algorithm: Self-Evolution Loop

Input: Initial topology G_0, task T, budget B
Output: Optimized topology G*

for n = 0 to B do:
    1. EXECUTE: Run workflow G_n on task T → trajectory τ_n
    2. EVALUATE: Compute J(G_n) from execution feedback
    3. DIAGNOSE: Identify bottlenecks, failures, inefficiencies in τ_n
    4. PROPOSE: Generate candidate modifications {G_n^(1), ..., G_n^(m)}
    5. SELECT: G_{n+1} ← best candidate (by predicted improvement)
end

return G* ← argmax_{G_0, ..., G_B} J(G_n)
```

### 4.3 Topology Modification Operators

The search explores $\mathcal{W}$ through discrete modifications:

| Operator | Description |
|----------|-------------|
| **Add Agent** | Introduce new agent $a_{k+1}$ with role specification |
| **Remove Agent** | Eliminate redundant agent, redistribute responsibilities |
| **Merge Agents** | Combine $a_i, a_j$ into single agent with joint role |
| **Split Agent** | Decompose $a_i$ into specialized sub-agents |
| **Rewire Edge** | Modify information flow: add/remove $(a_i, a_j) \in E$ |
| **Reorder** | Change topological ordering of execution |

---

## 5. Summary: Three Nested Optimization Loops

| Level | What is Optimized | Search Space | Timescale | Method |
|-------|-------------------|--------------|-----------|--------|
| **Token** | Next output $s_{t+1}$ | Local context window | Milliseconds | Autoregressive sampling |
| **Workflow** | Subtask solutions $\{s_i^*\}$ | Constrained subregions $\{R_i\}$ | Minutes | Agent execution |
| **Meta** | Topology $\mathcal{G}^*$ | Workflow space $\mathcal{W}$ | Hours/Days | Evolutionary search |

### Key Insight

Self-evolving multi-agent systems perform **architecture search guided by empirical feedback**:

- **Static single-agent**: Optimizes over $\mathcal{S}$ with no constraints → high drift
- **Static multi-agent**: Optimizes over fixed partition $\{R_i\}$ → suboptimal constraints
- **Self-evolving multi-agent**: Optimizes both the solution *and* the constraint structure → adaptive to task