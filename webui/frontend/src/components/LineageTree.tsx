import { useMemo } from 'react'
import ReactFlow, { Background, Controls, type Edge, type Node } from 'reactflow'
import type { Tree } from '../types'
import { fmtScore, scoreColor, shortId } from '../ui'

/** Lay nodes out in rows by evolution iteration; colour by score. */
function layout(tree: Tree): { nodes: Node[]; edges: Edge[] } {
  const byIter = new Map<number, string[]>()
  const iterOf = (id: string) =>
    tree.nodes.find((n) => n.id === id)?.iteration ?? 0
  for (const n of tree.nodes) {
    const it = n.iteration ?? 0
    if (!byIter.has(it)) byIter.set(it, [])
    byIter.get(it)!.push(n.id)
  }
  const GAP_X = 190
  const GAP_Y = 130
  const pos = new Map<string, { x: number; y: number }>()
  for (const [it, ids] of byIter) {
    ids.forEach((id, i) => {
      pos.set(id, { x: i * GAP_X - ((ids.length - 1) * GAP_X) / 2, y: it * GAP_Y })
    })
  }

  const nodes: Node[] = tree.nodes.map((n) => ({
    id: n.id,
    position: pos.get(n.id) || { x: 0, y: 0 },
    data: {
      label: (
        <div style={{ textAlign: 'center', lineHeight: 1.3 }}>
          <div style={{ fontSize: 10, fontFamily: 'var(--mono)', opacity: 0.85 }}>
            {shortId(n.id)}
          </div>
          <div style={{ fontSize: 13, fontWeight: 700 }}>
            {n.score == null ? '—' : fmtScore(n.score)}
          </div>
          <div style={{ fontSize: 9, textTransform: 'uppercase', letterSpacing: 0.4, opacity: 0.7 }}>
            {n.evolution_kind || 'seed'}
          </div>
        </div>
      ),
    },
    style: {
      background: '#0c111b',
      color: '#e6ebf2',
      border: `2px solid ${n.is_focus ? '#f2c14e' : scoreColor(n.score)}`,
      borderRadius: 10,
      width: 120,
      padding: 6,
      boxShadow: n.is_focus ? '0 0 0 3px #f2c14e33' : 'none',
    },
  }))

  const edges: Edge[] = tree.edges.map((e, i) => ({
    id: `e${i}`,
    source: e.source,
    target: e.target,
    animated: iterOf(e.target) > iterOf(e.source),
    style: {
      stroke: e.kind === 'crossover' ? '#c78bf0' : '#7a879e',
      strokeDasharray: e.kind === 'crossover' ? '5 4' : undefined,
    },
  }))
  return { nodes, edges }
}

export default function LineageTree({ tree, onSelect }: { tree: Tree; onSelect?: (id: string) => void }) {
  const { nodes, edges } = useMemo(() => layout(tree), [tree])

  if (tree.nodes.length <= 1) {
    return (
      <div className="hint" style={{ padding: 20 }}>
        This run is a single <b>seed</b> with no offspring yet — the lineage tree fills in
        as learning-mode evolution produces mutations and crossovers.
      </div>
    )
  }

  return (
    <div className="flow-wrap">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        proOptions={{ hideAttribution: true }}
        onNodeClick={(_, n) => onSelect?.(n.id)}
        nodesDraggable={false}
        nodesConnectable={false}
      >
        <Background color="#1e2637" gap={20} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  )
}
