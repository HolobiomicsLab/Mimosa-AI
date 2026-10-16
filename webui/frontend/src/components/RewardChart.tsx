import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip,
  XAxis, YAxis, Legend,
} from 'recharts'
import type { Series } from '../types'

const AXIS = { stroke: '#6b7688', fontSize: 11 }

export default function RewardChart({ series }: { series: Series }) {
  const data = series.points.map((p) => ({
    iteration: p.iteration ?? 0,
    reward: p.overall_score,
    qd: p.qd_score,
    novelty: p.novelty_score,
    cost: p.cumulative_cost_usd,
  }))

  if (data.length === 0) {
    return <div className="hint" style={{ padding: 20 }}>No per-iteration metrics recorded yet.</div>
  }
  const single = data.length === 1

  return (
    <div>
      {single && (
        <div className="hint" style={{ marginBottom: 10 }}>
          One data point (seed). The curve populates across learning-mode iterations.
        </div>
      )}
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 8, right: 16, left: -8, bottom: 4 }}>
          <CartesianGrid stroke="#1e2637" strokeDasharray="3 3" />
          <XAxis dataKey="iteration" {...AXIS} label={{ value: 'iteration', position: 'insideBottom', offset: -2, fill: '#6b7688', fontSize: 11 }} />
          <YAxis yAxisId="r" domain={[0, 1]} {...AXIS} />
          <YAxis yAxisId="c" orientation="right" {...AXIS} tickFormatter={(v) => `$${v}`} />
          <Tooltip
            contentStyle={{ background: '#121722', border: '1px solid #2f3b52', borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: '#97a1b3' }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Line yAxisId="r" type="monotone" dataKey="reward" stroke="#4ec98a" strokeWidth={2} dot={{ r: single ? 5 : 3 }} name="reward" />
          <Line yAxisId="r" type="monotone" dataKey="qd" stroke="#f2c14e" strokeWidth={1.5} dot={false} name="QD score" />
          <Line yAxisId="r" type="monotone" dataKey="novelty" stroke="#5db8f0" strokeWidth={1.5} strokeDasharray="4 3" dot={false} name="novelty" />
          <Line yAxisId="c" type="monotone" dataKey="cost" stroke="#ef6a6a" strokeWidth={1.5} dot={false} name="cost $" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
