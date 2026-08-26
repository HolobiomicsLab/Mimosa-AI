import type { AstraCapsule, AstraEnvironment } from '../types'

/**
 * ReproducibilityCard (D18) — a fixed-facet ledger graded AT RENDER TIME from
 * what the record actually carries. Never stored, never a scalar score.
 * Closed vocabulary; every cell is a fixed-grammar statement
 * (`statement_grammar: 1`): `<grade>` or `<grade> (<detail>)`. Self-declared
 * and independently-observed are separate columns — independent observation
 * is the evaluator's layer and is almost always absent today, which the
 * ledger says rather than hides. Categories share ONE neutral palette; only
 * "recorded" gets a (still neutral) distinct style.
 */

type Grade =
  | 'recorded'
  | 'partial'
  | 'absent'
  | 'not_applicable'
  | 'nondeterministic_by_design'
  | 'transcript_only'

interface Cell {
  grade: Grade
  /** Rendered as the parenthesised part of the statement. */
  detail?: string
}

interface Facet {
  name: string
  self: Cell
  observed: Cell
}

/** Today no facet has an independent observer wired in — the honest constant. */
const NOT_OBSERVED: Cell = { grade: 'absent', detail: 'no independent observation recorded' }

const NO_ENV_BLOCK = 'no environment block; capsule predates environment capture'

/** env_capture encodes absence as "absent (<reason>)" and partial capture as
 * "partial (<scope>)" strings — decode them into the closed vocabulary,
 * keeping the reason text verbatim. An unrecognized encoding grades
 * ``partial`` with the raw string quoted: "recorded" (the ledger's only
 * full-strength grade) is reserved for values the decoder positively
 * understands, so a drifted producer can only understate, never inflate. */
function fromEnvString(v: string): Cell {
  const absent = v.match(/^absent \((.+)\)$/)
  if (absent) return { grade: 'absent', detail: absent[1] }
  const partial = v.match(/^partial \((.+)\)$/)
  if (partial) return { grade: 'partial', detail: partial[1] }
  return { grade: 'partial', detail: `unrecognized encoding: "${v}"` }
}

function gitFacet(env: AstraEnvironment | null): Cell {
  if (!env) return { grade: 'absent', detail: NO_ENV_BLOCK }
  const git = env.git
  if (!git || typeof git !== 'object' || !git.commit) {
    return { grade: 'absent', detail: 'no git commit in the environment block' }
  }
  const dirty = git.dirty == null ? 'dirty unknown' : git.dirty ? 'dirty tree' : 'clean tree'
  return {
    grade: 'recorded',
    detail: `commit ${String(git.commit).slice(0, 12)} on ${git.branch ?? '?'}, ${dirty}; self-declared`,
  }
}

function pythonPlatformFacet(env: AstraEnvironment | null): Cell {
  if (!env) return { grade: 'absent', detail: NO_ENV_BLOCK }
  const py = env.python_version
  const plat = env.platform
  if (py && plat) return { grade: 'recorded', detail: `python ${py} on ${plat}` }
  if (py || plat) {
    return { grade: 'partial', detail: py ? `python ${py}; platform unrecorded` : `${plat}; python version unrecorded` }
  }
  return { grade: 'absent', detail: 'no python/platform in the environment block' }
}

function modelRolesFacet(env: AstraEnvironment | null): Cell {
  if (!env) return { grade: 'absent', detail: NO_ENV_BLOCK }
  const roles = env.model_roles
  if (roles == null) return { grade: 'absent', detail: 'no model roles in the environment block' }
  if (typeof roles === 'string') return fromEnvString(roles)
  const pairs = Object.entries(roles).map(([k, v]) => `${k}=${v}`)
  if (pairs.length === 0) return { grade: 'absent', detail: 'config declares no model roles' }
  return { grade: 'recorded', detail: pairs.join(', ') }
}

function temperatureFacet(env: AstraEnvironment | null): Cell {
  if (!env) return { grade: 'absent', detail: NO_ENV_BLOCK }
  const t = env.temperature
  if (t == null) return { grade: 'absent', detail: 'no temperature aggregate in the environment block' }
  if (typeof t === 'string') return fromEnvString(t)
  const range = `min ${t.min ?? '?'} max ${t.max ?? '?'} over ${t.n_calls ?? '?'} calls`
  if (typeof t.max === 'number' && t.max > 0) {
    return { grade: 'nondeterministic_by_design', detail: `sampling temperature ${range}` }
  }
  return { grade: 'recorded', detail: `temperature ${range}` }
}

function configDigestFacet(env: AstraEnvironment | null): Cell {
  if (!env) return { grade: 'absent', detail: NO_ENV_BLOCK }
  const digest = env.config_digest
  if (digest == null) return { grade: 'absent', detail: 'no config digest in the environment block' }
  // Positive recognition of env_capture's digest shape gates "recorded".
  if (/^sha256:[0-9a-f]{64}$/.test(digest)) {
    return { grade: 'recorded', detail: `secret-scrubbed ${digest.slice(0, 23)}…` }
  }
  return fromEnvString(digest)
}

function groundingFacet(env: AstraEnvironment | null): Cell {
  if (!env) return { grade: 'absent', detail: NO_ENV_BLOCK }
  const grounding = env.grounding
  if (grounding == null) return { grade: 'absent', detail: 'no grounding block in the environment block' }
  if (typeof grounding === 'string') return fromEnvString(grounding)
  return { grade: 'recorded', detail: 'self-declared by the run (run_metrics.json); not independently measured' }
}

function runnerEnvFacet(env: AstraEnvironment | null): Cell {
  if (!env) return { grade: 'absent', detail: NO_ENV_BLOCK }
  const runner = env.runner_env
  if (runner == null) return { grade: 'absent', detail: 'no runner-env statement in the environment block' }
  return fromEnvString(runner)
}

function outputsPinningFacet(capsule: AstraCapsule): Cell {
  const manifest = capsule.outputs_manifest
  if (manifest) {
    const n = Object.keys(manifest).length
    return { grade: 'recorded', detail: `${n} output${n === 1 ? '' : 's'} pinned with content sha256` }
  }
  return { grade: 'absent', detail: capsule.outputs_manifest_absent_reason ?? 'no outputs manifest' }
}

function recipeFacet(capsule: AstraCapsule): Cell {
  if (capsule.recipe_header != null) {
    // The file's own header line, quoted — the transcript claim is the
    // artifact's, not the UI's.
    return { grade: 'transcript_only', detail: `"${capsule.recipe_header}"` }
  }
  return { grade: 'absent', detail: capsule.recipe_header_absent_reason ?? 'no recipe.py in the capsule' }
}

function genotypePinFacet(env: AstraEnvironment | null): Cell {
  if (!env) return { grade: 'absent', detail: NO_ENV_BLOCK }
  const pin = env.workflow_genotype
  if (pin == null) return { grade: 'absent', detail: 'no workflow genotype pin recorded' }
  if (typeof pin === 'string') return fromEnvString(pin)
  if (pin.path && pin.sha256) {
    return { grade: 'recorded', detail: `${pin.path} · sha256:${pin.sha256.slice(0, 12)}…` }
  }
  return { grade: 'partial', detail: `pin lacks ${pin.path ? 'sha256' : 'path'}` }
}

function gradeFacets(capsule: AstraCapsule): Facet[] {
  const env = capsule.environment
  const facet = (name: string, self: Cell): Facet => ({ name, self, observed: NOT_OBSERVED })
  return [
    facet('git commit', gitFacet(env)),
    facet('python / platform', pythonPlatformFacet(env)),
    facet('model roles', modelRolesFacet(env)),
    facet('temperature range', temperatureFacet(env)),
    facet('config digest', configDigestFacet(env)),
    facet('grounding', groundingFacet(env)),
    facet('runner sandbox env', runnerEnvFacet(env)),
    facet('outputs pinning', outputsPinningFacet(capsule)),
    facet('recipe.py', recipeFacet(capsule)),
    facet('workflow genotype pin', genotypePinFacet(env)),
  ]
}

/** One neutral palette for every grade; "recorded" merely reads at full text
 * strength — no colour ramp across categories. */
function CellView({ cell }: { cell: Cell }) {
  const recorded = cell.grade === 'recorded'
  return (
    <span style={{ fontSize: 12, color: recorded ? 'var(--text)' : 'var(--text-dim)' }}>
      <span className="mono" style={{ fontWeight: recorded ? 600 : 400 }}>{cell.grade}</span>
      {cell.detail && <span> ({cell.detail})</span>}
    </span>
  )
}

export default function ReproducibilityCard({ capsule }: { capsule: AstraCapsule }) {
  const facets = gradeFacets(capsule)
  const th: React.CSSProperties = {
    textAlign: 'left', fontSize: 10.5, textTransform: 'uppercase', letterSpacing: 0.5,
    color: 'var(--text-dim)', padding: '4px 12px 4px 0', borderBottom: '1px solid var(--border)',
  }
  const td: React.CSSProperties = { padding: '6px 12px 6px 0', verticalAlign: 'top' }
  return (
    <div className="card">
      <div className="card-head">
        <span>reproducibility ledger · graded at render time</span>
        <span className="mono">statement_grammar: 1</span>
      </div>
      <div className="card-body" style={{ overflowX: 'auto' }}>
        <div className="hint" style={{ marginBottom: 10 }}>
          Graded from the record each time this page renders — nothing here is
          stored, and there is deliberately no overall score. Closed vocabulary:
          recorded · partial · absent(reason) · not_applicable ·
          nondeterministic_by_design · transcript_only.
        </div>
        <table style={{ borderCollapse: 'collapse', width: '100%' }}>
          <thead>
            <tr>
              <th style={th}>facet</th>
              <th style={th}>self-declared</th>
              <th style={th}>independently observed</th>
            </tr>
          </thead>
          <tbody>
            {facets.map((f) => (
              <tr key={f.name} style={{ borderBottom: '1px solid #ffffff0a' }}>
                <td style={{ ...td, whiteSpace: 'nowrap', fontSize: 12 }}>{f.name}</td>
                <td style={td}><CellView cell={f.self} /></td>
                <td style={td}><CellView cell={f.observed} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
