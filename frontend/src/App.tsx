import { useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { api } from './api'
import type { Explanation, ImportanceRow, Meta, OptimizeResult } from './api'
import { Gauge } from './components/Gauge'
import { Slider } from './components/Slider'

type State = Record<string, number>

const LABELS: Record<string, string> = {
  "% Iron Feed": "Iron Feed - نسبة تغذية الحديد",
  "% Silica Feed": "Silica Feed - نسبة تغذية السيليكا",
  "Ore Pulp Flow": "Ore Pulp Flow - تدفق خام اللب",
  "Ore Pulp Density": "Ore Pulp Density - كثافة خام اللب",
  "Starch Flow": "Starch Flow - تدفق النشا",
  "Amina Flow": "Amina Flow - تدفق الأمينا",
  "Ore Pulp pH": "Ore Pulp pH - درجة الحموضة (pH)",
  "Flotation Column 01 Air Flow": "Col 1 Air Flow - تدفق هواء عمود 1",
  "Flotation Column 02 Air Flow": "Col 2 Air Flow - تدفق هواء عمود 2",
  "Flotation Column 03 Air Flow": "Col 3 Air Flow - تدفق هواء عمود 3",
  "Flotation Column 04 Air Flow": "Col 4 Air Flow - تدفق هواء عمود 4",
  "Flotation Column 05 Air Flow": "Col 5 Air Flow - تدفق هواء عمود 5",
  "Flotation Column 06 Air Flow": "Col 6 Air Flow - تدفق هواء عمود 6",
  "Flotation Column 07 Air Flow": "Col 7 Air Flow - تدفق هواء عمود 7",
  "Flotation Column 01 Level": "Col 1 Level - مستوى رغوة عمود 1",
  "Flotation Column 02 Level": "Col 2 Level - مستوى رغوة عمود 2",
  "Flotation Column 03 Level": "Col 3 Level - مستوى رغوة عمود 3",
  "Flotation Column 04 Level": "Col 4 Level - مستوى رغوة عمود 4",
  "Flotation Column 05 Level": "Col 5 Level - مستوى رغوة عمود 5",
  "Flotation Column 06 Level": "Col 6 Level - مستوى رغوة عمود 6",
  "Flotation Column 07 Level": "Col 7 Level - مستوى رغوة عمود 7",
}
const nice = (n: string) => LABELS[n] ?? n

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null)
  const [state, setState] = useState<State>({})
  const [live, setLive] = useState<number | null>(null)
  const [explain, setExplain] = useState<Explanation | null>(null)
  const [importance, setImportance] = useState<ImportanceRow[]>([])
  const [opt, setOpt] = useState<OptimizeResult | null>(null)
  const [optimizing, setOptimizing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [online, setOnline] = useState(false)
  const debounce = useRef<number | undefined>(undefined)

  // Initial load: contract + global importance.
  useEffect(() => {
    ;(async () => {
      try {
        const m = await api.meta()
        setMeta(m)
        const init: State = {}
        m.columns.forEach((c) => (init[c.name] = c.default))
        setState(init)
        setOnline(true)
        api.importance().then((r) => setImportance(r.importance)).catch(() => {})
      } catch (e) {
        setError((e as Error).message + ' — is the backend running on :8000?')
      }
    })()
  }, [])

  // Debounced live prediction + explanation whenever the state changes.
  useEffect(() => {
    if (!meta || Object.keys(state).length === 0) return
    window.clearTimeout(debounce.current)
    debounce.current = window.setTimeout(async () => {
      try {
        const [p, ex] = await Promise.all([api.predict(state), api.explain(state)])
        setLive(p.predicted_quality)
        setExplain(ex)
        setError(null)
      } catch (e) {
        setError((e as Error).message)
      }
    }, 250)
  }, [state, meta])

  const runOptimize = async () => {
    if (!meta) return
    setOptimizing(true)
    setError(null)
    try {
      const r = await api.optimize(state)
      setOpt(r)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setOptimizing(false)
    }
  }

  const applyRecommendation = () => {
    if (!opt) return
    setState((s) => ({ ...s, ...opt.recommended_setpoints }))
    setOpt(null)
  }

  const set = (name: string, v: number) => setState((s) => ({ ...s, [name]: v }))

  const fixedCols = useMemo(
    () => meta?.columns.filter((c) => c.role === 'fixed') ?? [],
    [meta],
  )
  const ctrlCols = useMemo(
    () => meta?.columns.filter((c) => c.role === 'controllable') ?? [],
    [meta],
  )

  const minimize = meta?.target.direction === 'minimize'
  const targetName = meta?.target.name ?? 'quality'
  const maxShap = Math.max(1e-6, ...(explain?.contributions.map((c) => Math.abs(c.shap)) ?? [1]))

  return (
    <div className="min-h-screen px-4 py-6 md:px-8">
      {/* Header */}
      <header className="mx-auto mb-6 flex max-w-7xl items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="grid h-11 w-11 place-items-center rounded-xl bg-gradient-to-br from-sky-500/30 to-emerald-500/20 text-2xl">
            🏭
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-tight">Flotation Plant Quality Optimizer</h1>
            <p className="text-xs text-slate-500">AI setpoint recommendations · iron-ore froth flotation (real plant data)</p>
          </div>
        </div>
        <div className="flex items-center gap-2 rounded-full border border-line bg-panel px-3 py-1.5 text-xs">
          <span className={`live-dot h-2 w-2 rounded-full ${online ? 'bg-emerald-400' : 'bg-red-400'}`} />
          {online ? 'Model online' : 'Offline'}
        </div>
      </header>

      {error && (
        <div className="mx-auto mb-4 max-w-7xl rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      <main className="mx-auto grid max-w-7xl gap-5 lg:grid-cols-12">
        {/* LEFT: Inputs */}
        <section className="space-y-5 lg:col-span-5">
          <Panel title="Operating Conditions" subtitle="Measured — not controllable">
            <div className="grid gap-3">
              {fixedCols.map((c) => (
                <Slider
                  key={c.name}
                  col={c}
                  value={state[c.name] ?? c.default}
                  onChange={(v) => set(c.name, v)}
                  accent="#94a3b8"
                />
              ))}
            </div>
          </Panel>

          <Panel title="Controllable Setpoints" subtitle="What the engineer dials">
            <div className="grid gap-3">
              {ctrlCols.map((c) => (
                <Slider
                  key={c.name}
                  col={c}
                  value={state[c.name] ?? c.default}
                  onChange={(v) => set(c.name, v)}
                  recommended={opt?.recommended_setpoints[c.name] ?? null}
                />
              ))}
            </div>
          </Panel>
        </section>

        {/* CENTER: Live quality + optimize */}
        <section className="space-y-5 lg:col-span-4">
          <Panel
            title="Predicted Quality"
            subtitle={
              minimize
                ? `${targetName} · live inference · lower is better`
                : `${targetName} · live inference · higher is better`
            }
          >
            <Gauge
              value={live ?? 0}
              min={meta?.target.min ?? 0}
              max={meta?.target.max ?? 100}
              label={`${targetName} (${meta?.target.unit ?? ''})`}
              compareTo={opt ? opt.current_quality : null}
              betterIsLow={minimize}
            />
            <button
              onClick={runOptimize}
              disabled={optimizing || !online}
              className="mt-4 w-full rounded-xl bg-gradient-to-r from-sky-500 to-cyan-400 px-4 py-3 font-semibold text-slate-900 transition hover:brightness-110 disabled:opacity-50"
            >
              {optimizing ? 'Searching setpoints…' : '⚡ Optimize Setpoints'}
            </button>
          </Panel>

          {opt && (
            <Panel title="Recommendation" subtitle={`Optuna · ${opt.n_trials} trials`}>
              <div className="mb-3 grid grid-cols-3 gap-2 text-center">
                <Stat label="Current" value={opt.current_quality.toFixed(1)} tone="muted" />
                <Stat label="Predicted" value={opt.predicted_quality.toFixed(1)} tone="good" />
                <Stat
                  label="Δ Gain"
                  value={`${opt.delta >= 0 ? '+' : ''}${opt.delta.toFixed(1)}`}
                  tone={opt.delta >= 0 ? 'good' : 'bad'}
                />
              </div>
              <div className="space-y-1.5">
                {ctrlCols.map((c) => {
                  const from = opt.current_setpoints[c.name]
                  const to = opt.recommended_setpoints[c.name]
                  return (
                    <div
                      key={c.name}
                      className="flex items-center justify-between rounded-lg bg-panel px-3 py-2 text-sm"
                    >
                      <span className="text-slate-300">{nice(c.name)}</span>
                      <span className="font-mono tabular-nums text-slate-400">
                        {from?.toFixed(1)} <span className="text-slate-600">→</span>{' '}
                        <span className="text-emerald-300">{to?.toFixed(1)}</span> {c.unit}
                      </span>
                    </div>
                  )
                })}
              </div>
              <button
                onClick={applyRecommendation}
                className="mt-3 w-full rounded-xl border border-emerald-400/40 bg-emerald-400/10 px-4 py-2.5 text-sm font-semibold text-emerald-300 transition hover:bg-emerald-400/20"
              >
                ✓ Apply to setpoints
              </button>
              {!opt.within_limits && (
                <p className="mt-2 text-xs text-amber-300">⚠ A recommendation hit an operating limit.</p>
              )}
            </Panel>
          )}
        </section>

        {/* RIGHT: Explanations */}
        <section className="space-y-5 lg:col-span-3">
          <Panel title="Why this quality?" subtitle="SHAP · local drivers">
            {explain ? (
              <div className="space-y-2">
                {explain.contributions.map((c) => {
                  const w = (Math.abs(c.shap) / maxShap) * 100
                  // Colour by effect on QUALITY, not the raw target sign:
                  // for a minimize target, a negative SHAP (lowers impurity) is good.
                  const good = c.direction === 'improves'
                  return (
                    <div key={c.feature}>
                      <div className="mb-0.5 flex justify-between text-xs">
                        <span className="text-slate-300">{nice(c.feature)}</span>
                        <span className={`font-mono ${good ? 'text-emerald-300' : 'text-red-300'}`}>
                          {c.shap >= 0 ? '+' : ''}
                          {c.shap.toFixed(2)}
                        </span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-panel">
                        <div
                          className={`h-full rounded-full ${good ? 'bg-emerald-400' : 'bg-red-400'}`}
                          style={{ width: `${w}%`, transition: 'width 0.4s ease' }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <p className="text-sm text-slate-500">Adjust a slider to see drivers…</p>
            )}
          </Panel>

          <Panel title="Global Importance" subtitle="Mean |SHAP| across dataset">
            <div className="space-y-2">
              {importance.map((r) => (
                <div key={r.feature}>
                  <div className="mb-0.5 flex justify-between text-xs">
                    <span className="text-slate-300">{nice(r.feature)}</span>
                    <span className="font-mono text-slate-400">{r.importance_pct.toFixed(1)}%</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-panel">
                    <div
                      className={`h-full rounded-full ${r.role === 'controllable' ? 'bg-sky-400' : 'bg-slate-500'}`}
                      style={{ width: `${r.importance_pct}%` }}
                    />
                  </div>
                </div>
              ))}
              <div className="mt-2 flex gap-4 text-[10px] text-slate-500">
                <span className="flex items-center gap-1">
                  <i className="h-2 w-2 rounded-full bg-sky-400" /> controllable
                </span>
                <span className="flex items-center gap-1">
                  <i className="h-2 w-2 rounded-full bg-slate-500" /> fixed
                </span>
              </div>
            </div>
          </Panel>
        </section>
      </main>

      <footer className="mx-auto mt-8 max-w-7xl text-center text-xs text-slate-600">
        Live HTTP integration · FastAPI + XGBoost + Optuna + SHAP · the desktop app talks to this same API.
      </footer>
    </div>
  )
}

function Panel({
  title,
  subtitle,
  children,
}: {
  title: string
  subtitle?: string
  children: ReactNode
}) {
  return (
    <div className="rounded-2xl border border-line bg-panel/70 p-5 shadow-xl shadow-black/20 backdrop-blur">
      <div className="mb-4">
        <h2 className="text-sm font-semibold text-slate-100">{title}</h2>
        {subtitle && <p className="text-xs text-slate-500">{subtitle}</p>}
      </div>
      {children}
    </div>
  )
}

function Stat({ label, value, tone }: { label: string; value: string; tone: 'good' | 'bad' | 'muted' }) {
  const color =
    tone === 'good' ? 'text-emerald-300' : tone === 'bad' ? 'text-red-300' : 'text-slate-300'
  return (
    <div className="rounded-lg bg-panel px-2 py-2">
      <div className={`text-xl font-bold tabular-nums ${color}`}>{value}</div>
      <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
    </div>
  )
}
