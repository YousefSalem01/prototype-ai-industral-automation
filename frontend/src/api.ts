// Typed client for the Industrial Quality Optimizer backend.
// In dev, calls go to /api/* which Vite proxies to http://127.0.0.1:8000.

const BASE = '/api'

export type Role = 'controllable' | 'fixed' | 'target'

export interface ColumnMeta {
  name: string
  role: Role
  unit: string
  description: string
  min: number
  max: number
  default: number
}

export interface Meta {
  target: { name: string; unit: string; min: number; max: number; direction: string }
  controllable: string[]
  fixed: string[]
  columns: ColumnMeta[]
}

export interface OptimizeResult {
  fixed_conditions: Record<string, number>
  current_setpoints: Record<string, number>
  current_quality: number
  recommended_setpoints: Record<string, number>
  predicted_quality: number
  delta: number
  n_trials: number
  within_limits: boolean
}

export interface Contribution {
  feature: string
  role: Role
  unit: string
  value: number
  shap: number
  target_effect: 'increases' | 'decreases' // effect on the raw target value
  direction: 'improves' | 'worsens' // effect on QUALITY (direction-aware)
}

export interface Explanation {
  base_value: number
  predicted_quality: number
  contributions: Contribution[]
}

export interface ImportanceRow {
  feature: string
  role: Role
  unit: string
  mean_abs_shap: number
  importance_pct: number
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error(detail?.detail ?? `${path} failed (${res.status})`)
  }
  return res.json() as Promise<T>
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`${path} failed (${res.status})`)
  return res.json() as Promise<T>
}

export const api = {
  meta: () => get<Meta>('/meta'),
  health: () => get<{ status: string; model_loaded: boolean }>('/health'),
  importance: () => get<{ importance: ImportanceRow[] }>('/importance'),
  predict: (state: Record<string, number>) =>
    post<{ predicted_quality: number; target: string }>('/predict', { state }),
  optimize: (state: Record<string, number>, n_trials?: number) =>
    post<OptimizeResult>('/optimize', { state, n_trials }),
  explain: (state: Record<string, number>) =>
    post<Explanation>('/explain', { state }),
}
