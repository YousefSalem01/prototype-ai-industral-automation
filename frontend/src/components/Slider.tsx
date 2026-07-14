import type { ColumnMeta } from '../api'

interface SliderProps {
  col: ColumnMeta
  value: number
  onChange: (v: number) => void
  recommended?: number | null
  accent?: string
}

const LABELS: Record<string, string> = {
  gas_temperature: 'Gas Temperature',
  oxygen_pct: 'Oxygen',
  flow_rate: 'Air Flow Rate',
  furnace_speed: 'Furnace Speed',
  feed_rate: 'Feed Rate',
  material_moisture: 'Material Moisture',
  ambient_temp: 'Ambient Temp',
}

export function Slider({ col, value, onChange, recommended = null, accent = '#38bdf8' }: SliderProps) {
  const label = LABELS[col.name] ?? col.name
  const span = col.max - col.min
  const step = span > 100 ? 1 : span > 10 ? 0.1 : 0.01
  const pct = ((value - col.min) / span) * 100
  const recPct = recommended != null ? ((recommended - col.min) / span) * 100 : null

  return (
    <div className="rounded-xl border border-line bg-panel-2/60 p-4">
      <div className="mb-2 flex items-baseline justify-between">
        <span className="text-sm font-medium text-slate-200">{label}</span>
        <span className="font-mono text-sm tabular-nums" style={{ color: accent }}>
          {value.toFixed(step < 1 ? 2 : 0)}
          <span className="ml-1 text-xs text-slate-500">{col.unit}</span>
        </span>
      </div>
      <div className="relative">
        <input
          type="range"
          className="w-full"
          min={col.min}
          max={col.max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          style={{
            background: `linear-gradient(to right, ${accent} ${pct}%, #1f2a37 ${pct}%)`,
          }}
        />
        {recPct != null && (
          <div
            className="pointer-events-none absolute -top-1 h-4 w-0.5 bg-good"
            style={{ left: `calc(${Math.max(0, Math.min(100, recPct))}% - 1px)` }}
            title={`Recommended: ${recommended?.toFixed(2)} ${col.unit}`}
          />
        )}
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-slate-600">
        <span>{col.min}</span>
        {recommended != null && (
          <span className="text-good">rec {recommended.toFixed(step < 1 ? 2 : 0)}</span>
        )}
        <span>{col.max}</span>
      </div>
    </div>
  )
}
