// Semicircular quality gauge rendered as an SVG arc.

interface GaugeProps {
  value: number
  min?: number
  max?: number
  label?: string
  compareTo?: number | null // optional "before" marker
}

function color(pct: number): string {
  if (pct >= 0.75) return '#34d399'
  if (pct >= 0.5) return '#fbbf24'
  return '#f87171'
}

export function Gauge({ value, min = 0, max = 100, label = 'Quality', compareTo = null }: GaugeProps) {
  const clamped = Math.max(min, Math.min(max, value))
  const pct = (clamped - min) / (max - min)
  const R = 90
  const CX = 110
  const CY = 110
  const startAngle = Math.PI // 180deg
  const endAngle = 0 // 0deg
  const angle = startAngle - pct * (startAngle - endAngle)

  const arcPoint = (a: number, r = R) => [CX + r * Math.cos(a), CY - r * Math.sin(a)]
  const [sx, sy] = arcPoint(startAngle)
  const [ex, ey] = arcPoint(endAngle)
  const [vx, vy] = arcPoint(angle)
  const [nx, ny] = arcPoint(angle, R - 26) // needle base offset

  const cmpAngle =
    compareTo != null
      ? startAngle - ((Math.max(min, Math.min(max, compareTo)) - min) / (max - min)) * Math.PI
      : null

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 220 140" className="w-full max-w-[280px]">
        {/* track */}
        <path
          d={`M ${sx} ${sy} A ${R} ${R} 0 0 1 ${ex} ${ey}`}
          fill="none"
          stroke="#1f2a37"
          strokeWidth={16}
          strokeLinecap="round"
        />
        {/* value arc */}
        <path
          d={`M ${sx} ${sy} A ${R} ${R} 0 ${pct > 0.5 ? 1 : 0} 1 ${vx} ${vy}`}
          fill="none"
          stroke={color(pct)}
          strokeWidth={16}
          strokeLinecap="round"
          style={{ transition: 'all 0.5s cubic-bezier(0.22,1,0.36,1)' }}
        />
        {/* compare marker */}
        {cmpAngle != null && (
          <line
            x1={CX + (R - 10) * Math.cos(cmpAngle)}
            y1={CY - (R - 10) * Math.sin(cmpAngle)}
            x2={CX + (R + 10) * Math.cos(cmpAngle)}
            y2={CY - (R + 10) * Math.sin(cmpAngle)}
            stroke="#64748b"
            strokeWidth={3}
            strokeDasharray="2 2"
          />
        )}
        {/* needle */}
        <line
          x1={nx}
          y1={ny}
          x2={vx}
          y2={vy}
          stroke={color(pct)}
          strokeWidth={3}
          strokeLinecap="round"
          style={{ transition: 'all 0.5s cubic-bezier(0.22,1,0.36,1)' }}
        />
        <circle cx={CX} cy={CY} r={6} fill={color(pct)} />
      </svg>
      <div className="-mt-6 text-center">
        <div className="text-5xl font-bold tabular-nums" style={{ color: color(pct) }}>
          {value.toFixed(1)}
        </div>
        <div className="mt-1 text-xs uppercase tracking-widest text-slate-400">{label}</div>
      </div>
    </div>
  )
}
