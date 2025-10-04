type ProbMap = Record<string, number>

function isNoTumorLabel(label: string) {
  const l = label.toLowerCase().replaceAll(/\s|_|-/g, "")
  return l.includes("notumor") || l.includes("notumour") || l.includes("notumor") || l.includes("noTumor")
}

export function ProbabilityBars({
  probs,
  predicted,
}: {
  probs: ProbMap
  predicted?: string
}) {
  const entries = Object.entries(probs).sort((a, b) => b[1] - a[1])

  return (
    <div className="space-y-3">
      {entries.map(([label, value]) => {
        const pct = Math.round(value * 100)
        const isNoTumor = isNoTumorLabel(label)
        const isPred = predicted && label.toLowerCase() === predicted.toLowerCase()

        // use valid tokens (v4): --destructive, --chart-2, --primary
        let barColor = "var(--primary)"
        if (isNoTumor) barColor = "var(--chart-2)"
        if (isPred && !isNoTumor) barColor = "var(--destructive)"
        if (isPred && isNoTumor) barColor = "var(--chart-2)"

        return (
          <div
            key={label}
            className={`space-y-1 rounded-md ${isPred ? "bg-muted/30 p-2" : ""}`}
            style={
              isPred
                ? {
                    outline: `1px solid ${isNoTumor ? "var(--chart-2)" : "var(--destructive)"}`,
                  }
                : undefined
            }
          >
            <div className="flex items-center justify-between text-sm">
              <span className="flex items-center gap-2 text-muted-foreground">
                <span
                  aria-hidden
                  className="inline-block h-2.5 w-2.5 rounded-full"
                  style={{ backgroundColor: barColor }}
                />
                {label}
              </span>
              <span className="font-medium">{pct}%</span>
            </div>
            <div className="h-2 w-full rounded-md bg-muted">
              <div
                className="h-2 rounded-md"
                style={{ width: `${pct}%`, backgroundColor: barColor }}
                aria-valuenow={pct}
                aria-valuemin={0}
                aria-valuemax={100}
                role="progressbar"
                aria-label={`${label} probability ${pct}%`}
              />
            </div>
          </div>
        )
      })}
      {/* legend */}
      <div className="pt-1 text-xs text-muted-foreground">
        <div className="flex flex-wrap items-center gap-3">
          <span className="inline-flex items-center gap-1">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: "var(--destructive)" }} />
            Predicted tumor
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: "var(--chart-2)" }} />
            No tumor
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: "var(--primary)" }} />
            Other classes
          </span>
        </div>
      </div>
    </div>
  )
}
