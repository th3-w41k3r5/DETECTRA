"use client"

import { useMemo, useState } from "react"
import useSWR from "swr"
import { ActivitySquare, CheckCircle2, Send, AlertTriangle } from "lucide-react"
import { API_BASE } from "@/lib/config"
import { UploadForm, type PredictionResponse } from "./upload-form"
import { ProbabilityBars } from "./probability-bars"
import type { JSX } from "react/jsx-runtime"

type ModelInfo = { model_version: string; classes: string[] }

const fetcher = (url: string) => fetch(url).then((r) => r.json())

function isNoTumorLabel(label: string) {
  const l = label?.toLowerCase().replaceAll(/\s|_|-/g, "") || ""
  return (
    l.includes("notumor") ||
    l.includes("notumour") ||
    l.includes("notumor") ||
    l.includes("notumorclass") ||
    l === "none"
  )
}

export function Predictor() {
  const { data: info } = useSWR<ModelInfo>(`${API_BASE}/model-info`, fetcher)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [feedbackLabel, setFeedbackLabel] = useState<string>("")
  const [feedbackComment, setFeedbackComment] = useState<string>("")
  const [feedbackSent, setFeedbackSent] = useState(false)
  const [sending, setSending] = useState(false)

  const classes = info?.classes ?? []

  const gradcamSrc = useMemo(() => {
    if (!result?.gradcam) return null
    return `data:image/png;base64,${result.gradcam}`
  }, [result])

  const topEntry = useMemo(() => {
    if (!result?.probabilities) return null
    const sorted = Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    return sorted[0] || null
  }, [result])

  const predictedLabel = result?.prediction
  const topPct = topEntry ? Math.round(topEntry[1] * 100) : null
  const noTumor = predictedLabel ? isNoTumorLabel(predictedLabel) : false

  let tone = { label: "", border: "", icon: null as JSX.Element | null, bgClass: "", textClass: "" }
  if (predictedLabel) {
    if (noTumor) {
      if ((topPct ?? 0) >= 80) {
        tone = {
          label: "No tumor detected",
          border: "var(--chart-2)", // use valid token
          icon: <CheckCircle2 size={18} style={{ color: "var(--chart-2)" }} />,
          bgClass: "bg-secondary",
          textClass: "text-foreground",
        }
      } else {
        tone = {
          label: "Likely no tumor (uncertain)",
          border: "var(--chart-5)", // token family
          icon: <AlertTriangle size={18} style={{ color: "var(--chart-5)" }} />,
          bgClass: "bg-secondary",
          textClass: "text-foreground",
        }
      }
    } else {
      tone = {
        label: "Tumor detected",
        border: "var(--destructive)", // use valid token
        icon: <AlertTriangle size={18} className="text-destructive-foreground" />,
        bgClass: "bg-destructive/10",
        textClass: "text-destructive-foreground",
      }
    }
  }

  const sendFeedback = async () => {
    if (!result?.request_id || !feedbackLabel) return
    setSending(true)
    try {
      const res = await fetch(`${API_BASE}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          request_id: result.request_id,
          correct_label: feedbackLabel,
          comment: feedbackComment,
        }),
      })
      if (!res.ok) throw new Error(await res.text())
      setFeedbackSent(true)
    } catch (e: any) {
      console.error("[v0] Feedback error:", e?.message)
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="grid gap-6 md:grid-cols-2">
      <div>
        <UploadForm onResult={setResult} />
        {/* Model info card */}
        <div className="mt-4 rounded-lg border border-border bg-card p-4 md:p-6">
          <div className="mb-3 flex items-center gap-2">
            <div className="inline-flex h-8 w-8 items-center justify-center rounded-md bg-primary text-primary-foreground">
              <ActivitySquare size={18} />
            </div>
            <div>
              <h3 className="text-sm font-semibold">Model</h3>
              <p className="text-xs text-muted-foreground">Version: {info?.model_version ?? "…"}</p>
            </div>
          </div>
          <div className="text-sm text-muted-foreground">
            Classes: {classes.length > 0 ? classes.join(", ") : "Loading classes…"}
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <div className="rounded-lg border border-border bg-card p-4 md:p-6">
          <h2 className="mb-2 text-lg font-semibold">Result</h2>
          {result ? (
            <div className="space-y-4">
              {/* Detection summary banner with clear red/green wording */}
              <div
                className={`rounded-md border p-3 ${tone.bgClass} ${tone.textClass}`}
                style={{ borderColor: tone.border }}
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="flex min-w-0 items-start gap-2">
                    <div className="mt-0.5 shrink-0">{tone.icon}</div>
                    <div className="min-w-0">
                      <div className="text-sm font-semibold">
                        {tone.label}
                        {predictedLabel && !noTumor ? ` (${predictedLabel})` : null}
                      </div>
                      <div className="mt-1 text-xs opacity-80">
                        Confidence: {topPct ?? "—"}%. This is an AI-aided result, not a medical diagnosis.
                      </div>
                    </div>
                  </div>
                  <div className="text-xs md:self-start">
                    <span className="rounded-md bg-secondary px-2 py-1 text-secondary-foreground">
                      Req: {result.request_id}
                    </span>
                  </div>
                </div>
              </div>

              {/* Existing chips */}
              <div className="flex flex-wrap items-center gap-2">
                <span className="rounded-md bg-primary px-2 py-1 text-sm font-medium text-primary-foreground">
                  Prediction: {result.prediction}
                </span>
              </div>

              {/* Probabilities with clear color mapping */}
              <ProbabilityBars probs={result.probabilities} predicted={predictedLabel} />

              {/* Add “What this means” + “Next steps” for clarity */}
              <div className="grid gap-2 rounded-md border border-border bg-muted/20 p-3">
                <p className="text-sm font-medium">What this means</p>
                <ul className="list-disc pl-5 text-sm text-muted-foreground">
                  {noTumor ? (
                    <>
                      <li>No tumor class has the highest probability for this image.</li>
                      <li>If symptoms persist or confidence is low, consider re-imaging or expert review.</li>
                    </>
                  ) : (
                    <>
                      <li>A tumor class has the highest probability: {predictedLabel}.</li>
                      <li>Review the Grad-CAM to see model focus areas; confirm with a radiologist.</li>
                    </>
                  )}
                </ul>
                <p className="pt-1 text-xs text-muted-foreground">
                  DETECTRA assists clinicians and is not a substitute for professional judgment.
                </p>
              </div>

              {gradcamSrc && (
                <div>
                  <h3 className="mb-2 text-sm font-medium text-foreground">Grad-CAM Overlay</h3>
                  <div className="overflow-hidden rounded-md border border-border">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={gradcamSrc || "/placeholder.svg"}
                      alt="Grad-CAM heatmap overlay"
                      className="max-h-[420px] w-full object-cover"
                    />
                  </div>
                </div>
              )}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Upload an MRI to see predictions and probabilities.</p>
          )}
        </div>

        <div className="rounded-lg border border-border bg-card p-4 md:p-6">
          <h3 className="mb-3 text-sm font-semibold">Feedback</h3>
          {feedbackSent ? (
            <div className="flex items-center gap-2 rounded-md border border-border bg-accent px-3 py-2 text-sm text-accent-foreground">
              <CheckCircle2 size={18} />
              Thanks for your feedback!
            </div>
          ) : (
            <div className="grid gap-3">
              <label className="text-sm text-muted-foreground">
                Is the prediction correct? If not, select the correct label.
              </label>
              <select
                className="rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={feedbackLabel}
                onChange={(e) => setFeedbackLabel(e.target.value)}
                disabled={!result}
              >
                <option value="">{result ? "Select label…" : "Predict first…"}</option>
                {classes.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
              <textarea
                className="min-h-24 rounded-md border border-input bg-background px-3 py-2 text-sm"
                placeholder="Optional comment…"
                value={feedbackComment}
                onChange={(e) => setFeedbackComment(e.target.value)}
                disabled={!result}
              />
              <button
                type="button"
                onClick={sendFeedback}
                disabled={!result || !feedbackLabel || sending}
                className="inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <Send size={18} />
                {sending ? "Sending…" : "Send feedback"}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
