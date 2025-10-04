"use client"

import type React from "react"
import { useMemo, useState } from "react"
import { Brain, Upload, Info } from "lucide-react"
import { API_BASE } from "@/lib/config"

export type PredictionResponse = {
  request_id: string
  prediction: string
  probabilities: Record<string, number>
  gradcam?: string // base64 png (no prefix)
}

export function UploadForm({
  onResult,
}: {
  onResult: (res: PredictionResponse) => void
}) {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [explain, setExplain] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const canSubmit = useMemo(() => !!file && !loading, [file, loading])

  const handleFile = (f: File | null) => {
    setError(null)
    setFile(f)
    if (f) {
      const url = URL.createObjectURL(f)
      setPreview(url)
    } else {
      setPreview(null)
    }
  }

  const onChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] || null
    handleFile(f)
  }

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!file) return

    setLoading(true)
    setError(null)

    try {
      const fd = new FormData()
      fd.append("file", file)
      fd.append("explain", String(explain))

      const res = await fetch(`${API_BASE}/predict-image`, {
        method: "POST",
        body: fd,
      })

      if (!res.ok) {
        const txt = await res.text()
        throw new Error(txt || "Prediction failed")
      }

      const data = (await res.json()) as PredictionResponse
      onResult(data)
    } catch (err: any) {
      console.error("[v0] Prediction error:", err?.message)
      setError("Prediction failed. Please try another image.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={submit} className="rounded-lg border border-border bg-card p-4 md:p-6">
      <div className="mb-4 flex items-center gap-2">
        <div className="inline-flex h-8 w-8 items-center justify-center rounded-md bg-primary text-primary-foreground">
          <Brain size={18} />
        </div>
        <div>
          <h2 className="text-base md:text-lg font-semibold leading-none">DETECTRA</h2>
          <p className="text-xs md:text-sm text-muted-foreground">MRI Brain Tumor Detection</p>
        </div>
      </div>

      <label htmlFor="mri" className="mb-3 block text-sm font-medium text-foreground">
        Upload MRI image
      </label>
      <div className="mb-4">
        <input
          id="mri"
          type="file"
          accept="image/*"
          onChange={onChange}
          className="w-full cursor-pointer rounded-md border border-input bg-background p-2 text-sm file:mr-4 file:rounded-md file:border-0 file:bg-secondary file:px-3 file:py-2 file:text-sm file:text-secondary-foreground hover:file:bg-accent hover:file:text-accent-foreground"
          aria-describedby="mri-help"
        />
        <p id="mri-help" className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
          <Info size={14} />
          Supported: JPG, PNG. Best results with clear axial MRI slices.
        </p>
      </div>

      <div className="mb-4 flex items-center justify-between">
        <label className="text-sm text-foreground">Explain with Grad-CAM</label>
        <button
          type="button"
          onClick={() => setExplain((s) => !s)}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
            explain ? "bg-primary" : "bg-muted"
          }`}
          aria-pressed={explain}
          aria-label="Toggle Grad-CAM explainability"
        >
          <span
            className={`inline-block h-5 w-5 transform rounded-full bg-background transition-transform ${
              explain ? "translate-x-6" : "translate-x-1"
            }`}
          />
        </button>
      </div>

      {preview && (
        <div className="mb-4 overflow-hidden rounded-md border border-border">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={preview || "/placeholder.svg"}
            alt="Selected MRI preview"
            className="max-h-64 w-full object-cover"
          />
        </div>
      )}

      {error && (
        <div
          role="alert"
          className="mb-3 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive-foreground"
        >
          {error}
        </div>
      )}

      <button
        type="submit"
        disabled={!canSubmit}
        className="inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50"
      >
        <Upload size={18} />
        {loading ? "Analyzing..." : "Analyze MRI"}
      </button>
    </form>
  )
}
