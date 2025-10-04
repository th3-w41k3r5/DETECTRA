import { ThemeToggle } from "@/components/theme-toggle"
import { Predictor } from "@/components/predictor"
import { Brain } from "lucide-react"

export default function Page() {
  return (
    <main className="mx-auto max-w-6xl px-4 py-8 md:py-12">
      <header className="mb-8 flex flex-col items-start justify-between gap-4 md:mb-12 md:flex-row md:items-center">
        <div className="flex items-center gap-3">
          <div className="inline-flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Brain size={22} />
          </div>
          <div>
            <h1 className="text-pretty text-2xl font-semibold leading-tight md:text-3xl">DETECTRA</h1>
            <p className="text-sm text-muted-foreground">Premium MRI Brain Tumor Detection with Explainability</p>
          </div>
        </div>
        <ThemeToggle />
      </header>

      <section className="mb-8 rounded-xl border border-border bg-card p-6">
        <div className="grid items-center gap-6 md:grid-cols-2">
          <div>
            <h2 className="text-balance text-xl font-semibold md:text-2xl">Upload. Analyze. Understand.</h2>
            <p className="mt-2 text-pretty text-sm text-muted-foreground md:text-base">
              Get instant predictions across multiple tumor classes and visualize model focus using Grad-CAM. Built for
              clarity, speed, and trust.
            </p>
          </div>
          <ul className="grid gap-2 text-sm text-muted-foreground">
            <li className="flex items-center gap-2">
              <span className="inline-block h-2 w-2 rounded-full bg-primary" />
              Clean, professional interface with light/dark modes
            </li>
            <li className="flex items-center gap-2">
              <span className="inline-block h-2 w-2 rounded-full bg-primary" />
              Secure image upload with detailed probabilities
            </li>
            <li className="flex items-center gap-2">
              <span className="inline-block h-2 w-2 rounded-full bg-primary" />
              Feedback loop to improve quality over time
            </li>
          </ul>
        </div>
      </section>

      <Predictor />
    </main>
  )
}
