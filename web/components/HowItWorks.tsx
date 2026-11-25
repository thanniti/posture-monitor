// web/components/HowItWorks.tsx
const STEPS = [
  {
    title: "We watch your pose",
    body: "MediaPipe tracks your head, neck, and shoulders only. Nothing is recorded or sent anywhere.",
  },
  {
    title: "15-second calibration",
    body: "It learns your own proportions first — it doesn't assume you're sitting correctly yet.",
  },
  {
    title: "Live feedback",
    body: "Flags slouching or leaning in real time. Side view uses a trained ML model; facing the camera uses geometric rules against your own calibrated baseline.",
  },
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="mx-auto max-w-5xl px-6 py-20">
      <h2 className="text-center text-3xl font-semibold tracking-tight text-ink">How it works</h2>
      <div className="mt-12 grid gap-8 sm:grid-cols-3">
        {STEPS.map((step, i) => (
          <div key={step.title}>
            <div className="text-sm font-semibold text-accent">{String(i + 1).padStart(2, "0")}</div>
            <h3 className="mt-2 font-semibold text-ink">{step.title}</h3>
            <p className="mt-2 text-sm text-muted">{step.body}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
