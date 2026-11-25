// web/components/HowToUse.tsx
const TIPS = [
  "Use good lighting so your face and shoulders are clearly visible.",
  "Both sitting side-on and facing the camera work.",
  "Sit naturally during the 15-second calibration — don't force \"perfect\" posture.",
];

export default function HowToUse() {
  return (
    <section className="mx-auto max-w-3xl px-6 py-20">
      <h2 className="text-center text-3xl font-semibold tracking-tight text-ink">How to use it</h2>
      <ul className="mt-8 space-y-3">
        {TIPS.map((tip) => (
          <li key={tip} className="flex gap-3 text-muted">
            <span className="text-accent">•</span>
            <span>{tip}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
