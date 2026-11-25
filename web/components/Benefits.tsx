// web/components/Benefits.tsx
const BENEFITS = [
  { title: "Less strain", body: "Reduces back and neck pain caused by hours of slouching." },
  { title: "Better breathing", body: "An upright spine improves breathing and circulation." },
  { title: "Sharper focus", body: "Good posture is linked to more energy and concentration." },
  { title: "Long-term health", body: "Prevents the spinal issues that build up from years of bad habits." },
];

export default function Benefits() {
  return (
    <section id="benefits" className="mx-auto max-w-5xl px-6 py-20">
      <h2 className="text-center text-3xl font-semibold tracking-tight text-ink">
        Why good posture matters
      </h2>
      <div className="mt-12 grid gap-8 sm:grid-cols-2 lg:grid-cols-4">
        {BENEFITS.map((benefit) => (
          <div key={benefit.title} className="rounded-2xl border border-hairline p-6">
            <h3 className="font-semibold text-ink">{benefit.title}</h3>
            <p className="mt-2 text-sm text-muted">{benefit.body}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
