// web/components/Hero.tsx
export default function Hero() {
  return (
    <section className="flex flex-col items-center gap-4 px-6 py-24 text-center">
      <p className="text-xs font-semibold uppercase tracking-widest text-muted">PostureIQ</p>
      <h1 className="max-w-2xl text-5xl font-semibold tracking-tight text-ink sm:text-6xl">
        Sit better. Feel better.
      </h1>
      <p className="max-w-xl text-lg text-muted">
        Real-time posture coaching that runs entirely in your browser.
      </p>
      <a
        href="#try-it"
        className="mt-4 rounded-full bg-accent px-8 py-3 font-medium text-white shadow-md transition hover:opacity-90"
      >
        Try it now
      </a>
      <p className="text-sm text-muted">No sign-up. Nothing leaves your device.</p>
    </section>
  );
}
