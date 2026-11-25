// web/components/Nav.tsx
export default function Nav() {
  return (
    <header className="sticky top-0 z-10 flex items-center justify-between border-b border-hairline bg-white/80 px-6 py-4 backdrop-blur">
      <span className="text-lg font-semibold text-ink">PostureIQ</span>
      <nav className="flex gap-6 text-sm text-ink">
        <a href="#benefits" className="hover:text-accent">Benefits</a>
        <a href="#how-it-works" className="hover:text-accent">How it works</a>
        <a
          href="#try-it"
          className="rounded-full bg-accent px-4 py-1.5 font-medium text-white hover:opacity-90"
        >
          Try it now
        </a>
      </nav>
    </header>
  );
}
