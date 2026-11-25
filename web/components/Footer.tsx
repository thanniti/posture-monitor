// web/components/Footer.tsx
export default function Footer() {
  return (
    <footer className="border-t border-hairline px-6 py-10 text-center text-sm text-muted">
      <p>Built as a class project.</p>
      <a
        href="https://github.com/thanniti/posture-monitor"
        className="mt-2 inline-block text-accent hover:underline"
      >
        View the code on GitHub
      </a>
    </footer>
  );
}
