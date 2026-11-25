import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PostureIQ — Real-time posture coaching",
  description: "A browser-based posture monitor that watches your sitting posture and gives real-time feedback, powered by an on-device ML model.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-white">{children}</body>
    </html>
  );
}
