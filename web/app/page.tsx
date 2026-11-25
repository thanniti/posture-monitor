import Nav from "@/components/Nav";
import Hero from "@/components/Hero";
import Benefits from "@/components/Benefits";
import HowItWorks from "@/components/HowItWorks";
import HowToUse from "@/components/HowToUse";
import TryItTool from "@/components/TryItTool";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <Benefits />
        <HowItWorks />
        <HowToUse />
        <section id="try-it" className="px-6 py-20">
          <h2 className="mb-12 text-center text-3xl font-semibold tracking-tight text-ink">Try it</h2>
          <TryItTool />
        </section>
      </main>
      <Footer />
    </>
  );
}
