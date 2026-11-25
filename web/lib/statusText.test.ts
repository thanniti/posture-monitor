// web/lib/statusText.test.ts
import { describe, expect, it } from "vitest";
import { statusDisplay } from "./statusText";

describe("statusDisplay", () => {
  it("shows No pose detected in red", () => {
    const { text, colorClass } = statusDisplay({ kind: "no-pose" });
    expect(text).toBe("No pose detected");
    expect(colorClass).toBe("status-red");
  });

  it("prompts to turn to the expected view when calibration isn't matching", () => {
    const { text, colorClass } = statusDisplay({ kind: "calibrating", phase: "side", matches: false, remainingSeconds: 10 });
    expect(text).toBe("Please turn to SIDE view to calibrate");
    expect(colorClass).toBe("status-yellow");
  });

  it("shows a countdown while calibration is matching", () => {
    const { text } = statusDisplay({ kind: "calibrating", phase: "front", matches: true, remainingSeconds: 7.0 });
    expect(text).toBe("Calibrating Front view... 7s");
  });

  it("shows mode, label, and confidence for a live CorrectSitting frame in green", () => {
    const { text, colorClass } = statusDisplay({ kind: "live", mode: "side", label: "CorrectSitting", confidence: 92.3, flagged: false });
    expect(text).toBe("[Side] CorrectSitting (92%)");
    expect(colorClass).toBe("status-green");
  });

  it("omits the confidence suffix when confidence is null, and is red for non-CorrectSitting", () => {
    const { text, colorClass } = statusDisplay({ kind: "live", mode: "front", label: "Slouching", confidence: null, flagged: true });
    expect(text).toBe("[Front] Slouching");
    expect(colorClass).toBe("status-red");
  });
});
