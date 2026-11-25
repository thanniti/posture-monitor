// web/lib/smoothing.test.ts
import { describe, expect, it } from "vitest";
import { PostureMonitorState } from "./smoothing";

describe("PostureMonitorState", () => {
  it("returns the single label as smoothed when history has one entry", () => {
    const state = new PostureMonitorState(2.5, 5.0);
    const [smoothed, flagged] = state.update(0.0, "CorrectSitting");
    expect(smoothed).toBe("CorrectSitting");
    expect(flagged).toBe(false);
  });

  it("smooths via majority vote within the window", () => {
    const state = new PostureMonitorState(2.5, 5.0);
    state.update(0.0, "LeanForward");
    state.update(1.0, "LeanForward");
    const [smoothed] = state.update(2.0, "CorrectSitting");
    expect(smoothed).toBe("LeanForward");
  });

  it("drops entries older than the window", () => {
    const state = new PostureMonitorState(1.0, 5.0);
    state.update(0.0, "LeanForward");
    const [smoothed] = state.update(5.0, "CorrectSitting");
    expect(smoothed).toBe("CorrectSitting");
  });

  it("does not flag until debounceSeconds of sustained bad posture has passed", () => {
    const state = new PostureMonitorState(0.1, 5.0);
    state.update(0.0, "LeanForward");
    const [, flaggedEarly] = state.update(1.0, "LeanForward");
    expect(flaggedEarly).toBe(false);
    const [, flaggedLate] = state.update(6.0, "LeanForward");
    expect(flaggedLate).toBe(true);
  });

  it("resets and never flags once CorrectSitting reappears", () => {
    const state = new PostureMonitorState(0.1, 5.0);
    state.update(0.0, "LeanForward");
    state.update(6.0, "CorrectSitting");
    const [, flagged] = state.update(6.1, "LeanForward");
    expect(flagged).toBe(false);
  });
});
