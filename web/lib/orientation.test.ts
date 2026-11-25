// web/lib/orientation.test.ts
import { describe, expect, it } from "vitest";
import { OrientationState, rawOrientationReading } from "./orientation";

describe("rawOrientationReading", () => {
  it("reads side below the side threshold", () => {
    expect(rawOrientationReading(0.5)).toBe("side");
  });

  it("reads front above the front threshold", () => {
    expect(rawOrientationReading(2.0)).toBe("front");
  });

  it("is null (ambiguous) between the thresholds", () => {
    expect(rawOrientationReading(1.45)).toBeNull();
  });
});

describe("OrientationState", () => {
  it("starts at the initial mode", () => {
    const state = new OrientationState(1.5, "side");
    expect(state.confirmedMode).toBe("side");
  });

  it("does not flip on a single extreme-lean frame", () => {
    const state = new OrientationState(1.5, "side");
    state.update(0.0, 0.9); // side
    const mode = state.update(1.0, 2.0); // one front-reading frame, outvoted by the prior side reading
    expect(mode).toBe("side");
  });

  it("flips once a second consecutive reading agrees", () => {
    const state = new OrientationState(1.5, "side");
    state.update(0.0, 0.9); // side
    state.update(1.0, 2.0); // front (outvoted, stays side)
    const mode = state.update(2.0, 2.0); // second consecutive front reading wins
    expect(mode).toBe("front");
  });

  it("does not let ambiguous readings affect the mode", () => {
    const state = new OrientationState(1.5, "side");
    const mode = state.update(0.0, 1.45); // ambiguous, no vote
    expect(mode).toBe("side");
  });

  it("ages old readings out of the window", () => {
    const state = new OrientationState(1.5, "side");
    state.update(0.0, 0.9); // side
    const mode = state.update(10.0, 2.0); // far enough that the old side reading is gone
    expect(mode).toBe("front");
  });
});
