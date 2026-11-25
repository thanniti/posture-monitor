// web/lib/orientation.ts
import { FRONT_VIEW_RATIO_THRESHOLD, SIDE_VIEW_RATIO_THRESHOLD } from "./config";

export type Orientation = "side" | "front";

export function rawOrientationReading(ratio: number): Orientation | null {
  if (ratio < SIDE_VIEW_RATIO_THRESHOLD) return "side";
  if (ratio > FRONT_VIEW_RATIO_THRESHOLD) return "front";
  return null;
}

export class OrientationState {
  confirmedMode: Orientation;
  private history: Array<[number, Orientation | null]> = [];

  constructor(
    private windowSeconds: number,
    initialMode: Orientation = "side"
  ) {
    this.confirmedMode = initialMode;
  }

  update(timestamp: number, ratio: number): Orientation {
    const reading = rawOrientationReading(ratio);
    this.history.push([timestamp, reading]);
    const cutoff = timestamp - this.windowSeconds;
    while (this.history.length > 0 && this.history[0][0] < cutoff) {
      this.history.shift();
    }

    let sideVotes = 0;
    let frontVotes = 0;
    for (const [, r] of this.history) {
      if (r === "side") sideVotes++;
      else if (r === "front") frontVotes++;
    }
    if (sideVotes > frontVotes) this.confirmedMode = "side";
    else if (frontVotes > sideVotes) this.confirmedMode = "front";
    return this.confirmedMode;
  }
}
