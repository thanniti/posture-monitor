// web/lib/smoothing.ts
export class PostureMonitorState {
  private history: Array<[number, string]> = [];
  private badPostureSince: number | null = null;

  constructor(
    private windowSeconds: number,
    private debounceSeconds: number
  ) {}

  update(timestamp: number, label: string): [string, boolean] {
    this.history.push([timestamp, label]);
    const cutoff = timestamp - this.windowSeconds;
    while (this.history.length > 0 && this.history[0][0] < cutoff) {
      this.history.shift();
    }

    const counts = new Map<string, number>();
    for (const [, l] of this.history) {
      counts.set(l, (counts.get(l) ?? 0) + 1);
    }
    let smoothed = this.history[0][1];
    let bestCount = -1;
    for (const [label_, count] of counts) {
      if (count > bestCount) {
        bestCount = count;
        smoothed = label_;
      }
    }

    if (smoothed === "CorrectSitting") {
      this.badPostureSince = null;
      return [smoothed, false];
    }

    if (this.badPostureSince === null) {
      this.badPostureSince = timestamp;
    }
    const flagged = timestamp - this.badPostureSince >= this.debounceSeconds;
    return [smoothed, flagged];
  }
}
