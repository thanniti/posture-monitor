// web/lib/statusText.ts
import type { SessionStatus } from "./session";

export interface StatusDisplay {
  text: string;
  colorClass: "status-green" | "status-red" | "status-yellow";
}

function capitalize(word: string): string {
  return word[0].toUpperCase() + word.slice(1);
}

export function statusDisplay(status: SessionStatus): StatusDisplay {
  if (status.kind === "no-pose") {
    return { text: "No pose detected", colorClass: "status-red" };
  }
  if (status.kind === "calibrating") {
    if (!status.matches) {
      return {
        text: `Please turn to ${status.phase.toUpperCase()} view to calibrate`,
        colorClass: "status-yellow",
      };
    }
    return {
      text: `Calibrating ${capitalize(status.phase)} view... ${Math.round(status.remainingSeconds)}s`,
      colorClass: "status-yellow",
    };
  }
  const confidenceSuffix = status.confidence !== null ? ` (${Math.round(status.confidence)}%)` : "";
  return {
    text: `[${capitalize(status.mode)}] ${status.label}${confidenceSuffix}`,
    colorClass: status.label === "CorrectSitting" ? "status-green" : "status-red",
  };
}
