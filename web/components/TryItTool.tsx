// web/components/TryItTool.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import type { PoseLandmarker } from "@mediapipe/tasks-vision";
import { createLandmarker, drawSkeleton, extractKeypoints } from "@/lib/pose";
import { loadModel, type ForestModel } from "@/lib/classifier";
import { PostureSession, type SessionStatus } from "@/lib/session";
import { statusDisplay } from "@/lib/statusText";

type ToolState =
  | "idle"
  | "loading"
  | "running"
  | "camera-error"
  | "load-error"
  | "runtime-error";

export default function TryItTool() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const sessionRef = useRef<PostureSession | null>(null);
  const rafRef = useRef<number | null>(null);
  // Tracks whether the component is currently mounted so in-flight async work
  // in start() can bail out (and clean up anything it already acquired)
  // instead of mutating refs / starting a camera after unmount.
  const isMountedRef = useRef(true);
  // Reentrancy guard: prevents a second start() from running concurrently
  // with one that's already loading/running (e.g. a rapid double-click).
  const isStartingRef = useRef(false);

  const [toolState, setToolState] = useState<ToolState>("idle");
  const [status, setStatus] = useState<SessionStatus>({ kind: "no-pose" });

  function loop() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = landmarkerRef.current;
    const session = sessionRef.current;
    if (!video || !canvas || !landmarker || !session) return;

    try {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");

      const keypoints = extractKeypoints(landmarker, video, performance.now());
      const newStatus = session.processFrame(keypoints, performance.now() / 1000);
      setStatus(newStatus);

      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (keypoints) drawSkeleton(ctx, keypoints);
      }

      rafRef.current = requestAnimationFrame(loop);
    } catch {
      // Stop scheduling further frames and surface a visible error instead
      // of silently freezing on the last-known status.
      rafRef.current = null;
      if (isMountedRef.current) setToolState("runtime-error");
    }
  }

  async function start() {
    if (isStartingRef.current) return;
    isStartingRef.current = true;

    try {
      setToolState("loading");

      // Skip reloading the landmarker/model if they're already loaded (e.g.
      // this is a camera-permission retry after "Try again") so we don't
      // leak the previous landmarker by overwriting it with a new one.
      if (!landmarkerRef.current || !sessionRef.current) {
        let landmarker: PoseLandmarker | null = null;
        try {
          const [createdLandmarker, model] = await Promise.all([
            createLandmarker("/models/pose_landmarker_lite.task"),
            loadModel("/models/classifier.json") as Promise<ForestModel>,
          ]);
          landmarker = createdLandmarker;

          if (!isMountedRef.current) {
            try {
              landmarker.close();
            } catch {
              // ignore
            }
            return;
          }

          landmarkerRef.current = landmarker;
          sessionRef.current = new PostureSession(model);
        } catch {
          if (isMountedRef.current) {
            setToolState("load-error");
          } else {
            try {
              landmarker?.close();
            } catch {
              // ignore
            }
          }
          return;
        }
      }

      let stream: MediaStream | null = null;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });

        if (!isMountedRef.current) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }

        if (!isMountedRef.current) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
      } catch {
        if (isMountedRef.current) {
          setToolState("camera-error");
        } else {
          stream?.getTracks().forEach((track) => track.stop());
        }
        return;
      }

      setToolState("running");
      rafRef.current = requestAnimationFrame(loop);
    } finally {
      isStartingRef.current = false;
    }
  }

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;

      try {
        if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      } catch {
        // ignore
      }

      try {
        landmarkerRef.current?.close();
      } catch {
        // ignore
      }

      try {
        const stream = videoRef.current?.srcObject as MediaStream | null;
        stream?.getTracks().forEach((track) => track.stop());
      } catch {
        // ignore
      }
    };
  }, []);

  const { text, colorClass } = statusDisplay(status);

  return (
    <div className="relative mx-auto aspect-video w-full max-w-3xl overflow-hidden rounded-2xl bg-black shadow-xl">
      <video ref={videoRef} className="h-full w-full object-cover" muted playsInline />
      <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />

      {toolState === "running" && (
        <div className={`absolute left-4 top-4 text-lg font-semibold drop-shadow ${colorClass}`}>{text}</div>
      )}

      {toolState === "idle" && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/90">
          <button
            onClick={start}
            className="rounded-full bg-accent px-6 py-3 font-medium text-white shadow-md transition hover:opacity-90"
          >
            Start camera
          </button>
        </div>
      )}

      {toolState === "loading" && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/90 text-ink">
          Loading posture detector…
        </div>
      )}

      {toolState === "camera-error" && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-white/90 px-6 text-center text-ink">
          <p>Camera access is needed to use this tool.</p>
          <button onClick={start} className="rounded-full bg-accent px-6 py-2 font-medium text-white">
            Try again
          </button>
        </div>
      )}

      {toolState === "load-error" && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/90 px-6 text-center text-ink">
          Couldn&apos;t load the posture detector — please refresh.
        </div>
      )}

      {toolState === "runtime-error" && (
        <div className="absolute inset-0 flex items-center justify-center bg-white/90 px-6 text-center text-ink">
          Something went wrong — please refresh.
        </div>
      )}
    </div>
  );
}
