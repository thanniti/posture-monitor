import fs from "fs";
import path from "path";
import type { NextConfig } from "next";

// Read the installed @mediapipe/tasks-vision version at build time so the
// MediaPipe WASM CDN URL used in lib/pose.ts can never drift from the actual
// npm dependency version. We can't `import "@mediapipe/tasks-vision/package.json"`
// directly because the package's "exports" map doesn't expose that subpath
// (Node rejects it with ERR_PACKAGE_PATH_NOT_EXPORTED even via require()) -
// so instead we resolve the package's main entry (which *is* exported) and
// read package.json from its directory via plain fs, bypassing module
// resolution entirely.
const mediapipeTasksVisionPkgDir = path.dirname(
  require.resolve("@mediapipe/tasks-vision")
);
const mediapipeTasksVisionVersion: string = JSON.parse(
  fs.readFileSync(path.join(mediapipeTasksVisionPkgDir, "package.json"), "utf8")
).version;

const nextConfig: NextConfig = {
  /* config options here */
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  env: {
    NEXT_PUBLIC_MEDIAPIPE_TASKS_VISION_VERSION: mediapipeTasksVisionVersion,
  },
};

export default nextConfig;
