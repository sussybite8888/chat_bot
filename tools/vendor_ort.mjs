// Copy the onnxruntime-web runtime out of node_modules into web/vendor/.
//
//   npm install && npm run vendor
//
// The page loads these files from its own origin rather than a CDN: the whole
// point of the web backend is that a trained model runs on the visitor's
// machine, and that claim is weaker if the runtime is fetched from someone
// else's server. It also means the app works offline and behind a firewall,
// like every other frontend in this repo.

import { copyFileSync, mkdirSync, readdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const from = join(root, "node_modules", "onnxruntime-web", "dist");
const to = join(root, "web", "vendor", "ort");

// The bundled ESM build plus the WASM binaries it loads at runtime. `jsep` is
// the WebGPU-capable build — same file serves the wasm backend, so shipping it
// is what lets the page offer WebGPU where the browser has it.
const WANTED = [
  "ort.webgpu.bundle.min.mjs",
  "ort-wasm-simd-threaded.jsep.mjs",
  "ort-wasm-simd-threaded.jsep.wasm",
];

try {
  statSync(from);
} catch {
  console.error("run `npm install` first — node_modules/onnxruntime-web is missing");
  process.exit(1);
}

mkdirSync(to, { recursive: true });
const available = new Set(readdirSync(from));
let copied = 0;
for (const name of WANTED) {
  if (!available.has(name)) {
    console.error(`missing ${name} in onnxruntime-web/dist — has the layout changed?`);
    process.exit(1);
  }
  copyFileSync(join(from, name), join(to, name));
  copied += statSync(join(to, name)).size;
}
console.log(`vendored ${WANTED.length} files (${(copied / 1e6).toFixed(1)} MB) into web/vendor/ort`);
