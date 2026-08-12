// Copy the onnxruntime-web runtime out of node_modules into web/vendor/.
//
//   npm install && npm run vendor
//
// The page loads these files from its own origin rather than a CDN: the whole
// point of the web backend is that a trained model runs on the visitor's
// machine, and that claim is weaker if the runtime is fetched from someone
// else's server. It also means the app works offline and behind a firewall,
// like every other frontend in this repo.

import { copyFileSync, mkdirSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const from = join(root, "node_modules", "onnxruntime-web", "dist");
const to = join(root, "web", "vendor", "ort");

// The WebGPU bundle, which also serves the wasm backend — so one file is what
// lets the page use WebGPU where the browser has it and fall back where it
// doesn't.
const BUNDLE = "ort.webgpu.bundle.min.mjs";

// Which WASM binaries that bundle loads is a detail of the release: 1.23 shipped
// a `jsep` pair, 1.27 an `asyncify` one. Rather than pin a guess that breaks on
// the next upgrade, read the names back out of the bundle itself.
const WASM_REFERENCE = /ort-wasm[\w.-]*\.(?:mjs|wasm)/g;

try {
  statSync(from);
} catch {
  console.error("run `npm install` first — node_modules/onnxruntime-web is missing");
  process.exit(1);
}

const available = new Set(readdirSync(from));
if (!available.has(BUNDLE)) {
  console.error(`missing ${BUNDLE} in onnxruntime-web/dist — has the layout changed?`);
  process.exit(1);
}
const referenced = [...new Set(readFileSync(join(from, BUNDLE), "utf8").match(WASM_REFERENCE))];
if (!referenced.length) {
  console.error(`${BUNDLE} names no .wasm runtime — cannot tell what to vendor`);
  process.exit(1);
}

mkdirSync(to, { recursive: true });
let copied = 0;
for (const name of [BUNDLE, ...referenced]) {
  if (!available.has(name)) {
    console.error(`${BUNDLE} wants ${name}, which is not in onnxruntime-web/dist`);
    process.exit(1);
  }
  copyFileSync(join(from, name), join(to, name));
  copied += statSync(join(to, name)).size;
}
console.log(
  `vendored ${1 + referenced.length} files (${(copied / 1e6).toFixed(1)} MB) into web/vendor/ort:` +
    `\n  ${[BUNDLE, ...referenced].join("\n  ")}`,
);
