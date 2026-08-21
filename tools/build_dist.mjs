// Package web/ into dist/ — a self-contained static site, ready to upload.
//
//   npm run build                       # dist/, models split at 25MB
//   npm run build -- --chunk-size 90MB  # a host with a bigger cap
//   npm run build -- --models chat      # only ship what you need
//   npm run build -- --no-chunk         # leave the .onnx files whole
//
// Two things make this more than a copy.
//
// **Chunking.** Static hosts cap individual files — Cloudflare Pages at 25 MiB,
// which every expert graph (124MB) is well over. So each model is split into
// numbered parts and the manifest records them in order; `fetchModel` in
// model.js concatenates them back into exactly the original bytes. The default
// is 25 *million* bytes rather than 25 MiB, so a part is comfortably under the
// cap however the host counts it.
//
// **Reporting.** The build says what it produced and refuses to pretend a file
// it cannot split (the onnxruntime WASM, which the runtime loads itself) is
// within a limit it is not.

import { copyFileSync, mkdirSync, openSync, readFileSync, readSync, closeSync,
         readdirSync, rmSync, statSync, writeFileSync, writeSync } from "node:fs";
import { dirname, join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const WEB = join(root, "web");
const args = parseArgs(process.argv.slice(2));
const DIST = join(root, args.out ?? "dist");
const CHUNK_BYTES = args["no-chunk"] ? Infinity : parseSize(args["chunk-size"] ?? "25MB");
const WANTED = args.models ? new Set(args.models.split(",")) : null;

// Copied verbatim. `models/` is handled separately because it is the only part
// that gets rewritten on the way through.
const STATIC = ["index.html", "css", "js", "vendor"];

// ------------------------------------------------------------------ preflight

for (const name of ["index.html", "js"]) {
  if (!exists(join(WEB, name))) fatal(`web/${name} is missing — is the checkout complete?`);
}
if (!exists(join(WEB, "vendor", "ort"))) {
  fatal("web/vendor/ort is missing — run `npm install && npm run vendor`");
}
const manifestPath = join(WEB, "models", "manifest.json");
if (!exists(manifestPath)) {
  fatal("web/models/manifest.json is missing — run `python -m sodachat.export_onnx`");
}
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
if (WANTED) {
  for (const name of WANTED) {
    if (!manifest.models[name]) fatal(`--models names ${name}, which is not in the manifest`);
  }
}

// ---------------------------------------------------------------------- build

rmSync(DIST, { recursive: true, force: true });
mkdirSync(join(DIST, "models"), { recursive: true });
for (const name of STATIC) copyInto(join(WEB, name), join(DIST, name));

const models = {};
const rows = [];
for (const [name, spec] of Object.entries(manifest.models)) {
  if (WANTED && !WANTED.has(name)) continue;
  const source = join(WEB, "models", spec.file);
  if (!exists(source)) fatal(`${spec.file} is in the manifest but not on disk — re-export`);

  const size = statSync(source).size;
  if (size !== spec.bytes) {
    fatal(`${spec.file} is ${size} bytes, manifest says ${spec.bytes} — re-export`);
  }
  const chunks = writeChunks(source, join(DIST, "models"), spec.file, CHUNK_BYTES);
  models[name] = { ...spec, chunks };
  rows.push([name, fmt(size), chunks.length ? `${chunks.length} parts` : "whole"]);

  const tokenizer = join(WEB, "models", spec.tokenizer);
  if (exists(tokenizer)) copyFileSync(tokenizer, join(DIST, "models", spec.tokenizer));
}
if (!Object.keys(models).length) fatal("no models selected — nothing to build");

writeFileSync(
  join(DIST, "models", "manifest.json"),
  JSON.stringify({ ...manifest, chunkBytes: CHUNK_BYTES, models }, null, 2) + "\n",
);
writeFileSync(join(DIST, "_headers"), headersFile(models));

// --------------------------------------------------------------------- report

const oversized = walk(DIST).filter((p) => statSync(p).size > CHUNK_BYTES);
console.log(`dist/ built from web/ — ${rows.length} model(s)\n`);
for (const [name, size, how] of rows) console.log(`  ${name.padEnd(16)} ${size.padStart(9)}  ${how}`);
console.log(`\n  ${walk(DIST).length} files, ${fmt(totalSize(DIST))} total`);

if (oversized.length) {
  // The WASM runtime is loaded by onnxruntime itself, so it cannot be split the
  // way a model can. Better to name it than to let a deploy fail on upload.
  console.log(`\n  over the ${fmt(CHUNK_BYTES)} limit and not splittable:`);
  for (const path of oversized) {
    console.log(`    ${relative(DIST, path)}  ${fmt(statSync(path).size)}`);
  }
  console.log("    (these must be served by a host that accepts them)");
}
console.log(`\nserve it with:  python -m sodachat.web --root ${relative(root, DIST)}`);

// -------------------------------------------------------------------- helpers

/**
 * Split `source` into `<file>.part000`, `.part001`, … under `outDir`.
 * Returns the part names in order, or [] when the file fits as it is (in which
 * case it is copied whole and the manifest keeps pointing at `spec.file`).
 */
function writeChunks(source, outDir, fileName, limit) {
  const size = statSync(source).size;
  if (size <= limit) {
    copyFileSync(source, join(outDir, fileName));
    return [];
  }
  const parts = [];
  const buffer = Buffer.alloc(Math.min(limit, 8 << 20));
  const input = openSync(source, "r");
  try {
    let offset = 0;
    while (offset < size) {
      const name = `${fileName}.part${String(parts.length).padStart(3, "0")}`;
      const output = openSync(join(outDir, name), "w");
      try {
        // Stream it: a 124MB graph should not have to be resident to be split.
        let written = 0;
        while (written < limit && offset < size) {
          const want = Math.min(buffer.length, limit - written, size - offset);
          const read = readSync(input, buffer, 0, want, offset);
          if (!read) break;
          writeSync(output, buffer, 0, read);
          written += read;
          offset += read;
        }
      } finally {
        closeSync(output);
      }
      parts.push(name);
    }
  } finally {
    closeSync(input);
  }
  return parts;
}

/**
 * A `_headers` file, which Cloudflare Pages and Netlify read (and other hosts
 * ignore harmlessly).
 *
 * The cross-origin isolation headers are the ones that matter: without them the
 * browser withholds `SharedArrayBuffer`, onnxruntime-web silently drops to a
 * single thread, and a deployed page is several times slower than the same
 * files served locally by `sodachat.web` — with nothing on screen to say why.
 *
 * Cache rules are written per file rather than as `/models/*` so that the
 * manifest and tokenizer, which sit in that directory and are rewritten by
 * every re-export, do not get pinned in caches for a year alongside the blobs.
 */
function headersFile(models) {
  const immutable = new Set();
  for (const spec of Object.values(models)) {
    for (const part of spec.chunks?.length ? spec.chunks : [spec.file]) {
      immutable.add(`/models/${part}`);
    }
  }
  for (const path of walk(join(DIST, "vendor"))) {
    if (path.endsWith(".wasm")) immutable.add("/" + relative(DIST, path).split(sep).join("/"));
  }
  const lines = [
    "# Generated by tools/build_dist.mjs.",
    "# Cross-origin isolation is what lets onnxruntime-web use threads.",
    "/*",
    "  Cross-Origin-Opener-Policy: same-origin",
    "  Cross-Origin-Embedder-Policy: require-corp",
    "",
    "# Model chunks and the WASM runtime change only when this is rebuilt.",
  ];
  for (const path of [...immutable].sort()) {
    lines.push(path, "  Cache-Control: public, max-age=31536000, immutable", "");
  }
  return lines.join("\n");
}

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    if (!argv[i].startsWith("--")) continue;
    const key = argv[i].slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith("--")) out[key] = true;
    else out[key] = argv[++i];
  }
  return out;
}

/** "25MB" | "25MiB" | "26214400" -> bytes. */
function parseSize(value) {
  const match = String(value).trim().match(/^(\d+(?:\.\d+)?)\s*(gib|mib|kib|gb|mb|kb|b)?$/i);
  if (!match) fatal(`could not read --chunk-size ${value}`);
  const units = { b: 1, kb: 1e3, mb: 1e6, gb: 1e9, kib: 1024, mib: 1024 ** 2, gib: 1024 ** 3 };
  return Math.floor(Number(match[1]) * units[(match[2] ?? "b").toLowerCase()]);
}

function copyInto(from, to) {
  if (statSync(from).isFile()) {
    mkdirSync(dirname(to), { recursive: true });
    copyFileSync(from, to);
    return;
  }
  mkdirSync(to, { recursive: true });
  for (const entry of readdirSync(from)) copyInto(join(from, entry), join(to, entry));
}

function walk(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) =>
    entry.isDirectory() ? walk(join(dir, entry.name)) : [join(dir, entry.name)],
  );
}

// Declarations, not `const` arrows: the preflight above runs before this point
// in the file, and only a function declaration is hoisted to meet it.
function totalSize(dir) {
  return walk(dir).reduce((n, p) => n + statSync(p).size, 0);
}

function exists(path) {
  try {
    statSync(path);
    return true;
  } catch {
    return false;
  }
}

function fmt(bytes) {
  return bytes === Infinity ? "unlimited" : `${(bytes / 1e6).toFixed(1)} MB`;
}

function fatal(message) {
  console.error(message);
  process.exit(1);
}
