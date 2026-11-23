#!/usr/bin/env node
// One command to run everything locally:
//   1. make sure the MongoDB container is up (Docker), unless --no-mongo
//   2. start the API server (server/index.js) on 127.0.0.1:5001
//   3. start the Vite dev server on http://localhost:8080
// Ctrl+C stops both.
import { spawn, spawnSync } from "node:child_process";
import { existsSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const C = { api: "\x1b[36m", web: "\x1b[35m", dev: "\x1b[33m", off: "\x1b[0m" };
const log = (tag, msg) => console.log(`${C[tag]}[${tag}]${C.off} ${msg}`);

// Local env files (gitignored). Created once with sensible defaults.
if (!existsSync(join(root, ".env"))) {
  writeFileSync(join(root, ".env"), "VITE_API_URL=http://localhost:5001\n");
  log("dev", "created .env");
}
if (!existsSync(join(root, "server", ".env"))) {
  writeFileSync(join(root, "server", ".env"), "MONGO_URI=mongodb://127.0.0.1:27017/mlcode\nPORT=5001\n");
  log("dev", "created server/.env");
}

const venvPy = process.platform === "win32" ? join(root, ".venv", "Scripts", "python.exe") : join(root, ".venv", "bin", "python");
if (!existsSync(venvPy)) {
  log("dev", "WARNING: no .venv found. The kata needs it: see README-LOCAL.md (python -m venv .venv, then install torch).");
}

if (!args.includes("--no-mongo")) {
  const docker = (a) => spawnSync("docker", a, { encoding: "utf8" });
  const ps = docker(["ps", "-a", "--filter", "name=^mlcode-mongo$", "--format", "{{.Status}}"]);
  if (ps.error || ps.status !== 0) {
    log("dev", "Docker is not available, skipping MongoDB. Kata works without it; sign-in and the ML problem set do not.");
  } else if (!ps.stdout.trim()) {
    log("dev", "creating the mlcode-mongo container…");
    const r = docker(["run", "-d", "--name", "mlcode-mongo", "-p", "127.0.0.1:27017:27017", "-v", "mlcode-mongo-data:/data/db", "--restart", "unless-stopped", "mongo:7"]);
    log("dev", r.status === 0 ? "MongoDB started" : `could not start MongoDB: ${r.stderr.trim()}`);
  } else if (!ps.stdout.startsWith("Up")) {
    const r = docker(["start", "mlcode-mongo"]);
    log("dev", r.status === 0 ? "MongoDB started" : `could not start MongoDB: ${r.stderr.trim()}`);
  } else {
    log("dev", "MongoDB already running");
  }
}

const kids = [];
function run(tag, cmd, cmdArgs, cwd) {
  const child = spawn(cmd, cmdArgs, { cwd, shell: process.platform === "win32", env: process.env });
  kids.push(child);
  const pipe = (stream) => {
    let buf = "";
    stream.on("data", (d) => {
      buf += d.toString();
      const lines = buf.split(/\r?\n/);
      buf = lines.pop();
      for (const l of lines) if (l.trim()) log(tag, l);
    });
  };
  pipe(child.stdout);
  pipe(child.stderr);
  child.on("exit", (code) => {
    log(tag, `exited (${code})`);
    shutdown(code ?? 0);
  });
}

let stopping = false;
function shutdown(code = 0) {
  if (stopping) return;
  stopping = true;
  for (const k of kids) {
    if (process.platform === "win32") spawnSync("taskkill", ["/pid", String(k.pid), "/T", "/F"]);
    else k.kill("SIGTERM");
  }
  process.exit(code);
}
process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));

run("api", "node", ["index.js"], join(root, "server"));
run("web", "npx", ["vite", "--port", "8080", "--strictPort"], root);
log("dev", "open http://localhost:8080/kata");
