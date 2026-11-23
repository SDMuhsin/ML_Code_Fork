// A pool of pre-started Python workers (harness.py).
//
// Importing torch and creating a CUDA context takes a few seconds, so workers
// are started ahead of time and sit at "ready" until a job arrives. Each worker
// runs one job and exits; the pool immediately starts a replacement. Every run gets a
// warm interpreter and a clean one.

const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const READY = '@@KATA_READY@@';
const RESULT = '@@KATA_RESULT@@';
const HARNESS = path.join(__dirname, 'harness.py');
const REPO_ROOT = path.resolve(__dirname, '..', '..');

function resolvePython() {
  if (process.env.MLCODE_PYTHON) return process.env.MLCODE_PYTHON;
  const candidates = [
    path.join(REPO_ROOT, '.venv', 'Scripts', 'python.exe'),
    path.join(REPO_ROOT, '.venv', 'bin', 'python'),
  ];
  for (const c of candidates) if (fs.existsSync(c)) return c;
  return process.platform === 'win32' ? 'python' : 'python3';
}

const PYTHON = resolvePython();
const WORK_DIR = path.join(os.tmpdir(), 'mlcode-kata');
fs.mkdirSync(WORK_DIR, { recursive: true });

class WarmPool {
  constructor({ size = 2 } = {}) {
    this.size = size;
    this.ready = [];
    this.starting = 0;
    this.waiters = [];
    this.env = null;
    this.lastError = null;
    this.failures = 0;
    this.retryTimer = null;
  }

  start() {
    this._topUp();
    return this;
  }

  status() {
    return { python: PYTHON, ready: this.ready.length, starting: this.starting, env: this.env, lastError: this.lastError };
  }

  _topUp() {
    if (this.retryTimer) return;
    while (this.ready.length + this.starting < this.size) this._spawn();
  }

  _spawn() {
    this.starting++;
    const proc = spawn(PYTHON, ['-u', HARNESS], {
      cwd: WORK_DIR,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1' },
      windowsHide: true,
    });
    const w = { proc, buf: '', stderr: '', state: 'starting', onResult: null, onExit: null };

    proc.stdout.setEncoding('utf8');
    proc.stdout.on('data', (chunk) => {
      w.buf += chunk;
      let i;
      while ((i = w.buf.indexOf('\n')) >= 0) {
        const line = w.buf.slice(0, i).replace(/\r$/, '');
        w.buf = w.buf.slice(i + 1);
        this._line(w, line);
      }
    });
    proc.stderr.setEncoding('utf8');
    proc.stderr.on('data', (c) => {
      w.stderr = (w.stderr + c).slice(-20000);
    });
    proc.on('error', (err) => {
      // e.g. the python executable does not exist
      w.stderr += `\n${err.message}`;
    });
    proc.on('exit', (code) => {
      if (w.state === 'starting') {
        this.starting--;
        this.failures++;
        this.lastError = `Python worker failed to start (exit ${code}) using ${PYTHON}\n${w.stderr.trim()}`;
        console.error('[kata] ' + this.lastError);
        // Back off rather than spin if the interpreter is broken.
        const delay = Math.min(30000, 1000 * 2 ** Math.min(this.failures, 5));
        this.retryTimer = setTimeout(() => {
          this.retryTimer = null;
          this._topUp();
        }, delay);
        this._failWaiters();
      } else if (w.state === 'ready') {
        this.ready = this.ready.filter((x) => x !== w);
        this._topUp();
      } else if (w.state === 'busy' && w.onExit) {
        w.onExit(code);
      }
      w.state = 'dead';
    });
  }

  _line(w, line) {
    if (line.startsWith(READY)) {
      try {
        this.env = JSON.parse(line.slice(READY.length)).env;
      } catch (_) { /* keep the previous env */ }
      this.starting--;
      this.failures = 0;
      this.lastError = null;
      w.state = 'ready';
      const waiter = this.waiters.shift();
      if (waiter) waiter.resolve(w);
      else this.ready.push(w);
    } else if (line.startsWith(RESULT) && w.onResult) {
      let parsed;
      try {
        parsed = JSON.parse(line.slice(RESULT.length));
      } catch (e) {
        parsed = { status: 'error', checks: [], error: { type: 'ProtocolError', message: e.message, traceback: line.slice(0, 2000) } };
      }
      w.onResult(parsed);
    }
  }

  _failWaiters() {
    if (!this.lastError) return;
    const ws = this.waiters.splice(0);
    for (const x of ws) x.reject(new Error(this.lastError));
  }

  _acquire(timeoutMs) {
    if (this.ready.length) return Promise.resolve(this.ready.shift());
    this._topUp();
    return new Promise((resolve, reject) => {
      const entry = { resolve: null, reject: null };
      const timer = setTimeout(() => {
        this.waiters = this.waiters.filter((x) => x !== entry);
        reject(new Error(this.lastError || 'Timed out waiting for a Python worker to start'));
      }, timeoutMs);
      entry.resolve = (w) => { clearTimeout(timer); resolve(w); };
      entry.reject = (e) => { clearTimeout(timer); reject(e); };
      this.waiters.push(entry);
    });
  }

  /** Run one job. Always resolves with a result object (never rejects). */
  async run(job, { timeoutMs = 20000 } = {}) {
    let w;
    try {
      w = await this._acquire(90000);
    } catch (e) {
      return { status: 'error', checks: [], console: [], error: { phase: 'worker', type: 'WorkerUnavailable', message: e.message, traceback: e.message } };
    }
    w.state = 'busy';
    this._topUp();
    const started = Date.now();
    return new Promise((resolve) => {
      let done = false;
      const finish = (r) => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        r.wall_ms = Date.now() - started;
        resolve(r);
      };
      const timer = setTimeout(() => {
        try { w.proc.kill(); } catch (_) { /* already gone */ }
        finish({
          status: 'error', checks: [], console: [],
          error: { phase: 'user', type: 'Timeout', message: `Stopped after ${timeoutMs / 1000}s. An infinite loop, or a very large tensor?`, traceback: '' },
        });
      }, timeoutMs);
      w.onResult = (r) => finish(r);
      w.onExit = (code) => finish({
        status: 'error', checks: [], console: [],
        error: { phase: 'worker', type: 'WorkerCrashed', message: `The Python process exited (code ${code}) before reporting a result.`, traceback: w.stderr.trim() },
      });
      w.proc.stdin.write(JSON.stringify(job) + '\n');
    });
  }

  shutdown() {
    for (const w of this.ready) { try { w.proc.kill(); } catch (_) { /* ignore */ } }
    this.ready = [];
  }
}

let shared = null;
function getPool() {
  if (!shared) shared = new WarmPool({ size: Number(process.env.KATA_POOL_SIZE || 2) }).start();
  return shared;
}

module.exports = { WarmPool, getPool, PYTHON, resolvePython };
