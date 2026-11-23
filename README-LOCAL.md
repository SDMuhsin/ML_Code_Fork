# Running the kata locally

The kata is the PyTorch practice section at `/kata`. It needs Node 20+ and Python 3.10+.
MongoDB and Docker are optional: only sign-in and the original problem set use them.

## 1. Install

```bash
npm install
cd server && npm install && cd ..
```

Create the Python environment in the repo root, as `.venv` (the server looks for it there):

```bash
python -m venv .venv
# Windows:      .venv\Scripts\python -m pip install --upgrade pip
# Linux/macOS:  .venv/bin/python -m pip install --upgrade pip
```

Install PyTorch. Pick one:

| Machine | Command (use the venv's python) |
|---|---|
| No NVIDIA GPU (any OS) | `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| NVIDIA GPU | `python -m pip install torch --index-url https://download.pytorch.org/whl/cu130` (or the CUDA version your driver supports, see pytorch.org) |
| macOS | `python -m pip install torch` |

The CPU build is about 200 MB; the CUDA build is about 3 GB.

Then:

```bash
python -m pip install -r requirements-kata.txt
```

## 2. Run

```bash
npm run dev:all
```

Open http://localhost:8080/kata. `dev:all` creates `.env` and `server/.env` with local defaults if they are missing,
starts the API on 127.0.0.1:5001 and Vite on :8080, and starts MongoDB in Docker if Docker is installed.
Use `npm run dev:all -- --no-mongo` to skip MongoDB.

To use a Python outside `.venv`, set `MLCODE_PYTHON` to its path.

## Without a GPU

Everything works on CPU. Every problem is verified on CPU, and on CUDA when a GPU is present. With no GPU the
home page shows "No CUDA device" and the CUDA option in the workspace is disabled. The few demos in Lesson 2 module 2g that read GPU memory print the
numbers measured on an RTX 3050 Ti instead, and say so.

## Progress

Your progress is stored in `server/data/kata-progress.json`. Copy that file between machines to carry your
solved problems and review schedule. Set `KATA_DATA_DIR` to keep it somewhere else.

## Checking content

```bash
cd server
npm run kata:test                 # scheduler, progress and track tests
node kata/verify.js               # every problem: solution passes, starter fails cleanly, demos run
node kata/verify.js 2a 2d-3       # some modules or problems
node kata/verify.js --cpu         # skip CUDA
```

`KATA_VERIFY_POOL` sets how many Python workers verify uses (default 3; use 1 or 2 on a machine with little RAM).
How to write problems and add lessons: `server/kata/content/README.md`.
