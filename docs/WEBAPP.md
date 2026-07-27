# ProtForge Web UI — setup & access (Kempner / FASRC)

The web UI is a [Streamlit](https://streamlit.io) front-end for the Snakemake
pipeline. It runs **on a cluster login node** (where it can read your files and
call `sbatch`/`squeue`/`sacct`), and you reach it from your laptop's browser.
It does not run any models itself — it edits a config, launches Snakemake, and
shows progress and results.

## What it does

Four tabs, all scoped to the **active session** (sessions live in
`.sessions/<id>/` and each has its own `config.yaml`, run log, and backups):

- **Configuration** — edit pipeline stages, input/output paths, per-stage SLURM
  resources, and import/generate input files. Writes the session `config.yaml`.
- **Run Pipeline** — validate inputs, dry-run, and launch Snakemake. Launching
  starts a background `snakemake` process that submits SLURM jobs.
- **Job Monitor** — per-stage progress bars (from output artifacts), live
  `squeue` jobs, recent `sacct` history, and a per-job log viewer.
- **Results** — browse predicted structures in 3D, compare structures
  (target vs queries: CA RMSD + TM-score, e.g. cross-predictor agreement or
  mutant-vs-WT), and view per-stage runtime / node-hour analytics from the
  Snakemake benchmarks.
- **Saturation Mutagenesis** — standalone single-sequence tool: paste a
  sequence, pick an ESM-C size, launch one GPU job, and inspect the
  position × amino-acid log-likelihood-ratio matrix as an interactive heatmap
  (with CSV download). Independent of the pipeline; reuses the session's
  container/cache/SLURM settings.

> The Streamlit process owns the Snakemake controller it launches. If that
> process dies, **already-submitted SLURM jobs keep running**, but no new stages
> get submitted. Keep it alive for the duration of a run (use `tmux`, below).

## Prerequisites

The cluster-side setup (containers, model weights, SLURM account, shared MSA
databases) is covered in [`CLUSTER_SETUP.md`](CLUSTER_SETUP.md). Do that first.
The webapp only needs the **host launcher environment** — the same env that runs
the `snakemake` CLI.

## Install (one-time)

On a login node, in your ProtForge checkout:

```bash
# Option A — the shared host env, recommended. Nothing to install.
module load python && mamba activate "$PROTFORGE_ASSETS/envs/host"

# Option B — your own, if you can't read the shared copy or need extra packages.
# Put it on lab storage, not in ~ or ~/.conda: home quotas are small.
module load python
mamba create -p "$PROTFORGE_ROOT/envs/host" python=3.13 -y
mamba activate "$PROTFORGE_ROOT/envs/host"
pip install -r requirements-host.txt          # includes the webapp + viz deps

# Option C — webapp only, into an env that already has Snakemake:
pip install -r webapp/requirements.txt
```

That's the whole install — the heavy ML stacks live in the containers, not here.
The Results tab's 3D viewer (`py3Dmol`) and charts (`plotly`) are included above;
without them the tab still works (download link + native charts).

**Activate the environment before starting the app.** Streamlit launches
Snakemake as a child process, so it inherits the shell you started it from, and
the workflow's local chunking rules shell out to a bare `python`. Starting the
app without an activated environment fails on the first rule with
`python: command not found` — after the UI has reported a successful launch.

## Run it

Always run inside `tmux` so the app — and the Snakemake controller it owns —
survive an SSH disconnect:

```bash
ssh <user>@<login-node>          # e.g. holylogin06.rc.fas.harvard.edu
tmux new -s protforge            # or: tmux attach -t protforge
cd "$PROTFORGE_ROOT/ProtForge"
module load python && mamba activate "$PROTFORGE_ASSETS/envs/host"
streamlit run webapp/app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
```

Detach with `Ctrl-b d` (the app keeps running). Pick a port unlikely to clash
with other users (8501 is the default; bump it if it's taken).

> **Login-node etiquette:** the app itself is light, but the Snakemake
> controller runs for the length of the pipeline. That's normal launcher usage,
> but don't run heavy compute in it. For very long runs, prefer Open OnDemand on
> a compute node (below).

## Accessing it from your laptop

You asked whether there's something better than `ssh -L`. Three options, from
least to most setup:

### 1. SSH port-forward (works everywhere, fully manual)

```bash
ssh -L 8501:localhost:8501 <user>@<login-node>
# then open http://localhost:8501
```

`webapp/connect.sh` wraps this: it prompts for your login details, opens the
tunnel, and starts Streamlit if it isn't already running. Fine for occasional
use; the annoyance is you must keep the tunnel shell open and rerun it each time.

### 2. VS Code Remote-SSH (recommended — near-automatic, no admin)

Open the cluster folder with the **Remote-SSH** extension and run the
`streamlit run …` command in VS Code's integrated terminal. VS Code
**auto-detects the port and forwards it**, and shows a clickable
`http://localhost:8501` link. No manual `-L`, the forward follows your editor
session, and it survives reconnects better than a raw tunnel. This is the
lowest-friction upgrade for day-to-day use.

### 3. Open OnDemand (recommended for tunnel-free / shared use)

FASRC provides Open OnDemand (the VDI portal, e.g.
`https://rcood.rc.fas.harvard.edu`). OOD proxies web apps running on cluster
nodes, so you get a **clickable browser link with no SSH tunnel at all**, and it
runs on a **compute node** (better for long pipelines than a login node). Two
ways to use it:

- **Quick:** launch an OOD *Remote Desktop* or *Jupyter* interactive session,
  open a terminal inside it, start Streamlit as above, and browse
  `http://localhost:8501` from within that session.
- **Best:** register a small custom OOD *interactive app* that `salloc`s a node
  and runs `streamlit run webapp/app.py` on `$port`, exposing it through OOD's
  node proxy. One-time setup, then it's a one-click "ProtForge" button for you
  (and labmates) with no tunnel. Ask FASRC/Kempner support for the interactive-app
  sandbox path if you want to go this route.

**Avoid** public tunnels (ngrok / cloudflared): they expose an unauthenticated
app to the internet and are generally against cluster acceptable-use policy.

**Recommendation:** use VS Code Remote-SSH for solo day-to-day work; set up an
Open OnDemand app if you want a tunnel-free link, want it on a compute node, or
want to share it.

## First run

1. **Sessions** (sidebar): a `Default` session is created automatically. Create
   one per experiment; "Clone config from" copies an existing session's settings.
2. **Configuration → Input / Output:** set the FASTA dir (MSA on) or YAML dir
   (MSA off) and the output dir. The page auto-scans and shows sequence stats.
   Use **Import** to copy/validate files from elsewhere on the cluster, or to
   generate inputs from a CSV/TSV or random mutations.
3. **Configuration:** toggle stages, set per-stage resources/partitions, then
   **Save Configuration**.
4. **Run Pipeline:** check the dry run, tick the confirm box, and **Launch**.
5. **Job Monitor:** watch progress bars and SLURM jobs (auto-refreshes).
6. **Results:** once structures land, view them in 3D and check the run analytics.

## Stopping & lifecycle

- **Stop a run:** Run Pipeline → *Stop Pipeline* kills the Snakemake controller.
  Already-submitted SLURM jobs continue; cancel those with `scancel` if needed.
- **App vs run:** closing the browser does nothing to the run. Killing the
  Streamlit process (or losing the `tmux` session it's in) stops *new* job
  submission but not in-flight jobs.
- **Config backups:** every save snapshots the previous `config.yaml` into the
  session's `.config_backups/`.

## Troubleshooting

- **`Address already in use`:** another Streamlit (yours or a labmate's) holds
  the port. Pick a different `--server.port` and forward that one.
- **App vanished after logout:** you didn't run it in `tmux`. Restart it inside
  `tmux` (or via an OOD session).
- **Results tab: "3D viewer needs py3Dmol" / no charts:** install the optional
  deps (`pip install -r webapp/requirements.txt` or `pip install '.[viz]'`).
- **No SLURM jobs / history shown:** the app shells out to `squeue`/`sacct`;
  make sure those are on `PATH` in the env you launched Streamlit from.
- **"Launch blocked by input validation":** the message lists the exact problem
  (missing input dir, wrong file type, no YAMLs for Boltz/OpenFold with MSA off).
