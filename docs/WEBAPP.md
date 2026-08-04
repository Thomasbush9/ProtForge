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

```bash
ssh <user>@<login-node>          # e.g. holylogin06.rc.fas.harvard.edu
cd "$PROTFORGE_ROOT/ProtForge"
module load python && mamba activate "$PROTFORGE_ASSETS/envs/host"
bash webapp/serve.sh
```

`serve.sh` picks a free port in 15000–20000 rather than a fixed 8501, so
labmates on the same login node never collide, and prints the `ssh -L` line to
paste on your laptop. It records the port and PID in
`~/.protforge/webapp.<node>.state`, so running it again **reattaches** to your
app instead of starting a second one. The app is `nohup`ed: it (and the
Snakemake controller it owns) survive your logging out, no `tmux` needed. Use
`tmux` anyway if you want to watch the Streamlit output live; otherwise it goes
to `~/.protforge/webapp.<node>.log`.

Other modes: `--status` (is mine up, and where), `--stop` (kill it — submitted
SLURM jobs keep running), `--print-port` (port alone, for scripts).

Overrides, all optional: `PROTFORGE_PORT` forces a specific port (this is the
hook for Open OnDemand, which hands you a `$port`), `PROTFORGE_PORT_RANGE`
changes the candidate range, `PROTFORGE_STREAMLIT` points at a specific
executable if the host env isn't active.

> **Login-node etiquette:** the app itself is light, but the Snakemake
> controller runs for the length of the pipeline. That's normal launcher usage,
> but don't run heavy compute in it. For very long runs, prefer Open OnDemand on
> a compute node (below).

## Accessing it from your laptop

You asked whether there's something better than `ssh -L`. Three options, from
least to most setup:

### 1. SSH port-forward (works everywhere, fully manual)

```bash
ssh -L <port>:localhost:<port> <user>@<login-node>
# then open http://localhost:<port>
```

`<port>` is whatever `serve.sh` printed on the cluster.

Run `bash webapp/connect.sh` **from your laptop** to skip that bookkeeping: it
prompts for your login details, starts (or reattaches to) the app over SSH,
learns the port it landed on, and forwards it. It multiplexes both steps over
one SSH connection, which matters because login-node names round-robin — a
second `ssh` can land on a different node than the one running your app.

Fine for occasional use; the annoyance is you must keep the tunnel shell open
and rerun it each time.

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

- **`Address already in use`:** you launched `streamlit run` by hand on a port a
  labmate holds. Use `bash webapp/serve.sh`, which picks a free one.
- **Browser shows nothing / connection refused:** your tunnel points at a port
  or node the app isn't on — most often after a reconnect landed you on a
  different login node. `bash webapp/serve.sh --status` on each login node says
  where yours actually is; `connect.sh` handles this for you.
- **App vanished after logout:** `serve.sh` `nohup`s it, so this means it
  crashed — check `~/.protforge/webapp.<node>.log`. (If you started it by hand
  without `nohup`/`tmux`, that's the cause.)
- **Results tab: "3D viewer needs py3Dmol" / no charts:** install the optional
  deps (`pip install -r webapp/requirements.txt` or `pip install '.[viz]'`).
- **No SLURM jobs / history shown:** the app shells out to `squeue`/`sacct`;
  make sure those are on `PATH` in the env you launched Streamlit from.
- **"Launch blocked by input validation":** the message lists the exact problem
  (missing input dir, wrong file type, no YAMLs for Boltz/OpenFold with MSA off).
