# ProtForge

ProtForge is a protein prediction pipeline context centered on running sequence-to-structure-and-analysis stages on a cluster. Its language distinguishes branch names, execution modes, and pipeline stages so design discussions do not blur them together.

## Language

**container2 branch**:
The Git branch used to develop the next container-supported execution design. It is a branch name, not an execution topology.
_Avoid_: container2 runtime, the new container

**Container mode**:
A pipeline execution mode where a stage runs inside a Singularity/Apptainer container.
_Avoid_: container2, docker mode

**Legacy mode**:
A pipeline execution mode where a stage runs through the existing cluster environment rather than a container.
_Avoid_: normal mode, old container

**Stage**:
One major pipeline step in ProtForge, such as MSA, Boltz, ESM, ESMFold, or ES.
_Avoid_: job, container

**Runtime mount**:
A host path made visible inside `Container mode` at job launch so a stage can read shared inputs or write user-selected outputs.
_Avoid_: baked-in path, container storage

**External asset**:
A large stage dependency kept outside the image and supplied through `Runtime mount`, such as reference databases or model weights.
_Avoid_: bundled dependency, downloaded-at-runtime asset

**Hybrid image family**:
A container design with a shared base plus stage-specific images layered on top.
_Avoid_: one giant image, unrelated per-stage containers

**Mount set**:
The exact collection of host paths bound into a container for one pipeline run.
_Avoid_: fixed bind roots, whatever the cluster mounts

**Executor-derived mount set**:
A `Mount set` computed by the workflow executor from configured inputs, outputs, and external assets rather than entered directly by the user.
_Avoid_: hand-written bind list, UI-owned mounts

**Control plane**:
The long-lived process that prepares configuration and launches workflow execution, currently the Streamlit webapp plus Snakemake invocation.
_Avoid_: the container, the pipeline job

**Stage runtime**:
The environment in which a pipeline stage actually executes on SLURM.
_Avoid_: webapp runtime, branch runtime

**Reproducible execution**:
Running the same configured stage with the same software environment and external assets so results are less sensitive to cluster-env drift.
_Avoid_: just packaged, dockerized

**Job-contained execution**:
A deployment shape where the `Stage runtime` is containerized but the `Control plane` remains on the host cluster environment.
_Avoid_: fully containerized protforge, containerized webapp

**Stage launcher**:
The per-stage wrapper responsible for entering the container and starting the stage command with the right mounts and environment.
_Avoid_: generic sbatch wrapper, rule shell glue

**Stage image**:
The container image that carries the software environment for one stage runtime.
_Avoid_: host env path, user environment

**Input directory**:
A user-selected host directory containing the run inputs that the pipeline should process.
_Avoid_: repo, repository

**Output directory**:
A user-selected host directory where pipeline results are written.
_Avoid_: repo, workspace

**Asset bootstrap**:
An optional setup flow that downloads missing `External assets` for a user-managed environment without changing the default expectation that shared assets already exist.
_Avoid_: normal runtime, mandatory pre-run download

**Cluster-first execution**:
A support policy where the SLURM-cluster workflow is the primary target, and any non-cluster execution is secondary and adapted later.
_Avoid_: equally supported everywhere, local-first

**Native Snakemake orchestration**:
A workflow shape where Snakemake remains the scheduler-facing layer for SLURM submission, retries, and dependency handling.
_Avoid_: wrapper-owned scheduling, hidden sbatch layer

**Container-only stage runtime**:
A support contract where stage jobs are expected to run only through `Stage image` + `Stage launcher`, with no normal host-environment fallback.
_Avoid_: supported host fallback, dual-mode runtime

**Image store**:
The shared host directory where built `Stage image` files are kept for supported cluster runs.
_Avoid_: registry, random user path, ad hoc sif folder

**Stage image mapping**:
An explicit per-stage record of which `Stage image` file a stage should use.
_Avoid_: hidden filename convention, implicit image guess

**Shared stage image set**:
A cluster-managed set of `Stage image` files reused by multiple users while mounting the same shared `External assets`.
_Avoid_: per-user image rebuilds, repo image

**Per-user stage image set**:
A user-managed set of `Stage image` files built or selected for that user's own workflow runs.
_Avoid_: implicit shared image policy, repo image



**Per-stage launcher family**:
A launcher design where each stage has its own launcher while sharing common container-entry and mount-helper code.
_Avoid_: one giant generic launcher, duplicated launcher plumbing

**Chunk planning**:
The control-plane work that splits an `Input directory` into stage job units before SLURM execution.
_Avoid_: inside-image scheduling, manual sbatch planning

**Stage job unit**:
One chunked unit of stage work that Snakemake submits to SLURM and the `Stage launcher` runs inside a `Stage image`.
_Avoid_: whole pipeline run, wrapper-owned job tree

**Control-plane helper**:
A host-side script used by Snakemake for planning, path collection, or output organization rather than for containerized stage execution.
_Avoid_: compute stage launcher, inside-image runner

**Core stage path**:
The first supported container-only workflow scope, consisting of MSA, Boltz, and ESM.
_Avoid_: all stages, ES-inclusive baseline

**Path-preserving mount**:
A mount policy where a host path is exposed inside the container at the same absolute path rather than being rewritten to a container-specific location.
_Avoid_: copied-into-image asset, translated container path

**Two-phase validation**:
A validation policy where setup checks default image and asset paths early, and launch performs a final preflight before submitting jobs.
_Avoid_: setup-only trust, launch-only discovery

**Per-user storage root**:
A default user-chosen base directory for user-managed ProtForge storage, with specific paths allowed to override it when needed.
_Avoid_: mandatory single storage location, unrelated output root

**Setup asset mode**:
The user-chosen setup mode that determines whether ProtForge reuses shared cluster assets or bootstraps user-managed assets.
_Avoid_: implicit asset policy, one-size-fits-all setup

**Selective asset bootstrap**:
A bootstrap policy where user-managed model downloads can be chosen asset-by-asset instead of requiring every model for every enabled stage.
_Avoid_: forced full model download, all-assets bootstrap

**Stage-image bootstrap default**:
A setup policy where the enabled core-stage images are built by default unless the user explicitly chooses a narrower image build scope.
_Avoid_: implicit partial image set, mandatory manual image selection

**Stage-derived image build scope**:
A build-scope default where setup chooses stage images to build from the enabled pipeline stages unless the user explicitly overrides that scope.
_Avoid_: unrelated build scope, mandatory manual stage list

**Advanced runtime override**:
A non-default override path exposed for testing or debugging stage-image and asset-path settings without making them part of the normal user flow.
_Avoid_: normal required input, always-visible container plumbing

**Singularity-native image recipe**:
A container recipe authored directly for Singularity/Apptainer builds on the cluster rather than derived from Docker as the primary source of truth.
_Avoid_: Docker-first canonical recipe, implicit conversion pipeline

**Normal-user cluster build**:
A build path that a regular cluster user can execute without administrator intervention in order to produce their own `Stage image` files.
_Avoid_: admin-only build, privileged-only image workflow

**Online-capable bootstrap**:
A first-cut build/bootstrap path that may use cluster internet access when available rather than requiring a fully pre-staged offline workflow.
_Avoid_: offline-only assumption, guaranteed airgapped bootstrap

**Interactive stage resolution**:
A setup behavior that asks the user how to proceed when a selected stage lacks required images or assets, instead of silently changing the pipeline contract.
_Avoid_: silent stage disable, unexplained hard stop

**Base setup config**:
The setup-generated default configuration that captures durable environment choices and seeds runnable session configs.
_Avoid_: final per-run config, ad hoc session edits

**Session run config**:
The per-session runnable configuration that adds or overrides run-specific choices such as input paths, output paths, and scheduler settings.
_Avoid_: shared setup default, global cluster policy

**Stage-toggle default**:
A setup-provided default stage selection that sessions may override for individual runs.
_Avoid_: fixed global stage set, run-owned stage discovery

**CLI-only provisioning**:
A first-cut boundary where image building and asset bootstrap are performed through command-line setup/build flows rather than from the webapp.
_Avoid_: provisioning from run UI, mixed implicit provisioning path

**Interactive provisioning shell**:
A normal-path CLI flow that asks guided setup questions while still relying on separate commands underneath for advanced or procedural provisioning tasks.
_Avoid_: single opaque mega-script, advanced-only command maze

**Container-native config block**:
A new canonical configuration area dedicated to the container-first execution contract rather than reusing the legacy `containers:` block.
_Avoid_: legacy dual-mode config reuse, optional-container semantics

**Layered config contract**:
A configuration approach where setup-owned metadata may exist in the base config, while the runnable session config keeps only execution-relevant container settings.
_Avoid_: one flat config for every concern, provisioning-only data in run-time state

**Hybrid runtime catalog**:
A container-native runtime shape with shared image and asset catalogs plus explicit per-stage runtime entries that tie those shared definitions together.
_Avoid_: fully duplicated stage config, fully indirect global-only runtime config





































## Flagged ambiguities

- **container2** previously sounded like a runtime shape. In this context it means only the development branch name.
- **db** is too vague here. Use **External asset** when you mean MMseqs/ColabFold databases or model-weight directories.
- **Fully dynamic runtime mounts** are in scope here, but they still require pre-run validation of path existence and suitability.
- The canonical source of truth for runtime binds is the **Executor-derived mount set**, not a user-authored bind list.
- “it runs in a container” is ambiguous. Distinguish the **Control plane** from the **Stage runtime**.
- The current preferred direction is **Job-contained execution**: containerize stage jobs, keep the control plane outside.
- The preferred container boundary lives in the **Stage launcher**, not directly in every rule shell block.
- In `container2`, host `env_path` values do not belong to the containerized stage-runtime contract; that software must live in the **Stage image**.
- **repo** is ambiguous here. Use **Input directory** when you mean the host path containing FASTA or YAML inputs.
- `External assets` are cluster-default by default, but advanced users may override them or populate them through `Asset bootstrap`.
- The current support policy is **Cluster-first execution**: make the SLURM path correct first, then adapt for other systems.
- `container2` keeps **Native Snakemake orchestration**; the `Stage launcher` owns container entry, not SLURM submission.
- `container2` uses a **Container-only stage runtime** for supported stage execution.
- `container2` expects setup-managed `Stage image` defaults from a shared `Image store`.
- The setup writes a **Stage image mapping** explicitly per stage rather than relying on one inferred image-directory convention.
- `container2` uses a **Per-stage launcher family** rather than one generic launcher.
- `container2` keeps `Chunk planning` in Snakemake, which creates `Stage job unit`s that are then executed inside stage images.
- `container2` keeps `Control-plane helper`s outside the images when they are only part of Snakemake-side planning or organization.
- The initial `container2` delivery targets the **Core stage path**; ES is out of scope for the first cut.
- For the first cut, `container2` uses **Path-preserving mount**s for `External assets`, inputs, and outputs instead of translating paths inside the container.
- `container2` uses **Two-phase validation** for stage images and mounted `External assets`.
- **env** is too vague here. Use **Stage image** for packaged software, or say host machine / cluster environment explicitly.
- **repo-image** is ambiguous. Use **Stage image** for the container itself, and **Shared stage image set** when the same built images are reused across users.
- Image sharing policy is not fixed for the first cut. `container2` must work with a **Per-user stage image set** first; a **Shared stage image set** can be introduced later.
- The first-cut setup flow should support a **Per-user storage root** with overrides, rather than forcing every path to be configured independently.
- The first-cut setup flow should expose **Setup asset mode** explicitly, allowing either shared-cluster assets or user-managed bootstrap.
- In the first cut, user-managed image building should default to the enabled core stages, while model downloads follow **Selective asset bootstrap**.
- In the first cut, image building should follow a **Stage-derived image build scope** by default, with explicit override available.
- The first-cut webapp should expose **Advanced runtime override** for stage-image and asset-path tweaks, not make them part of the normal flow.
- The first-cut image build path should use a **Singularity-native image recipe** as the canonical source of truth.
- `container2` requires **Normal-user cluster build** for its first-cut image workflow.
- The first-cut normal-user image workflow may be **Online-capable bootstrap** rather than offline-only.
- The first-cut setup should use **Interactive stage resolution** when a selected stage cannot yet run.
- The first cut should use a **Base setup config** plus derived **Session run config** rather than treating one repo-root config as the only runnable state.
- Use **scheduler settings** or **SLURM partition/account overrides** instead of vague “partitions from which require the computing nodes.”
- The first cut should use a **Stage-toggle default** from setup, with routine session-level overrides.
- The first cut should keep provisioning in **CLI-only provisioning** flows; the webapp consumes the resulting setup state but does not build images or bootstrap assets itself.
- The first cut should use an **Interactive provisioning shell** for normal users, while retaining separate command-level provisioning paths underneath.
- The first cut should use a **Container-native config block** instead of treating the legacy `containers:` block as canonical.
- The first cut should follow a **Layered config contract**: base setup state may retain provisioning context, but session run config should stay runtime-focused.
- The first-cut container runtime schema should use a **Hybrid runtime catalog** rather than a purely stage-centric or purely global layout.

## Example dialogue

Dev: “Should the container runtime config be fully stage-local or fully global?”

Domain expert: “Neither. Use a `Hybrid runtime catalog`: shared image and asset catalogs, plus explicit per-stage runtime entries.”