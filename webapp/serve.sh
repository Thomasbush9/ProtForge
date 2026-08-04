#!/usr/bin/env bash
# Start (or reattach to) the ProtForge web UI on this node.
#
# Login nodes are shared, so a hard-coded 8501 collides with whoever claimed it
# first. This picks a free port out of PROTFORGE_PORT_RANGE, records it per user
# and per node, and reattaches to an already-running app instead of starting a
# second one.
#
# Usage:
#   bash webapp/serve.sh                  # start or reattach; print the tunnel command
#   bash webapp/serve.sh --print-port     # stdout is the port alone (for scripts)
#   bash webapp/serve.sh --print-endpoint # stdout is "<port> <fqdn>"
#   bash webapp/serve.sh --pick-port      # print a free port, start nothing
#   bash webapp/serve.sh --status         # report what is running, if anything
#   bash webapp/serve.sh --stop           # stop this node's app
#
# Environment:
#   PROTFORGE_PORT        force a specific port (e.g. $port under Open OnDemand)
#   PROTFORGE_PORT_RANGE  candidate range, default 15000-20000
#   PROTFORGE_STATE_DIR   where the port/PID record lives, default ~/.protforge
#   PROTFORGE_STREAMLIT   path to the streamlit executable
#   PROTFORGE_STARTUP_TIMEOUT  seconds to wait for the app to bind, default 120

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${PROTFORGE_STATE_DIR:-$HOME/.protforge}"
NODE="$(hostname -s)"
FQDN="$(hostname -f 2>/dev/null || hostname)"
STATE_FILE="${STATE_DIR}/webapp.${NODE}.state"
LOG_FILE="${STATE_DIR}/webapp.${NODE}.log"

# Seconds to wait for Streamlit to bind. A first start off shared storage is
# slow (cold imports over NFS), so this is generous on purpose.
STARTUP_TIMEOUT="${PROTFORGE_STARTUP_TIMEOUT:-120}"

PORT_RANGE="${PROTFORGE_PORT_RANGE:-15000-20000}"
RANGE_LO="${PORT_RANGE%%-*}"
RANGE_HI="${PORT_RANGE##*-}"

MODE="run"
case "${1:-}" in
    "") ;;
    --print-port|--print-endpoint|--pick-port|--status|--stop) MODE="${1#--}" ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "serve.sh: unknown argument '$1' (try --help)" >&2; exit 2 ;;
esac

# Human-facing chatter goes to stderr so --print-port's stdout stays parseable.
say() { echo "$@" >&2; }

# --- Port selection ----------------------------------------------------------

# Local ports currently bound by anyone on this node. Listening sockets are what
# actually block a bind, but established connections are included too: their
# local port is taken until the socket closes.
ports_in_use() {
    ss -Htan 2>/dev/null | awk '{n = split($4, a, ":"); print a[n]}' | grep -E '^[0-9]+$' | sort -u
}

# True when nothing accepts a connection on the port. Second opinion on top of
# ports_in_use, and the only check available if `ss` is missing.
port_free() {
    ! timeout 1 bash -c "exec 3<>/dev/tcp/127.0.0.1/$1" 2>/dev/null
}

pick_port() {
    local used candidate
    used="$(ports_in_use || true)"
    while read -r candidate; do
        if port_free "$candidate"; then
            echo "$candidate"
            return 0
        fi
    done < <(seq "$RANGE_LO" "$RANGE_HI" \
             | awk 'NR == FNR { used[$1]; next } !($1 in used)' <(printf '%s\n' "$used") - \
             | shuf -n 50)
    say "serve.sh: no free port found in ${PORT_RANGE}."
    say "          Set PROTFORGE_PORT_RANGE to a different range and retry."
    return 1
}

resolve_port() {
    if [[ -n "${PROTFORGE_PORT:-}" ]]; then
        echo "$PROTFORGE_PORT"
        return 0
    fi
    pick_port
}

# --- Existing instance -------------------------------------------------------

# Sets RUNNING_PORT / RUNNING_PID when this node still hosts a live app of ours.
find_running() {
    RUNNING_PORT=""
    RUNNING_PID=""
    [[ -f "$STATE_FILE" ]] || return 1

    local port pid
    port="$(awk -F= '$1 == "PORT" {print $2}' "$STATE_FILE")"
    pid="$(awk -F= '$1 == "PID" {print $2}' "$STATE_FILE")"
    [[ -n "$port" && -n "$pid" ]] || return 1

    # PIDs get recycled, so confirm this one is still our Streamlit before
    # trusting the record.
    kill -0 "$pid" 2>/dev/null || return 1
    tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | grep -q streamlit || return 1
    port_free "$port" && return 1

    RUNNING_PORT="$port"
    RUNNING_PID="$pid"
    return 0
}

resolve_streamlit() {
    local candidate
    for candidate in \
        "${PROTFORGE_STREAMLIT:-}" \
        "$(command -v streamlit 2>/dev/null || true)" \
        "${PROTFORGE_ASSETS:-}/envs/host/bin/streamlit" \
        "${PROTFORGE_ROOT:-}/envs/host/bin/streamlit"
    do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    say "serve.sh: no streamlit executable found."
    say "          Activate the host env first, e.g."
    say "            module load python && mamba activate \"\$PROTFORGE_ASSETS/envs/host\""
    say "          or set PROTFORGE_STREAMLIT=/path/to/streamlit."
    return 1
}

start_app() {
    local port streamlit_bin
    port="$(resolve_port)" || return 1
    streamlit_bin="$(resolve_streamlit)" || return 1

    # Streamlit launches Snakemake as a child, and the workflow's local rules
    # shell out to a bare `python`. Putting the env on PATH keeps that working
    # even when serve.sh was invoked without activating it.
    export PATH="$(dirname "$streamlit_bin"):$PATH"

    mkdir -p "$STATE_DIR"
    cd "$REPO_ROOT"
    nohup "$streamlit_bin" run webapp/app.py \
        --server.port "$port" \
        --server.address 127.0.0.1 \
        --server.headless true \
        >> "$LOG_FILE" 2>&1 &
    local pid=$!
    disown "$pid" 2>/dev/null || true

    local waited=0
    while port_free "$port"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            say "serve.sh: Streamlit exited during startup. Last lines of ${LOG_FILE}:"
            tail -n 20 "$LOG_FILE" >&2 || true
            return 1
        fi
        if (( waited >= STARTUP_TIMEOUT )); then
            # Kill it rather than leaving an untracked app squatting the port:
            # nothing recorded it, so nothing would ever reattach to or stop it.
            kill "$pid" 2>/dev/null || true
            say "serve.sh: Streamlit did not start listening on ${port} within ${STARTUP_TIMEOUT}s; stopped it."
            say "          Last lines of ${LOG_FILE}:"
            tail -n 20 "$LOG_FILE" >&2 || true
            say "          Retry, or raise PROTFORGE_STARTUP_TIMEOUT if the node is just slow."
            return 1
        fi
        sleep 1
        waited=$((waited + 1))
    done

    printf 'PORT=%s\nPID=%s\nNODE=%s\nFQDN=%s\nREPO=%s\n' \
        "$port" "$pid" "$NODE" "$FQDN" "$REPO_ROOT" > "$STATE_FILE"
    say "Started ProtForge web UI on ${NODE} port ${port} (pid ${pid}, log ${LOG_FILE})."
    echo "$port"
}

# Prints the port, starting the app only if it is not already up.
ensure_running() {
    if find_running; then
        say "ProtForge web UI already running on ${NODE} port ${RUNNING_PORT} (pid ${RUNNING_PID})."
        echo "$RUNNING_PORT"
    else
        start_app
    fi
}

# --- Modes -------------------------------------------------------------------

case "$MODE" in
    pick-port)
        resolve_port
        ;;
    print-port)
        ensure_running
        ;;
    print-endpoint)
        port="$(ensure_running)"
        echo "$port $FQDN"
        ;;
    status)
        if find_running; then
            echo "running: node=${NODE} port=${RUNNING_PORT} pid=${RUNNING_PID} log=${LOG_FILE}"
        else
            echo "not running on ${NODE}"
            exit 1
        fi
        ;;
    stop)
        if find_running; then
            kill "$RUNNING_PID"
            rm -f "$STATE_FILE"
            echo "Stopped ProtForge web UI on ${NODE} (pid ${RUNNING_PID})."
            echo "Any SLURM jobs it submitted keep running; use scancel to stop those."
        else
            echo "Nothing to stop on ${NODE}."
        fi
        ;;
    run)
        port="$(ensure_running)"
        cat >&2 <<EOF

Reach it from your laptop with:

    ssh -L ${port}:localhost:${port} ${USER}@${FQDN}
    # then open http://localhost:${port}

Or let webapp/connect.sh do both steps for you.
Stop it later with: bash webapp/serve.sh --stop
EOF
        ;;
esac
