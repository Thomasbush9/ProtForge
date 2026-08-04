#!/usr/bin/env bash
# Connect to the ProtForge webapp on a cluster login node, from your laptop.
#
# Usage:
#   bash webapp/connect.sh
#
# This script will:
#   1. Ask for your cluster login details
#   2. Start the Streamlit app there (or reattach to yours, if it is up)
#   3. Forward whatever port it got, and print the URL to open
#
# The port is chosen on the cluster, not here: login nodes are shared, so a
# fixed 8501 collides with other users. See webapp/serve.sh.

set -euo pipefail

# --- Gather connection details ---
read -p "Cluster username: " CLUSTER_USER
read -p "Login node (e.g. holylogin06.rc.fas.harvard.edu): " CLUSTER_HOST
read -p "ProtForge directory on cluster [~/ProtForge]: " REMOTE_DIR
REMOTE_DIR="${REMOTE_DIR:-~/ProtForge}"

REMOTE="${CLUSTER_USER}@${CLUSTER_HOST}"
SERVE="cd ${REMOTE_DIR} && bash webapp/serve.sh"

echo ""
echo "Connecting to ${REMOTE}..."
echo "Remote directory: ${REMOTE_DIR}"
echo ""

# Reuse one SSH connection for both steps. Login-node names round-robin, so a
# second `ssh` can land on a different node than the one running the app —
# multiplexing pins us to the node that answered the first time.
CTL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/protforge-ssh.XXXXXX")"
CTL="${CTL_DIR}/ctl"
MASTER_UP=0

cleanup() {
    if (( MASTER_UP )); then
        ssh -S "$CTL" -O exit "$REMOTE" 2>/dev/null || true
    fi
    rm -rf "$CTL_DIR"
}
trap cleanup EXIT

# stderr stays attached: this is where the password / 2FA prompt appears.
if ssh -f -N -M -S "$CTL" "$REMOTE"; then
    MASTER_UP=1
    PORT="$(ssh -S "$CTL" "$REMOTE" "bash -lc '${SERVE} --print-port'")"
    ssh -S "$CTL" -O forward -L "${PORT}:localhost:${PORT}" "$REMOTE"

    echo ""
    echo "Open: http://localhost:${PORT}"
    echo "Press Ctrl-D to disconnect (the app keeps running on the cluster)."
    echo ""
    ssh -S "$CTL" -t "$REMOTE" "cd ${REMOTE_DIR}; exec \$SHELL -l"
else
    # No connection multiplexing (older/limited SSH). Ask the cluster which node
    # and port we ended up on, then tunnel to that node by name.
    echo "Connection multiplexing unavailable; falling back to two connections." >&2
    read -r PORT NODE <<< "$(ssh "$REMOTE" "bash -lc '${SERVE} --print-endpoint'")"

    echo ""
    echo "App is on ${NODE} port ${PORT}."
    echo "Open: http://localhost:${PORT}"
    echo "Press Ctrl-D to disconnect (the app keeps running on the cluster)."
    echo ""
    ssh -L "${PORT}:localhost:${PORT}" "${CLUSTER_USER}@${NODE}" -t "cd ${REMOTE_DIR}; exec \$SHELL -l"
fi
