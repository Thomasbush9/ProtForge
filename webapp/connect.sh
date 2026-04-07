#!/usr/bin/env bash
# Connect to ProtForge webapp on a cluster login node.
#
# Usage:
#   bash webapp/connect.sh
#
# This script will:
#   1. Ask for your cluster login details
#   2. SSH to the cluster with port forwarding
#   3. Start the Streamlit app if not already running
#   4. Open http://localhost:8501 in your browser

set -euo pipefail

PORT="${PROTFORGE_PORT:-8501}"

# --- Gather connection details ---
read -p "Cluster username: " CLUSTER_USER
read -p "Login node (e.g. holylogin06.rc.fas.harvard.edu): " CLUSTER_HOST
read -p "ProtForge directory on cluster [~/ProtForge]: " REMOTE_DIR
REMOTE_DIR="${REMOTE_DIR:-~/ProtForge}"

REMOTE="${CLUSTER_USER}@${CLUSTER_HOST}"

echo ""
echo "Connecting to ${REMOTE}..."
echo "Remote directory: ${REMOTE_DIR}"
echo ""
echo "Once connected, open: http://localhost:${PORT}"
echo "Press Ctrl+C to disconnect (app keeps running on cluster)."
echo ""

# SSH with port forwarding + start streamlit if not running
ssh -L "${PORT}:localhost:${PORT}" "${REMOTE}" -t "
    if lsof -i :${PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo 'Streamlit already running on port ${PORT}'
    else
        echo 'Starting Streamlit...'
        cd ${REMOTE_DIR} && \
        nohup streamlit run webapp/app.py \
            --server.port ${PORT} \
            --server.address 127.0.0.1 \
            --server.headless true \
            > /dev/null 2>&1 &
        sleep 2
        echo 'Streamlit started.'
    fi
    echo 'Port forwarding active. Open http://localhost:${PORT}'
    echo ''
    exec \$SHELL -l
"
