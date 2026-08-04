"""Tests for webapp/serve.sh's port selection.

The regression these guard: the webapp used to hard-code 8501, so the second
person to start it on a shared login node got `Address already in use`. serve.sh
now picks a port that nothing on the node is bound to.

Only `--pick-port` is exercised — it is the whole selection path with none of
the Streamlit startup, so these stay fast and start no servers.
"""

import socket
import subprocess
from pathlib import Path

import pytest

SERVE_SH = Path(__file__).resolve().parents[2] / "webapp" / "serve.sh"


def pick_port(**env_overrides):
    """Run `serve.sh --pick-port`; return (returncode, stdout)."""
    proc = subprocess.run(
        ["bash", str(SERVE_SH), "--pick-port"],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(Path.home()), **env_overrides},
    )
    return proc.returncode, proc.stdout.strip()


@pytest.fixture
def occupied_port():
    """A port that is bound and listening for the duration of the test."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    yield sock.getsockname()[1]
    sock.close()


def test_picks_a_port_in_range():
    code, out = pick_port(PROTFORGE_PORT_RANGE="15000-20000")
    assert code == 0, out
    assert 15000 <= int(out) <= 20000


def test_picked_port_is_actually_bindable():
    code, out = pick_port(PROTFORGE_PORT_RANGE="15000-20000")
    assert code == 0, out
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", int(out)))  # raises OSError if it was taken


def test_skips_an_occupied_port(occupied_port):
    # A narrow range anchored on the held port: whatever comes back, it is not
    # the one we are sitting on.
    code, out = pick_port(PROTFORGE_PORT_RANGE=f"{occupied_port}-{occupied_port + 20}")
    assert code == 0, out
    assert int(out) != occupied_port
    assert occupied_port < int(out) <= occupied_port + 20


def test_fails_loudly_when_the_range_is_exhausted(occupied_port):
    code, out = pick_port(PROTFORGE_PORT_RANGE=f"{occupied_port}-{occupied_port}")
    assert code != 0, f"expected failure, got port {out}"
    assert out == "", "no port should be printed when none is free"


def test_explicit_port_wins():
    # How Open OnDemand drives it: the portal hands the app a port to use.
    code, out = pick_port(PROTFORGE_PORT="8501")
    assert code == 0, out
    assert out == "8501"


def test_two_calls_do_not_collide():
    """Selection is randomized, so concurrent users rarely land on one port."""
    picks = {pick_port(PROTFORGE_PORT_RANGE="15000-20000")[1] for _ in range(8)}
    assert len(picks) > 1, f"expected varied ports, always got {picks}"
