"""Server start-up: a busy port fails fast, before any API call or run folder."""
import socket
import sys

import pytest

from dashboard import server


def test_busy_port_fails_before_any_network_or_files(monkeypatch, tmp_path):
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    port = blocker.getsockname()[1]

    def no_feed(*args, **kwargs):
        raise AssertionError("Kalshi feed must not be created when the port is busy")
    monkeypatch.setattr(server, "KalshiFeed", no_feed)
    monkeypatch.setattr(server.config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(sys, "argv", ["dashboard.server", "--port", str(port), "--no-ppo"])
    try:
        with pytest.raises(SystemExit) as exc:
            server.main()
    finally:
        blocker.close()
    assert f"Port {port} is already in use" in str(exc.value)
    assert not (tmp_path / "runs").exists()
