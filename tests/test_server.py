"""Server CLI device selection, without loading a model or opening a socket."""

from unittest.mock import Mock

import pytest

pytest.importorskip("torch")

from jevk5 import server  # noqa: E402


@pytest.mark.parametrize(
    "argv,device,graphs",
    [
        ([], "cuda", True),
        (["--device", "cuda"], "cuda", True),
        (["--device", "cuda:1"], "cuda:1", True),
        (["--device", "cpu"], "cpu", False),
        (["--device", "cpu:0"], "cpu:0", False),
        (["--device", "mps"], "mps", False),
    ],
)
def test_main_selects_device(monkeypatch, argv, device, graphs):
    loaded = []

    def load(source, device="cuda", graphs=True):
        loaded.append((source, device, graphs))
        return object()

    http = Mock()
    monkeypatch.setattr(server, "JevK5", load)
    monkeypatch.setattr(server, "ThreadingHTTPServer", http)

    assert server.main(["--model", "local-model", "--port", "8765", *argv]) == 0
    assert loaded == [("local-model", device, graphs)]
    assert http.call_args.args[0] == ("127.0.0.1", 8765)
    http.return_value.serve_forever.assert_called_once_with()
