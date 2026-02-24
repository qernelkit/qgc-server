# qgc-server

Implemenation of the qgc registry. For an example usage, see the [examples repository]().

## Getting Started

You should have [uv](https://docs.astral.sh/uv/) installed as your Python package manager. Once you have it, install dependencies and start a development server:

```bash
uv sync
uv run uvicorn qgc_server.main:app --host 0.0.0.0 --port 8080 --reload
```


