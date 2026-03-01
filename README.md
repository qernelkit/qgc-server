# qgc-server

Implemenation of the qgc registry. For an example usage, see the [examples repository]().

## Getting Started

You should have [uv](https://docs.astral.sh/uv/) installed as your Python package manager. Once you have it, install dependencies and start a development server:

```bash
uv sync
uv run uvicorn qgc_server.main:app --host 0.0.0.0 --port 8080 --reload
```

### TODO in README
- Discuss how parsing works with mineru and thank the repo owners
- Discuss how qgc works
  - Includes versioning and all of that with the quantum processor, etc. Should tell people what it's about
- Discuss the integration with Claude Desktop and general mcp servers, with a link to an example prompt and response done by Claude
- Discuss contributing, and how work that enables AI to work better and faster will be prioritized over "AI-competetive" work.
