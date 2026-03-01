# qgc-server

A versioned registry and compiler for quantum circuit gadgets. Store, search, compose, and analyze reusable OpenQASM 3 building blocks, from Bell pairs to multi-qubit adders, through a REST API or directly from an AI assistant via MCP.

## What is QGC?

Quantum circuits are built from repeated patterns: entanglement generation, phase estimation, Fourier transforms, arithmetic blocks, etc. QGC treats these patterns as **gadgets**: versioned, hashed, composable units with structured manifests. The goal of this project is to make it easy to compose high level quantum algorithms, from the bottom up, with smart gadgets. 

Each gadget has **OpenQASM 3 source**, **a manifest**, **a SHA-256 hash**, and **a change history**, which together look like this:

```qasm
// bell-pair@1.0.0
gate bell q0, q1 {
    h q0;
    cx q0, q1;
}
```

which has a manifest:

```json
{
  "name": "bell-pair",
  "version": "1.0.0",
  "interface": { "input_qubits": 2, "output_qubits": 2 },
  "metrics_hints": { "t_count": 0, "cnot_count": 1, "depth": 2 },
  "hashes": { "sha256": "5cd47d28..." },
  "tags": ["entanglement", "basic"],
  "description": "Create an entangled Bell pair |00> + |11>"
}
```

which has a change history tracking modifications over time:

```
Change 30d070f9 on adder-4q@1.0.0
  Source: ai
  Notes: "QAOA knapsack variant with ZZ penalty terms"
  Metrics: { t_count: 18, cnot_count: 22, depth: 22 }
  Diff: +rz(gamma * 3) q[0]; ...
```

The registry ships with 8 [builtin](builtin/gadgets/) gadgets. New gadgets can be [ingested](qgc_server/routes/ingest.py) at runtime.

### Versioning

Gadgets follow [semver](https://semver.org). When you modify a gadget's QASM, the change is tracked in a **change bucket**, a log of snapshots with diffs against the original. When a change is ready, it can be promoted to a new version, preserving the full provenance chain.

```
adder-4q@1.0.0  (base)
  └── change 30d070f9  "QAOA knapsack variant"
       └── promote → adder-4q@1.1.0  (new version)
```

### Compilation

The compiler takes an OpenQASM 3 circuit with `@gadget` markers and substitutes in real gadget implementations from the registry. You can override which version of a gadget gets used, or substitute inline QASM. The compiler reports gate metrics after substitution so you can compare implementations.

```qasm
// Input: circuit with gadget markers
OPENQASM 3.0;
qubit[4] q;
@gadget("qft-4q") q[0], q[1], q[2], q[3];
@gadget("bell-pair") q[0], q[1];
```
```
// Output: compiled with metrics
// Metrics: { t_count: 0, cnot_count: 7, depth: 8 }
```

## Getting Started

QGC uses [uv](https://docs.astral.sh/uv/) as its main package manager.

```bash
cd qgc_server
uv sync
```

### MCP Server

QGC exposes its functionality as an [MCP](https://modelcontextprotocol.io/) server over STDIO, designed for use with Claude Desktop, Claude Code, or any MCP-compatible client.

```bash
uv run qgc-mcp
```

| MCP Tool | REST Equivalent | Description |
|----------|----------------|-------------|
| `search_gadgets` | `GET /gadgets` | Search the catalog by keyword |
| `get_gadget` | `GET /gadgets/{name}/{version}` | Retrieve manifest and QASM for a gadget |
| `compile_circuit` | `POST /compile` | Compile QASM with gadget substitutions |
| `resolve_dependencies` | `POST /resolve` | Resolve transitive dependency tree |
| `ingest_gadget` | `POST /ingest` | Ingest a new gadget (LLM constructs the manifest) |
| `list_changes` | `GET /gadgets/{name}/{version}/changes` | List modification history for a gadget version |
| `add_change` | `POST /gadgets/{name}/{version}/changes` | Record a QASM modification with diff and metrics |
| `promote_change` | | Promote a tracked change to a new gadget version |
| `extract_paper` | | Extract a PDF paper to markdown via MinerU (async) |
| `analyze_paper_for_gadgets` | | Match paper content against the gadget catalog |

### REST API

For any non-MCP integrations, you can use the core operations via a REST API.

```bash
uv run uvicorn qgc_server.main:app --host 0.0.0.0 --port 8080 --reload
```

## Claude Desktop Integration

Add this to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "qgc": {
      "command": "uv",
      "args": [
        "--directory", "/absolute/path/to/qgc_server",
        "run", "qgc-mcp"
      ]
    }
  }
}
```

Restart Claude Desktop. The QGC tools will appear in the tool list.

### What this enables

You can have a conversation like:

> "Extract this paper and tell me which parts I can build with QGC gadgets"
> `https://arxiv.org/pdf/2511.18377`

Claude will:
1. Call `extract_paper` to download and parse the PDF (runs MinerU in the background, polls for completion)
2. Call `analyze_paper_for_gadgets` to match the paper's quantum concepts against the catalog
3. Identify which builtin gadgets apply (e.g. adder circuits for arithmetic, QFT for transforms)
4. Flag concept gaps, things the paper describes that aren't in the catalog yet
5. Offer to build new gadgets via `ingest_gadget` and compose them via `compile_circuit`

The `ingest_gadget` tool is designed so that **Claude is the intelligence**. It reads the QASM, understands the circuit, and constructs the manifest directly. No external AI service required.

## Ollama Integration

If you don't use Claude, you can use [Ollama](https://ollama.com) as the intelligence layer instead. Ollama runs open-weight models (Llama, Mistral, Gemma, etc.) either locally on your device or via Ollama Cloud, and handles the manifest extraction when ingesting gadgets through the REST API.

### Local Ollama

Install and start Ollama on your machine, then pull a model:

```bash
ollama serve
ollama pull qwen3-coder-next
```

That's it. The server defaults to `QGC_OLLAMA_MODE=local` and will connect to `localhost:11434` with no API key needed. When you `POST /ingest` with QASM, Ollama analyzes the circuit and builds the manifest.

### Ollama Cloud

If you'd rather not run models locally, you can point at Ollama Cloud:

```bash
QGC_OLLAMA_MODE=cloud QGC_OLLAMA_API_KEY=sk-... uv run qgc-server
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QGC_OLLAMA_MODE` | `local` | `local`, `cloud`, or `off` |
| `QGC_OLLAMA_LOCAL_URL` | `http://localhost:11434/api/chat` | Local Ollama endpoint |
| `QGC_OLLAMA_CLOUD_URL` | `https://ollama.com/api/chat` | Cloud Ollama endpoint |
| `QGC_OLLAMA_API_KEY` | | Required for `cloud` mode |
| `QGC_OLLAMA_MODEL` | `qwen3-coder-next` | Model to use for extraction |

### Claude vs Ollama

Both work as the intelligence layer, just through different paths:

- **Claude** (via MCP): Claude reads the QASM itself, constructs the manifest, and calls `ingest_gadget` with a complete manifest. The tool just validates and stores.
- **Ollama** (via REST API): you `POST /ingest` with raw QASM, and Ollama analyzes the circuit to extract the manifest server-side. Useful for scripts, pipelines, or if you prefer open-weight models.

You can also set `QGC_OLLAMA_MODE=off` if you only use the MCP path and don't need Ollama at all.

## Notes on MinerU

QGC uses [MinerU](https://github.com/opendatalab/MinerU) by [OpenDataLab](https://opendatalab.com/) to convert PDF papers into structured markdown with preserved LaTeX formulas, extracted tables, and figures. MinerU is included as a dependency and runs locally with no external API calls.

The `extract_paper` tool:

1. Accepts a local PDF path or a URL (downloads the PDF server-side)
2. Runs MinerU as a background process (non-blocking, returns immediately, poll to check progress)
3. Caches results in `/tmp/qgc_papers/` so repeated calls are instant
4. Returns markdown text, table HTML content, equation LaTeX, and an inventory of extracted images

MinerU is licensed under [AGPL-3.0](https://github.com/opendatalab/MinerU/blob/master/LICENSE.md). Thanks to the OpenDataLab team for building and maintaining it.

### Performance

MinerU loads PyTorch and ML models on startup. First extraction may take 2-5 minutes on CPU. Subsequent extractions are faster as models stay cached. For better performance, a GPU (CUDA or Apple Silicon via MLX) reduces extraction to ~30 seconds per paper.

## Contributing

Contributions are more than welcome. We ask that you prioritize features that don't compete with LLMs, but make them more efficient. Build tools the AI can use, not tools that replace what it already does well.

### Development

```bash
cd qgc_server
uv sync --group dev
uv run pytest
```

Configuration uses environment variables with a `QGC_` prefix (see `qgc_server/config.py`).

## Projects that inspired this initiative

- [AlphaTensor-Quantum](https://www.nature.com/articles/s42256-025-01001-1): AI-discovered circuits for faster quantum computation, from DeepMind
- [Stim](https://github.com/quantumlib/Stim): fast simulator for quantum error correction, useful for building and testing stabilizer codes
- [TQEC](https://github.com/tqec/tqec): open source design tools for topological quantum error correcting codes
