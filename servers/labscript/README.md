# Labscript MCP Server

An MCP server for running [labscript-suite](https://labscriptsuite.org/) Python experiment scripts.

## Overview

Labscript-suite is an open-source framework for composing and executing hardware-timed laboratory experiments in quantum/atomic physics, optics, microscopy, and materials science.

This server exposes a single tool, `run_labscript`, that executes any labscript-suite Python script and returns its stdout, stderr, and exit code.

## Tool: `run_labscript`

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `script` | `str` (optional) | Inline Python source code to run. Mutually exclusive with `script_path`. |
| `script_path` | `str` (optional) | Path to a `.py` labscript file on disk. Mutually exclusive with `script`. |
| `globals` | `dict` (optional) | Flat dict of parameter names → values injected into the script's `__builtins__` namespace before execution (runmanager convention). |

Exactly one of `script` or `script_path` must be provided.

## Examples

### Inline script, no globals
```python
run_labscript(script="from labscript import *\nstart()\nstop(1)")
```

### Globals injection
```python
run_labscript(
    script="print(duration)",
    globals={"duration": 1.5}
)
# stdout: 1.5
```

### Script file
```python
run_labscript(script_path="/path/to/experiment.py")
```

## Usage via launcher

```bash
uvx mcp-science labscript
```

## Local development

```bash
cd servers/labscript
pip install -e .
mcp-labscript
```
