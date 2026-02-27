import os
import subprocess
import sys
import tempfile
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

mcp = FastMCP()

TIMEOUT = 60

# Injected at the top of every wrapper to stub out Qt so labscript loads
# without a display or PyQt5 installed. Uses only stdlib.
# A plain MagicMock is not enough: Python's import system requires __path__
# for packages to support submodule imports (e.g. qtutils.outputbox).
# _AutoMockModule auto-creates child stubs on attribute access and registers
# them in sys.modules, handling arbitrary-depth import chains.
_QT_STUB_PREAMBLE = """\
import sys as _sys
import types as _types
from unittest.mock import MagicMock as _MagicMock

_QT_ROOTS = ('PyQt5', 'qtutils', 'blacs')

class _AutoMockModule(_types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []      # marks this as a package
        self.__file__ = None    # prevents os.path.realpath() crash
        self.__spec__ = None
        self.__loader__ = None
        self.__package__ = name
    def __getattr__(self, name):
        if name == '__all__':
            return []
        # Attribute access on an already-imported stub (e.g. qtutils.qt.QtCore.*)
        return _MagicMock()
    def __call__(self, *args, **kwargs):
        return _MagicMock()
    def __iter__(self):
        return iter([])

import importlib.util as _ilu

class _QtMockFinder:
    \"\"\"MetaPathFinder that intercepts all PyQt5.* and qtutils.* imports
    before the real packages on disk are reached.\"\"\"
    def find_spec(self, fullname, path, target=None):
        if any(fullname == r or fullname.startswith(r + '.') for r in _QT_ROOTS):
            return _ilu.spec_from_loader(fullname, self, origin=f'<mock:{fullname}>')
        return None
    def create_module(self, spec):
        return _AutoMockModule(spec.name)
    def exec_module(self, module):
        pass  # nothing to execute; attributes are lazy via __getattr__

_sys.meta_path.insert(0, _QtMockFinder())
"""


@mcp.tool()
def run_labscript(
    script: Optional[str] = None,
    script_path: Optional[str] = None,
    globals: Optional[dict] = None,
) -> str:
    """Run a labscript-suite Python script and return its output.

    Labscript-suite is an open-source framework for composing and executing
    hardware-timed laboratory experiments (quantum/atomic physics, optics,
    microscopy, materials science). A labscript Python script can:

      - Define a device connection table: pseudoclocks, DAQ cards, RF synthesizers,
        digital/analog I/O, cameras, shutters, and 20+ supported hardware types
        (NI DAQmx, SpinCore PulseBlaster, NovaTech DDS, Zaber stages, etc.)
      - Schedule timed outputs using start() / stop(t) with microsecond precision:
        AnalogOut (constant, ramp, sine, exp ramp, custom waveform),
        DigitalOut (go_high/go_low), DDS (frequency/amplitude/phase),
        Shutter, AnalogIn acquisition
      - Use full Python control flow for conditional logic, loops, and parameterized
        sequences
      - Compile to an HDF5 shot file containing hardware instructions for BLACS
      - Run analysis or post-processing using lyse routines

    Globals (experiment parameters) can optionally be injected into the script's
    namespace before execution, following the runmanager convention.

    Args:
        script:      Python source code of the labscript to run (mutually exclusive
                     with script_path).
        script_path: Path to a .py labscript file on disk (mutually exclusive with
                     script).
        globals:     Optional flat dict of parameter names -> values to inject into
                     the script's global namespace (e.g. {"duration": 1.0,
                     "frequency": 2.5e6}).

    Returns:
        A string containing exit code, stdout, and stderr.
    """
    if script is not None and script_path is not None:
        raise ValueError("Provide either 'script' or 'script_path', not both.")
    if script is None and script_path is None:
        raise ValueError("Provide either 'script' or 'script_path'.")

    tmp_script = None
    tmp_wrapper = None

    try:
        # Step 1: resolve the labscript file path
        if script is not None:
            tmp_script = tempfile.NamedTemporaryFile(
                suffix=".py", mode="w", delete=False, encoding="utf-8"
            )
            tmp_script.write(script)
            tmp_script.flush()
            tmp_script.close()
            labscript_file = tmp_script.name
        else:
            labscript_file = os.path.abspath(script_path)
            if not os.path.isfile(labscript_file):
                raise ValueError(f"script_path does not exist: {labscript_file}")

        # Step 2: always build a wrapper that injects Qt stubs, then optionally
        # injects globals, then exec()s the target script.
        globals_block = ""
        if globals:
            globals_repr = repr(globals)
            globals_block = (
                f"import builtins as _builtins\n"
                f"for _k, _v in {globals_repr}.items():\n"
                f"    setattr(_builtins, _k, _v)\n"
            )

        wrapper_code = (
            _QT_STUB_PREAMBLE
            + globals_block
            + f"exec(open({repr(labscript_file)}).read(), {{'__file__': {repr(labscript_file)}}})\n"
        )

        tmp_wrapper = tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False, encoding="utf-8"
        )
        tmp_wrapper.write(wrapper_code)
        tmp_wrapper.flush()
        tmp_wrapper.close()

        # Step 3: execute
        try:
            result = subprocess.run(
                [sys.executable, tmp_wrapper.name],
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return f"Exit code: -1\n--- stdout ---\n\n--- stderr ---\nExecution timed out after {TIMEOUT} seconds.\n"

        # Step 4: format output
        return (
            f"Exit code: {result.returncode}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )

    finally:
        for tmp in (tmp_script, tmp_wrapper):
            if tmp is not None:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass


def main():
    logger.info("Starting labscript MCP server")
    mcp.run("stdio")


if __name__ == "__main__":
    main()
