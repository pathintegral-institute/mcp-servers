
import logging
import os
from typing import Any, Callable, List, Optional, cast

import re
import subprocess
from mcp.server.lowlevel import Server as McpServer
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, ErrorData, TextContent, Tool
from pydantic import BaseModel, ValidationError
from python_code_execution.schemas import BASE_BUILTIN_MODULES
logger = logging.getLogger(__name__)
class PythonCodeExecutionArgs(BaseModel):
    code: str

# General Search Function
async def python_code_execution(code: str) -> list[TextContent]:
    """Execute the generated python code in a sandboxed environment.

    This tool allows you to run Python code with certain restrictions for security.

    IMPORTANT: Always use print() to show your results! Any values that aren't printed
    will not be returned to the conversation.

    Allowed imports (standard library only):
    {}

    Limitations:
    - No file system access, network operations, or system calls
    - Limited computation time and memory usage
    - No dynamic code execution (no eval, exec, etc.)
    - Custom imports beyond the allowed list will fail

    Examples:

    Basic calculations and printing:
    ```python
    x = 10
    y = 20
    result = x * y
    print(f"The result is {result}")
    ```

    Working with lists and functions:
    ```python
    def square(n):
        return n * n

    numbers = [1, 2, 3, 4, 5]
    squared = [square(n) for n in numbers]
    print(f"Original: {numbers}")
    print(f"Squared: {squared}")
    ```

    Data analysis with built-in tools:
    ```python
    import statistics

    data = [12, 15, 18, 22, 13, 17, 16]
    mean = statistics.mean(data)
    median = statistics.median(data)
    print(f"Mean: {mean}, Median: {median}")
    ```
    """.format("\n".join(f"- {module}" for module in BASE_BUILTIN_MODULES))
    # Clean the code by removing markdown code blocks if present
    cleaned_code = re.sub(r'```(?:python|py)?\s*\n|```\s*$', '', code)
    
    # Run the code evaluation by calling safe_execute.py with a subprocess
    try:
        # Construct the command with proper escaping
        cmd = [
            "uv",
            "run",
            "safe-execute",
            "--code", cleaned_code
        ]

        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=20
        )

        # Get the output
        if process.returncode == 0:
            result = process.stdout
        else:
            result = process.stdout
            if process.stderr:
                result += f"\nError: {process.stderr}"

    except subprocess.TimeoutExpired:
        result = "Execution timed out. The code took too long to run."
    except Exception as e:
        result = f"An error occurred while executing the code: {str(e)}"

    return [TextContent(
        text=result,
        type="text",
    )]

python_code_execution_tool = Tool(
    name="python_code_execution",
    description=python_code_execution.__doc__,
    inputSchema=PythonCodeExecutionArgs.model_json_schema()
)

async def serve():
    server = McpServer(name="mcp-python_code_execution")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [python_code_execution_tool]

    @server.call_tool()
    async def call_tool(tool_name: str, arguments: dict[str, Any]) -> list[TextContent]:
        try:
            args = PythonCodeExecutionArgs(**arguments)
        except ValidationError as e:
            raise McpError(ErrorData(
                code=INTERNAL_ERROR,
                message=f"Invalid arguments for {tool_name}: {str(e)}"
            ))
        match tool_name:
            case python_code_execution_tool.name:
                return await python_code_execution(args.code)
            case _:
                raise McpError(ErrorData(
                    code=INTERNAL_ERROR,
                    message=f"Invalid tool name: {tool_name}"
                ))


    async with stdio_server() as (read_stream, write_stream):
        logger.info("Starting LocalPython Code Execution Server...")
        await server.run(
            read_stream,
            write_stream,
            initialization_options=server.create_initialization_options(),
            raise_exceptions=False
        )
