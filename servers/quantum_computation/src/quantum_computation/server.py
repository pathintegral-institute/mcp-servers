# ssh to server and run gpaw calculation. Send log and calculation status back to local. When finished, return figure.
import base64
import json
import os
import shlex
import uuid
from typing import Literal, Optional, TypedDict, cast, Annotated

import plotly.io as pio
from pydantic.networks import AnyUrl
from loguru import logger
from mcp.server.fastmcp import FastMCP
from mcp.types import EmbeddedResource, ImageContent, TextContent, TextResourceContents

from .structure_helper import get_structure_info

mcp = FastMCP("dft_computation")

remote_calculation_output_abs_path = "/home/Geoff/soan-quantum2/working/qm_sci/users/xlqi_cd"
remote_calculation_script_abs_path = "/home/Geoff/soan-quantum2/apps/qm_sci/workflows/main_create_calculation_project.py"
remote_python_path = "/home/Geoff/.cache/pypoetry/virtualenvs/soan-quantum-5rZrSzAK-py3.12/bin/python"


def execute_command_on_server(command: str):
    return None, None, None


def extract_project_id_from_uri(project_uri: str) -> str:
    """Extract the project ID from a project URI.

    Args:
        project_uri: The project URI to extract the project ID from

    Returns:
        bool: True if the URI has the correct format and contains a valid UUID
    """
    import uuid

    if not project_uri.startswith("project://"):
        raise ValueError("Project URI must start with 'project://'")

    try:
        # Extract the UUID part and validate it
        uuid_str = project_uri.split("://")[1]
        uuid.UUID(uuid_str)
        # Ensure string representation matches original (prevents invalid formats that uuid.UUID corrects)
        return uuid_str
    except (ValueError, IndexError):
        raise ValueError("Invalid UUID in project URI")


class RelaxCalculationParams(TypedDict):
    """Parameters for a gpaw relax calculation.

    Attributes:
        calculation_type: must be 'relax'
        spinpol: The boolean parameter determines whether the calculation is spin polarized.
        initial_magnetic_moments: An optional list that sets the initial magnetic moments for the atoms. This parameter is only needed if spinpol is True.
        kpoints: A list of 3 positive integers, representing the number of k-points in each direction of the reciprocal space. For example [1,1,1] or [4,4,2]
        is_bulk: The boolean parameter determines whether the calculation is for a bulk structure (periodic in all three directions) or a slab structure (periodic in two directions and open boundary condition in the third direction
        relax_intervals: A list of lists, such as [[z_1,z_2]] or [[z_1,z_2],[z_3,z_4]] which specifies intervals of z coordinates (in unit of angstrom) for relaxing atoms. If not selected, all atoms will be relaxed.
    """

    calculation_type: Literal["relax"]
    spinpol: bool
    initial_magnetic_moments: Optional[list[float]]
    kpoints: list[int]
    is_bulk: bool
    relax_intervals: Optional[list[list[float]]]


class GroundStateCalculationParams(TypedDict):
    """Parameters for a gpaw ground state calculation.

    Attributes:
        calculation_type: must be 'ground_state'
        spinpol: The boolean parameter determines whether the calculation is spin polarized.
        initial_magnetic_moments: A list that sets the initial magnetic moments for the atoms. This parameter is only needed if spinpol is True.
        kpoints: A list of 3 positive integers, representing the number of k-points in each direction of the reciprocal space. For example [1,1,1] or [4,4,2]
    """

    calculation_type: Literal["ground_state"]
    spinpol: bool
    initial_magnetic_moments: Optional[list[float]]
    kpoints: list[int]


class BandCalculationParams(TypedDict):
    """Parameters for a gpaw band structure calculation.

    Attributes:
        calculation_type: must be 'band'
        kpath: The string containing high-symmetry points that defines the k-path. For example 'GXMYG' or 'GXMYG,XG' if there are multiple segments. Use 'G' to represent the gamma point.
        npoints: The number of k-points along each segment of the k-path.
        include_soc: The parameter determines whether the calculation includes spin-orbit coupling.
    """

    calculation_type: Literal["band"]
    kpath: str
    npoints: int
    include_soc: Optional[bool]


@mcp.tool(name="create_new_project", description="Create a new project based on a given structure.")
async def create_new_project(
    structure_uri: Annotated[str, "structure_uri"],
    structure_type: Annotated[Literal["bulk", "supercell"], "structure_type, bulk or supercell"],
) -> TextContent:
    """
    Create a new project based on a given structure.

    Args:
        structure_uri: the uri of the structure with the format 'structure://{structure_id}
        structure_type: bulk (bulk crystal structure) or supercell (a thin film or supercell structure)

    Returns:
        The TextContext that contains the project id and a list of calculation info. The
        list of calculation info should contain only one item that is the initial structure information.
    """
    structure_info = get_structure_info(structure_uri)
    if structure_info is None:
        response = TextContent(type="text", text="Error:Structure not found")
        return response
    if structure_type == "bulk":
        structure = structure_info.structure
    elif structure_type == "supercell":
        structure = structure_info.structure
        if not structure:
            response = TextContent(
                type="text", text="Error: Supercell structure is not found.")
            return response
    structure_str = structure.to(fmt="json")

    project_id = str(uuid.uuid4())

    project_folder_path = os.path.join(
        remote_calculation_output_abs_path, project_id)

    python_command = f"{remote_python_path} {remote_calculation_script_abs_path} -p {project_folder_path} -s {shlex.quote(structure_str)}"
    # execute_command_on_server(python_command)
    # TODO: execute command on server
    result = json.loads("{}")
    result["project_uri"] = "project://" + project_id
    return TextContent(type="text", text=json.dumps(result))


@mcp.tool(name="start_calculation", description="Start a DFT calculation.")
async def start_calculation(
    params: Annotated[RelaxCalculationParams | GroundStateCalculationParams | BandCalculationParams, "params"],
    project_uri: str
) -> TextContent:
    """
    SSH to a computation server and start a DFT computation. There are three types of computation,
    1. 'relax': Relaxation of the structure
    2. 'ground_state': Self-consistent calculation of the charge density
    3. 'band': Calculation of the electron band structure, which requires a pre-calculated charge density,
    so it can only be carried if at least one relax or ground state calculation has been finished.

    Args:
        params: Parameters for the computation. Note that different types of computation requires different parameter fields.
        project_uri: The uri of the project. A project is defined by a structure. When the structure being studied
            is changed, one needs to start a new project

    Returns:
        dict[str, str]: A dictionary with the keys 'project_folder_path' and 'calculation_id'.
            - project_folder_path: The path of the project folder on the computation server.
            - calculation_id: The id of the new calculation.
    """
    logger.info("start calculation...")
    try:
        project_id = extract_project_id_from_uri(project_uri)
    except ValueError as e:
        return TextContent(type="text", text=f"Error: {str(e)}")

    output_calculation_id = str(uuid.uuid4())
    params_str = json.dumps(params)
    project_folder_path = os.path.join(
        remote_calculation_output_abs_path, project_id)

    python_command = (
        "/usr/bin/mpirun -np 4 /home/Geoff/.cache/pypoetry/virtualenvs/soan-quantum-5rZrSzAK-py3.12/bin/gpaw "
        f"{remote_python_path} {remote_calculation_script_abs_path} -- -pm {shlex.quote(params_str)} -p "
        f"{project_folder_path} -c {output_calculation_id}"
    )
    logger.info(f"Executing command: {python_command}")
    stdin, stdout, stderr = execute_command_on_server(python_command)

    result = {
        "calculation_id": output_calculation_id,
        "project_id": project_id,
        # "stdout": stdout_content,
        # "stderr": stderr_content,
    }
    logger.info(f"Result: {result}")
    return TextContent(type="text", text=json.dumps(result))


@mcp.tool(name="check_calculation_result", description="Check the result of a DFT computation.")
async def check_calculation_result(
    output_calculation_id: Annotated[str, "output_calculation_id"],
    project_uri: Annotated[str, "project_uri"]
) -> list[TextContent | ImageContent | EmbeddedResource]:
    """
    SSH to a computation server and check the result of a DFT computation.

    Args:
        output_calculation_id: The id of the calculation to be checked.
        project_uri: The uri of the project. A project is defined by a structure. Each
            project contains a list of calculations.

    Returns:
        list[TextContent | ImageContent]: The list that contains the calculation_status,
        log and possibly the output figure.
            - The TextContent object contains the calculation status and log.
            - If the calculation is finished, the response will contain a
            ImageContent object that contains the figure.
    """
    try:
        project_id = extract_project_id_from_uri(project_uri)
    except ValueError as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]

    project_folder_path = os.path.join(
        remote_calculation_output_abs_path, project_id)

    python_command = (
        f"{remote_python_path} {remote_calculation_script_abs_path} -p {project_folder_path} -c {output_calculation_id}"
    )
    logger.info(f"actual python command, {python_command}")
    # stdin, stdout, stderr = execute_command_on_server(python_command)
    try:
        result = json.loads("{}")
    except Exception as e:
        logger.info(f"Error: {e}")
        # ...output stderr
        return [TextContent(type="text", text=f"Error: {e}")]

    text_result = {
        "calculation_status": result["calculation_status"],
        "log": result["log"],
    }
    response: list[TextContent | ImageContent | EmbeddedResource] = [
        TextContent(type="text", text=json.dumps(text_result, indent=4))
    ]

    if result.get("figure_str"):
        # Convert plotly figure data to actual figure
        fig = pio.from_json(result["figure_str"])
        # Convert to PNG image bytes
        img_bytes = fig.to_image(format="png", scale=1)
        # Convert to base64 string
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        response.extend([
            ImageContent(
                type="image", data=img_base64, mimeType="image/png"
            ),
            EmbeddedResource(
                type="resource",
                resource=TextResourceContents(
                    uri=AnyUrl(project_uri), text=cast(str, fig.to_json()), mimeType="application/json"
                ),
                # EmbeddedResource has enabled extra fields, so add an extra_type to indicate it's a plotly figure
                extra_type="plotly",  # type: ignore
            )
        ])
    return response


if __name__ == "__main__":
    # Initialize and run the server
    logger.info("starting quantum computation server...")
    mcp.run(transport="stdio")
