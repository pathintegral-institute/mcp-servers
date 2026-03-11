import os
import logging
import requests
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

logger = logging.getLogger(__name__)

# Create the MCP server
mcp = FastMCP()

# Global cache for search results
_search_cache: Dict[str, List[Dict[str, Any]]] = {}

NEMAD_BASE_URL = os.getenv("NEMAD_BASE_URL", "https://api.nemad.org/api")

def get_api_key() -> str:
    """Get API key from environment variable"""
    api_key = os.getenv("NEMAD_API_KEY")
    if not api_key:
        raise ValueError("NEMAD_API_KEY environment variable is required")
    return api_key


def make_request(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Make authenticated request to NEMAD API"""
    headers = {"X-API-Key": get_api_key(), "accept": "application/json"}

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()


@mcp.tool()
def nemad_search(
    database_type: str, elements: str, exact_match: bool = False, limit: int = -1
) -> TextContent:
    """Search for materials by elements in NEMAD database.

    Args:
        database_type: Type of database (magnetic, thermoelectric, or superconductor)
        elements: Comma-separated list of elements (e.g., "Fe,Co,Ni")
        exact_match: If true, return materials with exactly these elements only
        limit: Number of results to return (-1 for all results)

    Returns:
        Search results with count and material information
    """
    try:
        # Validate database type
        valid_db_types = ["magnetic", "thermoelectric", "superconductor"]
        if database_type not in valid_db_types:
            return TextContent(
                type="text",
                text=f"Error: database_type must be one of {valid_db_types}",
            )

        # Construct URL and parameters
        url = f"{NEMAD_BASE_URL}/{database_type}/search"
        params = {"elements": elements, "exact_match": exact_match, "limit": limit}

        # Make request
        data = make_request(url, params)

        # Cache the results for later reading
        cache_key = f"{database_type}_{elements}_{exact_match}_{limit}"
        _search_cache[cache_key] = data.get("results", [])

        # Format response
        count = data.get("count", 0)
        results = data.get("results", [])

        response_text = "NEMAD Search Results\n"
        response_text += f"Database: {database_type}\n"
        response_text += f"Elements: {elements}\n"
        response_text += f"Exact match: {exact_match}\n"
        response_text += f"Total materials found: {count}\n"
        response_text += f"Results returned: {len(results)}\n"
        response_text += f"Cache key: {cache_key}\n\n"
        response_text += f"Use nemad_read_results with cache_key='{cache_key}' to read specific ranges of results."

        return TextContent(type="text", text=response_text)

    except Exception as e:
        logger.error(f"Error in nemad_search: {e}")
        return TextContent(type="text", text=f"Error: {str(e)}")


@mcp.tool()
def nemad_formula_search(
    database_type: str, formula: str, limit: int = -1
) -> TextContent:
    """Search for materials by exact chemical formula in NEMAD database.

    Args:
        database_type: Type of database (magnetic, thermoelectric, or superconductor)
        formula: Exact chemical formula to search for (e.g., "Fe2O3")
        limit: Number of results to return (-1 for all results)

    Returns:
        Search results with count and material information
    """
    try:
        # Validate database type
        valid_db_types = ["magnetic", "thermoelectric", "superconductor"]
        if database_type not in valid_db_types:
            return TextContent(
                type="text",
                text=f"Error: database_type must be one of {valid_db_types}",
            )

        # Construct URL and parameters
        url = f"{NEMAD_BASE_URL}/{database_type}/formula"
        params = {"formula": formula, "limit": limit}

        # Make request
        data = make_request(url, params)

        # Cache the results for later reading
        cache_key = f"{database_type}_formula_{formula}_{limit}"
        _search_cache[cache_key] = data.get("results", [])

        # Format response
        count = data.get("count", 0)
        results = data.get("results", [])

        response_text = "NEMAD Formula Search Results\n"
        response_text += f"Database: {database_type}\n"
        response_text += f"Formula: {formula}\n"
        response_text += f"Total materials found: {count}\n"
        response_text += f"Results returned: {len(results)}\n"
        response_text += f"Cache key: {cache_key}\n\n"
        response_text += f"Use nemad_read_results with cache_key='{cache_key}' to read specific ranges of results."

        return TextContent(type="text", text=response_text)

    except Exception as e:
        logger.error(f"Error in nemad_formula_search: {e}")
        return TextContent(type="text", text=f"Error: {str(e)}")


@mcp.tool()
def nemad_read_results(
    cache_key: str, start_index: int = 0, end_index: Optional[int] = None
) -> TextContent:
    """Read specific range of cached search results.

    Args:
        cache_key: The cache key from previous search results
        start_index: Starting index (0-based, inclusive)
        end_index: Ending index (0-based, exclusive). If None, read to end.

    Returns:
        Detailed information for materials in the specified range
    """
    try:
        # Check if cache key exists
        if cache_key not in _search_cache:
            available_keys = list(_search_cache.keys())
            return TextContent(
                type="text",
                text=f"Error: Cache key '{cache_key}' not found.\nAvailable keys: {available_keys}",
            )

        results = _search_cache[cache_key]
        total_count = len(results)

        # Validate indices
        if start_index < 0:
            start_index = 0
        if start_index >= total_count:
            return TextContent(
                type="text",
                text=f"Error: start_index {start_index} is out of range. Total results: {total_count}",
            )

        if end_index is None:
            end_index = total_count
        else:
            end_index = min(end_index, total_count)

        if end_index <= start_index:
            return TextContent(
                type="text",
                text=f"Error: end_index {end_index} must be greater than start_index {start_index}",
            )

        # Extract the requested range
        selected_results = results[start_index:end_index]

        # Format response
        response_text = f"NEMAD Results (Range {start_index} to {end_index - 1})\n"
        response_text += f"Cache key: {cache_key}\n"
        response_text += f"Total available results: {total_count}\n"
        response_text += f"Showing results: {len(selected_results)}\n\n"

        for i, material in enumerate(selected_results):
            actual_index = start_index + i
            response_text += f"Result #{actual_index + 1}:\n"
            response_text += (
                f"Material Name: {material.get('Material_Name', 'Unknown')}\n"
            )

            for key, value in material.items():
                if key != "Material_Name":
                    response_text += f"{key}: {value}\n"
            response_text += "\n" + "-" * 50 + "\n\n"

        return TextContent(type="text", text=response_text)

    except Exception as e:
        logger.error(f"Error in nemad_read_results: {e}")
        return TextContent(type="text", text=f"Error: {str(e)}")


@mcp.tool()
def nemad_list_cache() -> TextContent:
    """List all available cached search results.

    Returns:
        List of all cache keys with basic information
    """
    if not _search_cache:
        return TextContent(
            type="text",
            text="No cached search results available. Perform a search first.",
        )

    response_text = "Available Cached Search Results:\n\n"
    for cache_key, results in _search_cache.items():
        response_text += f"Cache Key: {cache_key}\n"
        response_text += f"Results Count: {len(results)}\n"
        response_text += "-" * 30 + "\n"

    return TextContent(type="text", text=response_text)


def main(transport: str = "stdio"):
    """Main entry point for the NEMAD MCP server"""
    logger.info("Starting NEMAD MCP server")
    mcp.run(transport)
