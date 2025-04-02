import re


def extract_blocks(content: str, keywords: list[str]):
    """Extract information in status (txt) file and output them in the format of dictionary

    Args:
        content (str): The content of the status file.
        keywords (list[str]): The keywords that will be used to extract information.

    Returns:
        dict: The initial blocks and optimized blocks.
    """

    initial_blocks = {}
    final_blocks = {}

    for keyword in keywords:
        pattern = rf'^{re.escape(keyword)}.*(?:\n(?:[ \t].*|$))*'
        blocks = re.findall(pattern, content, re.MULTILINE)
        if blocks:
            initial_blocks[keyword] = blocks[0]
            if len(blocks) > 1:
                final_blocks[keyword] = blocks[-1]

    return initial_blocks, final_blocks
