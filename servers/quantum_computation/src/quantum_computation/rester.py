import os
from mp_api.client import MPRester
from loguru import logger


def get_mp_rester():
    mp_api_key = os.getenv("MP_API_KEY")
    if not mp_api_key:
        raise ValueError("cannot find MP_API_KEY in environment variables")
    logger.info("initializing mp_rester")
    return MPRester(mp_api_key)


mp_rester = get_mp_rester()
