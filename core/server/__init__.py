import asyncio
import aiohttp
import logging

logging.basicConfig(level=logging.INFO)


def aio_retry(coro):
    counter: int = 3
    interval: int = 1
    increment: bool = False

    async def wrapper(*args, **kwargs):
        attempts = 0
        while attempts < counter:
            try:
                return await coro(*args, **kwargs)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                attempts += 1
                delay = interval * attempts if increment else interval
                logging.info(
                    f"Attempt {attempts} failed: {e}. Retrying in {delay} seconds..."
                )
                await asyncio.sleep(delay)
        raise Exception(f"All {counter} attempts failed.")

    return wrapper
