#!/usr/bin/env python

'''Utility functions for Phenocover'''
import os
import sys
import time
import requests
from functools import wraps
from rich import print
from pathlib import Path
from phenocover.logging import get_logger

__author__ = "Joseph Gitahi"
__email__ = "joemureithi@live.com"
__maintainer__ = "Joseph Gitahi"
__year__ = "2025"

logger = get_logger(__name__)


def clear():
    '''Clears Console'''
    os.system('cls' if os.name == 'nt' else 'clear')


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(
            f'[blue]{func.__name__} took {end - start:.6f} seconds to complete')
        return result
    return wrapper


def get_file_name(file_path: str) -> str:
    return os.path.splitext(os.path.basename(file_path))[0]


def get_file_extension(file_path: Path) -> str:
    return file_path.suffix


def get_files(input_dir: str, extensions: list) -> list:
    '''Returns a list of files with the specified extensions in the input directory'''
    files = []
    for file in os.listdir(input_dir):
        if file.endswith(tuple(extensions)):
            files.append(os.path.join(input_dir, file))
    return files


def fetch_data(url) -> dict:
    """Fetch data from an API

    Args:
        url (_type_): API URL
    Returns:
        response (_type_): API response
    """
    response = None
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f'An error occurred while fetching data: {e}')
        sys.exit(1)
    return response.json()


def fetch_sensorthingsapi(url) -> list:
    """Fetch SensorThings Paginated API endpoint

    Args:
        url (_type_): API URL

    Returns:
        json (_type_): JSON data
    """

    data = fetch_data(url)
    fetched_entities = data['value']

    while data.get('@iot.nextLink'):
        data = fetch_data(data['@iot.nextLink'])
        fetched_entities.extend(data['value'])

    return fetched_entities
