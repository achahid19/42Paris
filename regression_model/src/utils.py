import csv
from pathlib import Path

RED = '\033[31m'
GREEN = '\033[32m'
BLUE  = "\033[34m"
RESET = '\033[0m'


def load_data(file_path: str) -> tuple[list[float], list[float]]:
    """Load car mileages and prices from a CSV file.

    Args:
        file_path: Path to the CSV dataset.

    Returns:
        A tuple containing mileages and prices as float lists.

    Raises:
        ValueError: If the dataset is empty or contains invalid numbers.
        FileNotFoundError: If the dataset file does not exist.
    """
    col1: list[float] = []
    col2: list[float] = []

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        dataset = csv.reader(csvfile, delimiter=',')
        next(dataset)
        for row in dataset:
            col1.append(float(row[0]))
            col2.append(float(row[1]))

    if not col1 or not col2:
        raise ValueError("Dataset is empty")

    return (col1, col2)
