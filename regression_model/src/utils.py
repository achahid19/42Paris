import csv
from pathlib import Path
from config import MODEL_PATH

RED = '\033[31m'
GREEN = '\033[32m'
BLUE  = "\033[34m"
RESET = '\033[0m'

def load_model_params() -> tuple[float, float]:
    if MODEL_PATH.exists():
        with open(MODEL_PATH, newline='', encoding='utf-8') as csvfile:
            data = csv.reader(csvfile, delimiter=',')
            next(data)
            return map(float, next(data))
    
    raise FileNotFoundError('Can\'t find model\'s params file!')

def load_data(file_path: str | Path) -> tuple[list[float], list[float]]:
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

# Feature scaling used for gradient descent and normalize mileages to [0, 1] range. 
# this is done to avoid large values of mileages which can lead to large gradients
# and cause the model to diverge during training
def scale_feature(values: list[float]) -> list[float]:
    """Scale non-negative numerical values by their maximum (max scaling).

    Args:
        values: Non-empty list of non-negative numerical values.

    Returns:
        A new list containing the scaled values.
    """
    scaled_list: list[float] = []
    max_value = max(values)

    for value in values:
        scaled_list.append(value / max_value)

    return scaled_list

def save_model_params(w: float, b: float) -> None:
    """
        Save the model parameters (weight and bias) to a CSV file. 

        Args:
            w: Model weight.
            b: Model bias.
    """
    with open(MODEL_PATH, 'w', newline='', encoding='utf-8') as csvfile:
        model_writer = csv.writer(csvfile, delimiter=',')
        model_writer.writerow(['weight', 'bias'])
        model_writer.writerow([w, b])
