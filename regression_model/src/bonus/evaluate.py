from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import compute_mae, compute_mse
import utils as u
from utils import load_data
from config import DATA_PATH


# Load Model params (w, b)
try:
    weight, bias = u.load_model_params()
except FileNotFoundError as e:
    print(f'{u.RED}Error: {e}{u.RESET}')
    raise SystemExit(1)

# Load dataset
try:
    mileages, prices = load_data(DATA_PATH)
except (ValueError, FileNotFoundError, StopIteration, IndexError) as e:
    print(f"{u.RED}Error loading dataset: {e}{u.RESET}")
    raise SystemExit(1)


mean_absolute_error = compute_mae(mileages, prices, weight, bias)
mean_squared_error = compute_mse(mileages, prices, weight, bias)

print('------- Metrics (Loss Functions) -------')
print(f'MAE: {mean_absolute_error}')
print(f'MSE: {mean_squared_error}')
