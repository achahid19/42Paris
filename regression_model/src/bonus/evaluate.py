from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import compute_mae, compute_mse, compute_r2
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
r2_score = compute_r2(mileages, prices, weight, bias)

print('------- Metrics (Model Evaluation) -------')
print(f'MAE: ${mean_absolute_error:.2f}')
print(f'MSE: {mean_squared_error:.2f}')
print(f'R²: {r2_score:.2f}')
