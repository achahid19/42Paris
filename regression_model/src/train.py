from utils import load_data, scale_feature, save_model_params
from config import DATA_PATH
from model import compute_gradients
import utils as u
from visualize import plot_data

weight: float = 0.0
bias: float = 0.0
learning_rate: float = 0.1
epochs = 1000

# Get training data
try:
    mileages, prices = load_data(DATA_PATH)
except (ValueError, FileNotFoundError, StopIteration, IndexError) as e:
    print(f"{u.RED}Error loading dataset: {e}{u.RESET}")
    raise SystemExit(1)

# Feature max scaling between [0-1]
scaled_mileages = scale_feature(mileages)

for i in range(epochs):
    weight_gradient, bias_gradient = compute_gradients(
        scaled_mileages, prices, weight, bias
    )
    # gradient descent update
    bias = bias - (learning_rate * bias_gradient)
    weight = weight - (learning_rate * weight_gradient)

# unscale weight and save model's parameters
max_mileage = max(mileages)
original_weight = weight / max_mileage
save_model_params(original_weight, bias)

print('++++++++++++++++++++++++++++++++++++++++++++++')
print(f'model bias: {bias}')
print(f'model weight: {original_weight}')

# visualize data
plot_data(mileages, prices, original_weight, bias)
