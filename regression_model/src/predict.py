import utils as u
from pathlib import Path as p
import csv

# Model parameters initialized to zero
weight: float = 0
bias: float = 0

# Read model parameters from 'model.csv'
model_path = '../data/model.csv'
if p(model_path).exists():
    with open(model_path, newline='', encoding='utf-8') as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        next(data)
        weight, bias = map(float, next(data))

while True:
    try:
        mileage = float(input(f'{u.BLUE}Enter the mileage of the car in KM: {u.RESET}')) # feature
        if mileage < 0:
            print(f'{u.RED}Mileage cannot be negative !{u.RESET}')
            continue
    except ValueError:
        print(f'{u.RED}Please enter a valid non-negative number mileage{u.RESET}')
    except (KeyboardInterrupt, EOFError):
        print(f'{u.RED}\nProgram was interrupted by user.{u.RESET}')
        break
    else:
        # Prediction model, compute target
        predicted_price = weight * mileage + bias
        print(f'{u.GREEN}Prediction price: {predicted_price}${u.RESET}')
        break
