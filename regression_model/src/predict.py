import utils as u
from model import estimate_price

# Read model parameters from 'model.csv'
try:
    weight, bias = u.load_model_params()
except FileNotFoundError as e:
    print(f'{u.RED}Error: {e}{u.RESET}')
    raise SystemExit(1)

while True:
    try:
        mileage = float(input(f'{u.BLUE}Enter the mileage of the car in KM: {u.RESET}'))
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
        predicted_price = estimate_price(mileage, weight, bias)
        print(f'{u.GREEN}Prediction price: {predicted_price}${u.RESET}')
        break
