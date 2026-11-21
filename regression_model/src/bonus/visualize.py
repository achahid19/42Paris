import matplotlib.pyplot as plt
from model import estimate_price

def plot_data(
        mileages: list[float], prices: list[float], w: float, b: float
    ) -> None:
    """
        Plot the training data and the regression line.

        Args:
            mileages: List of car mileages (features).
            prices: List of actual car prices (targets).
            w: Model weight.
            b: Model bias.
    """
    # Calculate predicted prices
    sorted_mileages = sorted(mileages)
    predicted_prices = [estimate_price(m, w, b) for m in sorted_mileages]

    plt.scatter(x=mileages, y=prices, label='Training data')
    plt.plot(sorted_mileages, predicted_prices, color='red', label='Regression line')
    plt.xlabel('Mileage in Km')
    plt.ylabel('Price in $')
    plt.title('Cars Mileages/Prices')
    plt.legend()
    plt.show()