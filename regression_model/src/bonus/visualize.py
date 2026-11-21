import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
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

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.25)
    ax.scatter(mileages, prices, label='Training data')
    line, = ax.plot(sorted_mileages, predicted_prices,
                     color='red', label='Regression line')
    ax.set(xlabel='Mileage in Km', ylabel='Price in $',
           title='Cars Mileages/Prices')
    ax.legend()

    w_range = max(abs(w) * 0.1, 0.1)
    b_range = max(abs(b) * 0.1, 0.1)
    w_slider = Slider(fig.add_axes([0.2, 0.1, 0.65, 0.03]),
                      'Weight', w - w_range, w + w_range, valinit=w)
    b_slider = Slider(fig.add_axes([0.2, 0.05, 0.65, 0.03]),
                      'Bias', b - b_range, b + b_range, valinit=b)

    def update(_value: float) -> None:
        line.set_ydata([
            estimate_price(m, w_slider.val, b_slider.val)
            for m in sorted_mileages
        ])
        fig.canvas.draw_idle()

    w_slider.on_changed(update)
    b_slider.on_changed(update)
    plt.show()