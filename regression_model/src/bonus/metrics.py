from model import estimate_price

def compute_mae(
        mileages: list[float],
        prices: list[float],
        w: float,
        b: float
) -> float:
    """
        Compute the Mean Absolute Error (MAE) between predicted and actual prices.

        Args:
            mileages: List of car mileages (features).
            prices: List of actual car prices (targets).
            w: Model weight.
            b: Model bias.
        Returns:
            float: Mean Absolute Error (MAE) value.
    """
    sum_absolute_errors: float = 0.0
    cars_nums = len(mileages)

    for mileage, actual_price in zip(mileages, prices):
        predicted_price = estimate_price(mileage, w, b)
        prediction_error = predicted_price - actual_price
        sum_absolute_errors += abs(prediction_error)

    return sum_absolute_errors / cars_nums

def compute_mse(
        mileages: list[float],
        prices: list[float],
        w: float,
        b: float
) -> float:
    """
        Compute the Mean Squared Error (MSE) between predicted and actual prices.

        Args:
            mileages: List of car mileages (features).
            prices: List of actual car prices (targets).
            w: Model weight.
            b: Model bias.
        Returns:
            float: Mean Squared Error (MSE) value.
    """
    sum_squared_errors: float = 0.0
    cars_nums = len(mileages)

    for mileage, actual_price in zip(mileages, prices):
        predicted_price = estimate_price(mileage, w, b)
        prediction_error = predicted_price - actual_price
        sum_squared_errors += prediction_error ** 2

    return sum_squared_errors / cars_nums
