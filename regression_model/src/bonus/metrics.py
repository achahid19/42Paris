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

def compute_r2(
        mileages: list[float],
        prices: list[float],
        w: float,
        b: float
) -> float:
    """
        Compute the R²score (coefficient of determination) between predicted and actual prices.
        formula: R² = 1 - (sum_squared_residuals / sum_squared_total)
        Args:
            mileages: List of car mileages (features).
            prices: List of actual car prices (targets).
            w: Model weight.
            b: Model bias.
        Returns:
            float: R-squared (R²) score value.
    """
    mean_price = sum(prices) / len(prices)
    sum_squared_total: float = 0.0
    sum_squared_residuals: float = 0.0

    for mileage, actual_price in zip(mileages, prices):
        predicted_price = estimate_price(mileage, w, b)
        sum_squared_total += (actual_price - mean_price) ** 2
        sum_squared_residuals += (actual_price - predicted_price) ** 2

    return 1 - (sum_squared_residuals / sum_squared_total)
