def estimate_price(x: float, w: float, b: float) -> float:
    """
        Linear regression function for price prediction

        Args:
            x: feature
            w: model weight
            b: model bias
        
        Returns:
            float: price predicted
    """
    return x * w + b

def compute_gradients(
    mileages: list[float], prices: list[float], weight: float, bias: float
) -> tuple[float, float]:
    """
    Compute the gradients of the loss function with respect to weight and bias.
    
    Args:
        mileages: List of car mileages (features).
        prices: List of actual car prices (targets).
        weight: Current weight of the model.
        bias: Current bias of the model.

    Returns:
        A tuple containing the gradients for weight and bias.
    """
    cars_nums = len(mileages)
    # errors vars accumulators, set to 0 in each training epoch
    sum_signed_errors: float = 0.0
    sum_weight_gradient: float = 0.0
    
    # use zip to iterate over mileages and prices together
    for mileage, price in zip(mileages, prices):
        # Predict price using linear regression equation
        price_prediction = estimate_price(x=mileage, w=weight, b=bias)

        # Compute prediction's error
        prediction_error = price_prediction - price

        sum_signed_errors += prediction_error
        sum_weight_gradient += prediction_error * mileage

    bias_gradient = sum_signed_errors / cars_nums
    weight_gradient = sum_weight_gradient / cars_nums

    return weight_gradient, bias_gradient
