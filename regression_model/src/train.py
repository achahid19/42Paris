import csv

prices: list[float] = []
mileages: list[float] = []
# Feature scaling used for gradient descent and normalize mileages to [0, 1] range. 
# this is done to avoid large values of mileages which can lead to large gradients
# and cause the model to diverge during training
scaled_mileages: list[float] = []

weight: float = 0.0
bias: float = 0.0
learning_rate: float = 0.1
epochs = 10000
printing_interval = epochs // 10 # use '//' for integers division

# Read Data
with open('data/data.csv', newline='', encoding='utf-8') as csvfile:
    dataset = csv.reader(csvfile, delimiter=',')
    next(dataset)
    for row in dataset:
        mileages.append(float(row[0]))
        prices.append(float(row[1]))

cars_nums = len(prices)
max_mileage = max(mileages)
for m in mileages:
    scaled_mileages.append(m / max_mileage)

for i in range(epochs):
    if i > 0 and i % printing_interval == 0:
        print(f'Training start: epoch {i}')
    # errors vars accumulators, set to 0 in each training epoch
    sum_absolute_errors: float = 0.0
    sum_squared_errors: float = 0.0
    sum_signed_errors: float = 0.0
    sum_weight_gradient: float = 0.0
    
    # use zip to iterate over mileages and prices together
    for mileage, price in zip(scaled_mileages, prices):
        # Predict price using linear regression equation
        price_prediction = weight * mileage + bias

        # Compute error from the prediction
        prediction_error = price_prediction - price

        # MAE, MSE, mean signed error
        sum_absolute_errors += abs(prediction_error) # use abs to prevent from cancellation
        sum_squared_errors += pow(prediction_error, 2)
        sum_signed_errors += prediction_error

        sum_weight_gradient += prediction_error * mileage

    mean_absolute_error = sum_absolute_errors / cars_nums
    mean_squared_error = sum_squared_errors / cars_nums

    bias_gradient = sum_signed_errors / cars_nums
    weight_gradient = sum_weight_gradient / cars_nums

    if i > 0 and i % printing_interval == 0:
        print(f'w: {weight}, b: {bias}')
        print(f'Mean absolute error from predictions: {mean_absolute_error}')
        print(f'Mean squared error from predictions: {mean_squared_error}')
        print(f'bias gradient (Mean signed error) from predictions: {bias_gradient}')
        print(f'weight gradient from predictions: {weight_gradient}')
        print('--------------------------------------------------------------')

    # gradient descent update
    bias = bias - (learning_rate * bias_gradient)
    weight = weight - (learning_rate * weight_gradient)
    if i > 0 and i % printing_interval == 0:
        print(f'new bias: {bias}')
        print(f'new weight: {weight}')
        print('*****************************************************')

# model parameters after training end
print('++++++++++++++++++++++++++++++++++++++++++++++')
print(f'model bias: {bias}')
print(f'model weight: {weight / max_mileage}') # unscale the weight

# test predictions #
# scaled mileage
print(f'prediction 1: {weight * (240_000 / max_mileage) + bias}')
# unscaled weight
print(f'prediction 2: {(weight / max_mileage) * 240_000 + bias}')
print('++++++++++++++++++++++++++++++++++++++++++++++')

# Save model parameters into new csv file so it can be used by prediction program
# create if not exists, otherwise overwrite it
with open('data/model.csv', 'w', newline='', encoding='utf-8') as csvfile:
    model_writer = csv.writer(csvfile, delimiter=',')
    model_writer.writerow(['weight', 'bias'])
    model_writer.writerow([weight / max_mileage, bias])
