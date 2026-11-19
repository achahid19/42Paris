w: float = 2.0
gradient: float = 4.0
learning_rate: float = 0.1

# gradient descent update
w_n = w - learning_rate * gradient

print(f'old weight: {w}')
print(f'new weight: {w_n}')
