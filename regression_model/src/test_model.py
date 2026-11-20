from model import estimate_price

actual = estimate_price(100, -2, 1000)
expected = 800

assert actual == expected, (f"Expected {expected}, got {actual}")

print("All tests passed!")