import numpy as np

fs = 10000
f = 1000
duration = 0.01

t = np.arange(0, duration, 1 / fs)
x = np.sin(2 * np.pi * f * t)

noise = 0.1 * np.random.randn(len(x))
y = x + noise

print(t[:10])
print(x[:10])
print(y[:10])