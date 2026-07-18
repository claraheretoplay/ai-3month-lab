import numpy as np
import matplotlib.pyplot as plt

fs = 10000
f = 1000
duration = 0.01

t = np.arange(0, duration, 1 / fs)
x = np.sin(2 * np.pi * f * t)
y = x + 0.2 * np.random.randn(len(x))

plt.plot(t, x, label="clean")
plt.plot(t, y, label="noisy", alpha=0.7)
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.savefig("sine_noise.png")
plt.show()