import numpy as np

def generate_bits(n):
    return np.random.randint(0, 2, size=n)

def bpsk_modulate(bits):
    return 2 * bits - 1

def add_noise(signal, noise_std):
    noise = noise_std * np.random.randn(len(signal))
    return signal + noise

bits = generate_bits(20)
symbols = bpsk_modulate(bits)
received = add_noise(symbols, 0.5)

print("bits:", bits)
print("symbols:", symbols)
print("received:", received)