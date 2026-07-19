import numpy as np

bits=np.random.randint(0,2,1000)#相当于generate_bits

symbols=2*bits-1#相当于bpsk_modulate

noise_std=0.3
noise=noise_std*np.random.randn(1000)
received=symbols+noise#相当于add_awgn

detected_bits=(received>0).astype(int)

print("判决比特",detected_bits)
