import numpy as np
def generate_sine(fs,f,duration):
    t=np.arange(0,duration,1/fs)
    x=np.sin(2*np.pi*f*t)
    return t,x

def generate_bits(n):
    return np.random.randint(0,2,size=n)

def bpsk_modulate(bits):
    return 2*bits-1

def add_awgn(signal,noise_std):
    noise=noise_std*np.random.randn(len(signal))
    return signal+noise
