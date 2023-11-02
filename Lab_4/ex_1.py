import numpy as np
import time
import matplotlib.pyplot as plt

# Definiți DFT personalizată pentru N=8
def custom_dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

# Dimensiunile vectorilor N
N_values = [128, 256, 512, 1024, 2048, 4096, 8192]

custom_dft_times = []
numpy_fft_times = []

for N in N_values:
    x = np.random.random(N)  # Generați un semnal de intrare aleator
    start_time = time.time()
    custom_result = custom_dft(x)
    custom_time = time.time() - start_time
    custom_dft_times.append(custom_time)

    start_time = time.time()
    numpy_result = np.fft.fft(x)
    numpy_time = time.time() - start_time
    numpy_fft_times.append(numpy_time)

# Trasați un grafic cu timpii de execuție pe o scală logaritmică
plt.figure()
plt.semilogy(N_values, custom_dft_times, marker='o', label='Custom DFT')
plt.semilogy(N_values, numpy_fft_times, marker='x', label='numpy.fft')
plt.xlabel('Dimensiunea semnalului (N)')
plt.ylabel('Timp de execuție (secunde)')
plt.legend()
plt.title('Compararea timpilor de execuție pentru DFT personalizată și numpy.fft')
plt.grid()
plt.show()