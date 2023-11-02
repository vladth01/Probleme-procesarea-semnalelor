import numpy as np
import math
import matplotlib.pyplot as plt

# Parametrii semnalului
Fs = 1000  # Frecvența de eșantionare (Hz)
T = 1  # Durata semnalului (secunde)
N = int(T * Fs)  # Numărul de eșantioane
t = np.linspace(0, T, N, endpoint=False)  # Timpul

# Frecventele caracteristice ale componentelor sinusoidale
f1 = 5  # Hz
f2 = 10  # Hz
f3 = 20  # Hz

# Generarea semnalului compus
x = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t) + np.sin(2 * np.pi * f3 * t)

# Calculul transformatei Fourier folosind relația dată
def fourier_transform(x):
    N = len(x)
    X = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

X = fourier_transform(x)

# Calculul frecvențelor corespunzătoare transformatei Fourier
freq = np.arange(N) * Fs / N

# Afișarea semnalului și a modulului transformatei Fourier
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(t, x)
plt.xlabel('Timp (s)')
plt.ylabel('x[n]')
plt.title('Semnal în domeniul timpului')

plt.subplot(122)
plt.plot(freq, np.abs(X))
plt.xlabel('Frecvență (Hz)')
plt.ylabel('|X(ω)|')
plt.title('Modulul Transformatei Fourier')
plt.grid()
plt.tight_layout()

plt.show()