import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftshift, fft2

# Definim domeniul discret
n1 = np.arange(-10, 10)
n2 = np.arange(-10, 10)
N = len(n1)

# Definim functiile Xn1,n2 = sin(2pi*n1 + 3pi*n2) si Xn1,n2 = sin(4pi*n1) + cos(6pi*n2)
X1 = np.sin(2 * np.pi * n1[:, None] + 3 * np.pi * n2)
X2 = np.sin(4 * np.pi *n1[:, None]) + np.cos(6 * np.pi * n2)

# Definim functia Ym1,m2
Y = np.zeros((N, N))
Y[0, 5] = Y[0, -5] = 1
Y[5, 0] = Y[-5, 0] = 1
Y[5, 5] = Y[-5, -5] = 1

# Calculam spectrul pentru fiecare functie folosind FFT
X1_fft = np.fft.fft2(X1)
X2_fft = np.fft.fft2(X2)
Y_fft = np.fft.fft2(Y)

# Afisam imaginile si spectrul lor
plt.figure(figsize=(12, 10))

plt.subplot(331)
plt.title('Xn1,n2 = sin(2pi*n1 + 3pi*n2)')
plt.imshow(X1, cmap='viridis')
plt.colorbar()

plt.subplot(332)
plt.title('Spectru Xn1,n2 = sin(2pi*n1 + 3pi*n2)')
plt.imshow(20 * np.log10(abs(X1_fft)))
plt.colorbar()

plt.subplot(333)
plt.title('Xn1,n2 = sin(4pi*n1) + cos(6pi*n2)')
plt.imshow(X2, cmap='viridis')
plt.colorbar()

plt.subplot(334)
plt.title('Spectru pentru Xn1,n2 = sin(4pi*n1) + cos(6pi*n2)')
plt.imshow(20 * np.log10(abs(X2_fft)))
plt.colorbar()

plt.subplot(335)
plt.title('Ym1,m2')
plt.imshow(Y, cmap='viridis')
plt.colorbar()

plt.subplot(336)
plt.title('Spectrul pentru Ym1,m2')
plt.imshow(20 * np.log10(abs(Y_fft)))
plt.colorbar()

plt.tight_layout()
plt.show()