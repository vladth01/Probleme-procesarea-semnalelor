"""Exercitiile complete din laboratorul 3"""

import numpy as np
import matplotlib.pyplot as plt
import math

#Exercitiul 1

# Definim valoarea lui N
N = 8

# Cream matricei Fourier
fourier_matrix = np.zeros((N, N), dtype=np.complex64)

for n in range(N):
    for k in range(N):
        fourier_matrix[n, k] = np.exp(-2j * math.pi * n * k / N)

# Desenarea părții reale și imaginare pe subplot-uri
fig, axs = plt.subplots(1, 2, figsize=(10, 6))
fig.suptitle('Matricea Fourier pentru N=8')

# Partea reală a matricei Fourier
axs[0].imshow(np.real(fourier_matrix), cmap='gray', origin='upper', extent=(0, N, 0, N))
axs[0].set_title('Partea Reală')

# Partea imaginară a matricei Fourier
axs[1].imshow(np.imag(fourier_matrix), cmap='gray', origin='upper', extent=(0, N, 0, N))
axs[1].set_title('Partea Imaginară')

# Afișarea plot-urilor
plt.show()

# Calculam și desenarea produsului matricei cu transpusa sa
product = np.dot(fourier_matrix, np.conj(fourier_matrix).T)

# Verificam ortogonalitatea matricii Fourier
identity_matrix = np.eye(N)
is_orthogonal = np.allclose(product, identity_matrix)
if is_orthogonal:
    print("Matricea Fourier este ortogonala.")
else:
    print("Matricea Fourier nu este ortogonala.")