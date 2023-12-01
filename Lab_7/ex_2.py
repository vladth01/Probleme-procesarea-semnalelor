import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, fftpack

# Încărcați o imagine de test
X = misc.face(gray=True)

# Aplicați transformata Fourier
X_fft = fftpack.fft2(X)

# Calculați spectrul imaginii
X_fft_shifted = fftpack.fftshift(X_fft)
magnitude_spectrum = np.abs(X_fft_shifted)

# Calculați pragul pentru atenuarea frecvențelor înalte
prag_SNR = 50  # Pragul pentru raportul semnal-zgomot (SNR)
max_magnitude = np.max(magnitude_spectrum)
prag_valoare = max_magnitude / prag_SNR

# Anulați frecvențele de peste prag
X_fft_shifted_filtered = X_fft_shifted * (magnitude_spectrum < prag_valoare)
X_fft_filtered = fftpack.ifftshift(X_fft_shifted_filtered)

# Aplicați transformata Fourier inversă
X_comprimat = fftpack.ifft2(X_fft_filtered).real

# Afișați imaginea originală și imaginea comprimată
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(X, cmap='gray')
plt.title('Imaginea originală')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(X_comprimat, cmap='gray')
plt.title('Imaginea comprimată')
plt.axis('off')

plt.tight_layout()
plt.show()