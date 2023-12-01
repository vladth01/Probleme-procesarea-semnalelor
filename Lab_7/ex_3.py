import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

# Imaginea originală
X = misc.face(gray=True)

# Adăugare zgomot la imagine
pixel_noise = 200
noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
X_noisy = X + noise

# Funcție pentru filtrarea mediană
def median_filter(image, size=3):
    pad_size = size // 2
    padded_image = np.pad(image, pad_size, mode='constant')
    filtered_image = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded_image[i:i+size, j:j+size].flatten()
            median_value = np.median(neighborhood)
            filtered_image[i, j] = median_value
    
    return filtered_image

# Aplică filtrul median pe imaginea zgomotoasă
X_filtrat = median_filter(X_noisy.astype(np.uint8), size=3)

# Calculați raportul SNR înainte și după eliminarea zgomotului
SNR_initial = np.mean(X**2) / np.mean((X - X_noisy)**2)
SNR_final = np.mean(X**2) / np.mean((X - X_filtrat)**2)

print(f"Raportul SNR înainte de eliminarea zgomotului: {SNR_initial:.2f}")
print(f"Raportul SNR după eliminarea zgomotului: {SNR_final:.2f}")

# Afișează imaginea originală, imaginea cu zgomot și imaginea filtrată
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(X, cmap='gray')
plt.title('Imaginea originală')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(X_noisy, cmap='gray')
plt.title('Imaginea cu zgomot')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(X_filtrat, cmap='gray')
plt.title('Imaginea filtrată')
plt.axis('off')

plt.tight_layout()
plt.show()