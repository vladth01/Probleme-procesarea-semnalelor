import numpy as np
import matplotlib.pyplot as plt

# Încărcați datele din fișierul CSV
x = np.genfromtxt('Train.csv', delimiter=',')

# Lungimea semnalului
N = len(x)

# Aplicați transformata Fourier
X = np.fft.fft(x)

# Calculați modulul transformatei și normalizați
modul_transformata = np.abs(X / N)

# Folosiți doar jumătate din spectru datorită simetriei
modul_transformata = modul_transformata[:N//2]

# Frecvența de eșantionare (dacă este o înregistrare pe oră, Fs = 1)
Fs = 1  # Frecvența de eșantionare

# Generați vectorul de frecvențe
f = Fs * np.linspace(0, N/2, N//2) / N

# Afizați graficul modulului transformatei Fourier
plt.figure(figsize=(8, 6))
plt.plot(f, modul_transformata)
plt.xlabel('Frecvență (Hz)')
plt.ylabel('Modulul transformatei Fourier')
plt.title('Modulul transformatei Fourier a semnalului')
plt.grid()
plt.show()