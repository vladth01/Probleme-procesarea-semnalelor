import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

"""a)"""

# Citirea datelor din fișierul CSV
data = pd.read_csv('Train.csv')

# Selectarea unei porțiuni corespunzătoare pentru 3 zile (de exemplu, primele 72 de ore)
x = data['Count'].values[:72]

"""b)"""

window_sizes = [5, 9, 13, 17]

for w in window_sizes:
    # Utilizarea semnalului filtrat pentru ulterioare analize/vizualizări
    filtered_signal = np.convolve(x, np.ones(w), 'valid') / w
    
"""c)"""

# Frecvența de tăiere (aleasă pentru eliminarea zgomotului înalt)
cutoff_freq = 0.1  # Exemplu de frecvență de tăiere (0.1 corespunde la 10% din frecvența Nyquist)

# Calcularea frecvenței normalizate
nyquist_freq = 0.5
normalized_cutoff_freq = cutoff_freq / nyquist_freq

# Proiectarea filtrului trece-jos
b, a = signal.butter(5, normalized_cutoff_freq, btype='low', analog=False)
w, h = signal.freqz(b, a, worN=8000)

# Afișarea răspunsului în frecvență al filtrului pe scară logaritmică
plt.figure(figsize=(8, 6))
plt.plot(w, 20 * np.log10(abs(h)))
plt.xscale('log')
plt.title('Răspunsul în frecvență al filtrului trece-jos')
plt.xlabel('Frecvență [rad/sample]')
plt.ylabel('Amplitudine [dB]')
plt.grid(which='both', axis='both')
plt.show()

"""d)"""

# Proiectarea filtrului Butterworth
b_butter, a_butter = signal.butter(5, normalized_cutoff_freq, btype='low', analog=False)

# Proiectarea filtrului Chebyshev de tip 1
rp = 5  # Atenuarea ondulațiilor pentru filtrul Chebyshev (începeți cu 5 dB)
b_cheby, a_cheby = signal.cheby1(5, rp, normalized_cutoff_freq, btype='low', analog=False)

"""e)"""

# Filtrarea datelor de trafic cu cele două filtre
filtered_signal_butter = signal.filtfilt(b_butter, a_butter, x)
filtered_signal_cheby = signal.filtfilt(b_cheby, a_cheby, x)

# Afișarea semnalelor filtrate împreună cu datele brute
plt.figure(figsize=(12, 6))
plt.plot(x, label='Date brute')
plt.plot(filtered_signal_butter, label='Filtru Butterworth')
plt.plot(filtered_signal_cheby, label='Filtru Chebyshev')
plt.legend()
plt.title('Compararea semnalelor filtrate cu datele brute')
plt.xlabel('Eșantioane')
plt.ylabel('Număr de mașini')
plt.show()

"""f)"""

# Reproiectarea filtrelor cu ordine diferite (mai mică sau mai mare)
b_butter_low_order, a_butter_low_order = signal.butter(3, normalized_cutoff_freq, btype='low', analog=False)
b_butter_high_order, a_butter_high_order = signal.butter(7, normalized_cutoff_freq, btype='low', analog=False)

# Reproiectarea filtrului Chebyshev cu alte valori ale atenuării ondulațiilor
rp_new = 3  # Schimbarea valorii atenuării ondulațiilor pentru filtrul Chebyshev
b_cheby_new, a_cheby_new = signal.cheby1(5, rp_new, normalized_cutoff_freq, btype='low', analog=False)