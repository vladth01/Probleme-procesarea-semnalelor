import numpy as np
from numpy.fft import fft, ifft

"""Calculul direct al produsului polinoamelor"""

#Generam coeficientilor aleatori pentru polinoamele p(x) si q(x)
N = 5 # Gradul maxim al polinoamelor
coeficienti_p = np.random.randint(-10, 10, size= N + 1) # Coeficientii pentru p(x)
coeficienti_q = np.random.randint(-10, 10, size= N + 1) # Coeficientii pentru q(x)

# Calculam produsul polinoamelor direct si il afisam
produsul_coeficientilor_direct = np.convolve(coeficienti_p, coeficienti_q)
print("Produsul direct al polinoamelor este ", produsul_coeficientilor_direct)

"""Calculul produsului polinoamelor folosind cu fft"""

#Facem zero-padding pentru a ne asigura ca dimensiunea e corecta pentru transformata Fourier
marime_noua = 2 * N + 1
pad_p = np.pad(coeficienti_p, (0, marime_noua - len(coeficienti_p)), 'constant')
pad_q = np.pad(coeficienti_q, (0, marime_noua - len(coeficienti_q)), 'constant')

#Calculam produsul polinoamelor in domeniul frecventelor si il afisam
produs_fft = np.round(ifft(pad_p * pad_q).real).astype(int)
produs_coeficienti_fft = produs_fft[:len(produsul_coeficientilor_direct)]
print("Produsul polinoamelor folosind fft este ", produs_coeficienti_fft)