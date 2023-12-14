"""1."""

import numpy as np
import matplotlib.pyplot as plt

# Lungimea seriei de timp
N = 1000

# Generare timp
t = np.arange(N)

# Componenta trend - ecuație de grad 2
trend = 0.02 * t**2 + 3 * t + 1000

# Componenta sezon - două frecvențe
sezon = 10 * np.sin(2 * np.pi * t / 50) + 5 * np.cos(2 * np.pi * t / 30)

# Componenta variatiilor mici - zgomot alb gaussian
variatiile_mici = np.random.normal(0, 1, N)

# Seria de timp - suma celor trei componente
serie_timp = trend + sezon + variatiile_mici

# Plotează seria de timp și componentele separate
plt.figure(figsize=(12, 6))

plt.subplot(411)
plt.title('Seria de timp')
plt.plot(t, serie_timp, label='Seria de timp')
plt.legend()

plt.subplot(412)
plt.title('Trend')
plt.plot(t, trend, label='Trend')
plt.legend()

plt.subplot(413)
plt.title('Sezon')
plt.plot(t, sezon, label='Sezon')
plt.legend()

plt.subplot(414)
plt.title('Variatiile mici')
plt.plot(t, variatiile_mici, label='Variatiile mici')
plt.legend()

plt.tight_layout()
plt.show()

"""2."""

# Definim valorile lui alpha
valori_alpha = np.linspace(0.01, 0.99, 100)  # 100 de alpha iau valori intre 0.01 si 0.99

# Initializam variabilele pentru a stoca valorile optimale
eroare_minima = float('inf')
alpha_optim = None
s_optim = None

# Calculam medierea exponentiala pentru fiecare aplha 
for alpha in valori_alpha:
    s = np.zeros(N)
    s[0] = serie_timp[0]  # Setam prima valuare ca valuare initiala
    
    for i in range(1, N):
        s[i] = alpha * serie_timp[i] + (1 - alpha) * s[i - 1]
    
    # Calculam eroarea dintre seria originala si cea mediata
    eroare = np.sum(np.square(serie_timp - s))
    
    # Verificam dupa eroare minima
    if eroare < eroare_minima:
        eroare_minima = eroare
        alpha_optim = alpha
        s_optim = s

# Plotam seria originala si seria mediata exponential cu alpha optim
plt.figure(figsize=(10, 6))
plt.plot(t, serie_timp, label='Seria de timp originală', alpha=0.7)
plt.plot(t, s_optim, label=f'Serie mediată (Alpha = {alpha_optim:.4f})', linestyle='--')
plt.title('Seria de timp originală vs. Seria mediată exponențial')
plt.legend()
plt.show()

print(f"Valoarea optimă a lui alpha este: {alpha_optim:.4f}")

"""3."""

# Parametrii modelului MA
q = 3  # Orizontul modelului MA
theta = np.array([0.5, -0.3, 0.2])  # Coeficienții theta pentru ϵ[i - 1], ϵ[i - 2], ϵ[i - 3]

# Generarea termenilor de eroare aleatori din distribuția normală standard
error_terms = np.random.normal(0, 1, N)

# Inițializarea seriei MA cu valorile seriei de timp originale
ma_series = np.copy(serie_timp)

# Calculul seriei MA utilizând termenii de eroare și coeficienții theta
for i in range(q, N):
    ma_term = np.dot(theta, error_terms[i-q:i][::-1])  # Calculul termenului MA pentru momentul i
    ma_series[i] = error_terms[i] + ma_term

# Plotează seria de timp originală și seria MA
plt.figure(figsize=(10, 6))
plt.plot(t, serie_timp, label='Seria de timp originală', alpha=0.7)
plt.plot(t, ma_series, label=f'Serie MA (Orizont q = {q})', linestyle='--')
plt.title('Seria de timp originală vs. Seria MA')
plt.legend()
plt.show()