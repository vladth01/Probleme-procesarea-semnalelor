"""a)"""

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

"""b)"""

# Calcularea autocorelației folosind np.correlate
autocorrelation = np.correlate(serie_timp - serie_timp.mean(), serie_timp - serie_timp.mean(), mode='full')

# Normalizarea autocorelației
autocorrelation /= np.max(autocorrelation)

# Crearea vectorului de lag-uri corespunzător autocorelației
lags = np.arange(-N + 1, N)

# Desenarea autocorelației
plt.figure(figsize=(8, 4))
plt.title('Autocorelație a seriei de timp')
plt.xlabel('Lag')
plt.ylabel('Autocorelație')
plt.stem(lags, autocorrelation)
plt.show()

"""c)"""

from scipy.linalg import toeplitz, lstsq

# Funcție pentru a estima coeficienții modelului AR folosind matricea Toeplitz și lstsq
def estimate_ar_coefficients_scipy(data, p):
    # Calculul matricei de autocorelații până la lag-ul p
    autocorr = [np.correlate(data[i:], data[:-i], mode='valid')[0] for i in range(1, p + 1)]
    
    # Construirea matricei Toeplitz
    r = np.array(autocorr)
    r = np.concatenate([[1], r])  # Adăugarea primului element pentru autocorelația la lag 0
    r = toeplitz(r[:-1])
    
    # Calculul termenului constant din sistemul de ecuații
    b = np.array(autocorr)
    
    # Rezolvarea sistemului de ecuații liniare pentru a obține coeficienții AR
    ar_coef = lstsq(r, b)[0]
    
    return ar_coef

# Funcție pentru a face predicții pe baza coeficienților modelului AR
def predict_ar_model_scipy(data, ar_coef, p):
    N = len(data)
    predictions = np.zeros(N - p)
    
    for i in range(p, N):
        predicted_value = np.dot(data[i - p:i][::-1], ar_coef)
        predictions[i - p] = predicted_value
    
    return predictions

# Estimarea coeficienților modelului AR folosind scipy
ar_order = 10  # Ordinul modelului AR
ar_coef_scipy = estimate_ar_coefficients_scipy(serie_timp, ar_order)

# Realizarea predicțiilor pe baza coeficienților AR
predictions_ar_scipy = predict_ar_model_scipy(serie_timp, ar_coef_scipy, ar_order)

# Afișarea seriei de timp originală și predicțiilor modelului AR
plt.figure(figsize=(8, 4))
plt.title('Seria de timp originală și predicțiile AR (SciPy)')
plt.plot(t, serie_timp, label='Seria de timp originală')
plt.plot(t[ar_order:], predictions_ar_scipy, label='Predicții AR (SciPy)')
plt.legend()
plt.show()

"""d)"""

from scipy.linalg import toeplitz, lstsq

# Funcție pentru a estima coeficienții modelului AR folosind matricea Toeplitz și lstsq
def estimate_ar_coefficients_scipy(data, p):
    autocorr = [np.correlate(data[i:], data[:-i], mode='valid')[0] for i in range(1, p + 1)]
    r = np.array(autocorr)
    r = np.concatenate([[1], r])
    r = toeplitz(r[:-1])
    b = np.array(autocorr)
    ar_coef = lstsq(r, b)[0]
    return ar_coef

# Funcție pentru a face predicții pe baza coeficienților modelului AR
def predict_ar_model_scipy(data, ar_coef, p):
    N = len(data)
    predictions = np.zeros(N - p)
    for i in range(p, N):
        predicted_value = np.dot(data[i - p:i][::-1], ar_coef)
        predictions[i - p] = predicted_value
    return predictions

# Funcție pentru evaluarea modelului AR cu anumite valori pentru p și m
def evaluate_ar_model(data, p, m):
    train_size = int(len(data) * 0.8)  # Setul de antrenare - 80% din datele disponibile
    train, test = data[:train_size], data[train_size:]  # Separarea datelor
    
    # Estimarea coeficienților pentru modelul AR
    ar_coef = estimate_ar_coefficients_scipy(train, p)
    
    # Realizarea predicțiilor pentru setul de testare
    predictions = predict_ar_model_scipy(train, ar_coef, p)
    
    # Calcularea erorii pentru predicțiile realizate
    error = ((test[:m] - predictions[-m:]) ** 2).mean()
    
    return error

best_error = float('inf')
best_p, best_m = 0, 0

# Căutarea grilei pentru a găsi cele mai bune valori pentru p și m
for p in range(1, 15):
    for m in range(1, 15):
        error = evaluate_ar_model(serie_timp, p, m)
        if error < best_error:
            best_error = error
            best_p, best_m = p, m

print(f"Cea mai bună performanță obținută pentru p = {best_p} și m = {best_m} (MSE: {best_error})")