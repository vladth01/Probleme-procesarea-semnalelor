"""a)"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

#Declaram numarul de puncte in seria de timp
N = 1000

#Creem seria de timp cu timpul de la 0 la 10 cu N puncte
t = np.linspace(0, 10, N)

#Creem componenta trend a seriei - o ecuatie de grad 2 pe care am ales-o noi
trend = 0.1 * t**2 + 0.5 * t

#Creem componenta de sezon care are 2 frecvente pe care le creem noi
sezon = 0.5 * np.sin(2 * np.pi * 0.1 * t) + 0.3 * np.cos(2 * np.pi * 0.2 * t)

#Creem componente de variatii mici definita prin zgomot alb gaussian
variatii_mici = np.random.normal(0, 0.2, N)

#Creem seria de timp ca suma a celor 3 componente
serie_timp = trend + sezon + variatii_mici

#Plotam seria de timp si componentele sale separat
plt.figure(figsize=(10, 6))

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
plt.title('variatii mici')
plt.plot(t, variatii_mici, label='Variatii mici')
plt.legend()

plt.tight_layout()
plt.show()

"""b)"""

# Calculam autoacceleratia prin functia np.correlate, care o calculează
# pentru întreaga serie de timp, pentru toate întârzierile posibile.
# Apoi, rezultatul este normalizat pentru a fi comparabil între diferite serii de timp.
# Modul 'full' calculează autocorelația pentru toate întârzierile
autoacceleratie = np.correlate(serie_timp, serie_timp, mode='full')

# Normalizam autoacceleratia
autoacceleratie /= np.max(autoacceleratie)

# Creem vectorul de intarzieri corespunzatoare autoacceleratiei
intarzieri = np.arange(-N + 1, N)

# Plotam autoacceleratiei
plt.figure(figsize=(8, 5))
plt.plot(intarzieri, autoacceleratie)
plt.title('Autoacceleratia seriei de timp')
plt.xlabel('Intarzieri')
plt.ylabel('Autoacceleratie')
plt.grid(True)
plt.show()

"""c)"""

from scipy.linalg import toeplitz

# Definim oridnul modelului AR, a carui valoare o putem schimba in functie de cerintele noastre
p = 10

# Construim amtricea pentru modelul AR (matricea toeplitz)
def matrice_AR(date, p):
    lungime_date = len(date)
    matrice = [date[p - i - 1 : lungime_date - i - 1] for i in range(p)]
    return np.array(matrice)

# Construim matricea X a vectorului y
X = matrice_AR(serie_timp, p)
y = serie_timp[p:]

# Calculam coeficientii modelului AR folosind metoda minimelor patratelor
coeficienti = np.linalg.lstsq(X, y, rcond=None)[0]

# Calculam predictiile pe baza coeficientilro obtinuti
predictii = np.dot(X, coeficienti)

# Plotam seria de timp originala si predictiile
plt.figure(figsize=(10, 6))
plt.plot(t, serie_timp, label = 'Seria de timp originala', color = 'blue')
plt.plot(t[p:], predictii, label = 'Predictii AR', color = 'green')
plt.title('Seria de timp originala si predictiile AR')
plt.legend()
plt.xlabel('Timp')
plt.ylabel('Valoare')
plt.show()
