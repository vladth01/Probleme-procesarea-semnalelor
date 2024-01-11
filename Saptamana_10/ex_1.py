import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from numpy.linalg import eig

# Declaram o distributie Gaussiana unidimensionala
media_uni = 0 # media
var_uni = 1 # variantza
distributie_uni = np.random.normal(media_uni, np.sqrt(var_uni), 1000)

# Afisam graficul pentru distributia Gaussiana unidimensionala
plt.figure(figsize=(12, 6))
plt.hist(distributie_uni, bins=30, density=True)
plt.title("Distributie Gaussiana Unidimensionala")
plt.xlabel("Valoare")
plt.ylabel("Densitate")
plt.grid(True)
plt.show()

# Declaram o distributie Gaussiana bidimensioanla
media_bi = [0, 0] # media
cov_bi = [[1, 0.5], [0.5, 1]] # matricea de covarianta

# Esantionam distributia Gaussiana bidimensionala
distributie_bi = np.random.multivariate_normal(media_bi, cov_bi, 1000)

# Afisam graficul pentru distributia Gaussiana bidimensionala
plt.figure(figsize=(12, 6))
plt.scatter(distributie_bi[:, 0], distributie_bi[:, 1])
plt.title("Distributie Gaussiana Bidimensionala")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

# Verificam descompunerea valorilor proprii
U, Lambda, UT = np.linalg.svd(cov_bi)
Sigma = U @ np.diag(Lambda) @ UT

# Esantionam folosind metoda din cursul 10 silde 11
n = np.random.normal(0, 1, (2, 1000))
x = U @ np.sqrt(np.diag(Lambda)) @ n + np.array(media_bi).reshape(2, 1)

# Afisam graficul pentru metoda descompunerii valorilor proprii
plt.figure(figsize=(12, 6))
plt.scatter(x[0], x[1])
plt.title("Distributie Gaussiana Bidimensionala (Metoda Valorilor Proprii)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()