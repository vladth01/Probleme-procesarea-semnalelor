import numpy as np
import matplotlib.pyplot as plt

# Declaram numarul de mostre de numere
N = 100

# Intervalul de lucru pentru x
x = np.linspace(-1, 1, N)

# Hyperparametrii
alpha = 0.5 # Pentru procesele Gaussiene care necesita un alpha
beta = 1.0 # Pentru procesul gaussian periodic

# Definirea functiilor nucleu pentru diferitele procese Gaussiene

def nucleu_liniar(x, y):
    return np.dot(x, y)

def nucleu_miscare_browniana(s, t):
    return np.minimum(s, t)

def nucleu_exponentiala_patrata(x, y):
    return np.exp(-alpha * np.linalg.norm(x - y)**2)

def nucleu_Ornstein_Uhlenbeck(s, t):
    return np.exp(-alpha * np.abs(s - t))

def nucleu_periodic(x, y):
    return np.exp(-alpha * np.sin(np.pi * beta * (x - y))**2)

def nucleu_simetric(x, y):
    return np.exp(-alpha * min(np.abs(x - y), np.abs(x + y))**2)

# Functie care genereaza si afiseaza procesele Gaussiene
def genereaza_si_afiseaza_graficele_proceselor(functie_nucleu, tilu):
    # Calculul matricei de covarianta
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            C[i, j] = functie_nucleu(x[i], x[j])
            
    # Esantionam din distributia Gaussiana
    z = np.random.multivariate_normal(np.zeros(N), C)
    
    # Afisam graficul
    plt.figure(figsize=(10, 5))
    plt.plot(x, z)
    plt.title(tilu)
    plt.xlabel('x')
    plt.ylabel('z')
    plt.grid(True)
    plt.show()
    
# Generam si afisam procesle Gaussiene
genereaza_si_afiseaza_graficele_proceselor(nucleu_liniar, "Proces Gaussian liniar")
genereaza_si_afiseaza_graficele_proceselor(nucleu_miscare_browniana, "Proces Gaussian miscare Browniana")
genereaza_si_afiseaza_graficele_proceselor(nucleu_exponentiala_patrata, "Proces Gaussian exponential patratic")
genereaza_si_afiseaza_graficele_proceselor(nucleu_Ornstein_Uhlenbeck, "Proces Gaussian Ornstein-Uhlenbeck")
genereaza_si_afiseaza_graficele_proceselor(nucleu_periodic, "Proces Gaussian periodic")
genereaza_si_afiseaza_graficele_proceselor(nucleu_simetric, "Proces Gaussian simetric")