import numpy as np
import matplotlib.pyplot as plt

#Generam vectorul aleator
N = 100
x = np.random.rand(N)

#Cream lista pentru stocarea vectorilor la fiecare iteratie
iteratii = [x]

#Iteratia x <- x * x de 3 ori
for i in range(3):
    x = x * x
    iteratii.append(x)
    
#Afisam graficele pentru fiecare iteratie
plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.plot(iteratii[0])
plt.title('Vector initial')

for i in range(1, 4):
    plt.subplot(2, 2, i + 1)
    plt.plot(iteratii[i])
    plt.title(f'Interatia {i}')

plt.tight_layout()
plt.show()

#Observatii sunt urmatoarele:
#Creste rapid: Valorile din vector cresc exponențial odată cu fiecare iteratie.
#Convergenta spre 0 sau 1: Dacă valorile inițiale sunt cuprinse între 0 și 1, ele pot converge rapid către 0 sau 1 în primele câteva iterații.
#Sensibilitate la valorile initiale: În funcție de valorile initiale, unele iterații pot conduce la explodarea valorilor vectorului.