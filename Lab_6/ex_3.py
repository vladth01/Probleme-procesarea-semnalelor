import numpy as np
import math
import matplotlib.pyplot as plt

#Cream functia de initializare cu 1 a matricii ferestrei dreptunghiulare
def fereastra_dreptunghica(Nw):
    return np.ones(Nw)

#Cream functia de initializare cu formula Hanning a metricii ferestrei Hanning
def fereastra_Hanning(Nw):
    return np.hanning(Nw)

#Cream functia de afisare a ploturilor de comparatie a graficului ferestrei drpetunghiulare cu semnalul sinusoidal initial si tot de de comparare a acestuia cu ferestra Hamming
def semnal_plot_cu_fereastra(semnal, fereastra, titlu):
    plt.figure(figsize=(8, 6))
    plt.subplot(211)
    plt.plot(semnal)
    plt.title('Semnal original')
    
    plt.subplot(212)
    plt.plot(semnal * fereastra)
    plt.title(titlu)
    
    plt.tight_layout()
    plt.show()
    
#Declaram parametrii sinusoidei
f = 100 #Frecventa
A = 1 #Amplitudinea
phi = 0 #Faza

#Declaram dimensiunea ferestrei
Nw = 200

#Generam semnalul sinusoidal
t = np.arange(Nw)
semnal = A * np.sin(2 * math.pi * f * t / Nw + phi)

#Construim fereastra dreptunghiulara si afisam efectul asupra semnalului
dreptunghic = fereastra_dreptunghica(Nw)
semnal_plot_cu_fereastra(semnal, dreptunghic, 'Fereastra dreptunghiulara')

#Construim fereastra Hanning si afisam efectul asupra semnalului
hanning = fereastra_Hanning(Nw)
semnal_plot_cu_fereastra(semnal, hanning, 'Fereastra Hanning')