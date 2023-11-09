import numpy as np
import matplotlib.pyplot as plt

# Parametrii semnalului sinusoidal original
frecventa_originala = 10  # Frecvența semnalului în Hz
amplitudine = 1
faza = 0

# Frecvența de eșantionare mai mare decât Nyquist
frecventa_eșantionare = 30  # Frecvența de eșantionare mai mare decât Nyquist

# Timpul continuu
t = np.linspace(0, 1, 1000)

# Semnalul sinusoidal original
semnal_original = amplitudine * np.sin(2 * np.pi * frecventa_originala * t)

# Eșantionarea semnalului original cu frecvența de eșantionare mai mare decât Nyquist
t_eșantionat = np.linspace(0, 1, int(1 / (frecventa_eșantionare)))

semnal_eșantionat = amplitudine * np.sin(2 * np.pi * frecventa_originala * t_eșantionat)

# Construirea celorlalte două semnale
frecventa_1 = 5
frecventa_2 = 20

semnal_1 = amplitudine * np.sin(2 * np.pi * frecventa_1 * t)
semnal_2 = amplitudine * np.sin(2 * np.pi * frecventa_2 * t)

# Eșantionarea celor două semnale suplimentare cu aceeași frecvență de eșantionare mai mare decât Nyquist
semnal_eșantionat_1 = amplitudine * np.sin(2 * np.pi * frecventa_1 * t_eșantionat)
semnal_eșantionat_2 = amplitudine * np.sin(2 * np.pi * frecventa_2 * t_eșantionat)

# Plotează cele patru grafice
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(t, semnal_original, label='Semnal original')
plt.title('Semnal sinusoidal original')
plt.xlabel('Timp (s)')
plt.ylabel('Amplitudine')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t, semnal_original, label='Semnal original', linestyle='--', color='red')
plt.plot(t_eșantionat, semnal_eșantionat, 'bo', markersize=4, label='Eșantionare cu frecvență mai mare decât Nyquist')
plt.title('Eșantionare cu frecvență mai mare decât Nyquist')
plt.xlabel('Timp (s)')
plt.ylabel('Amplitudine')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t, semnal_1, label='Semnal 1 (frecvență diferită)')
plt.title('Semnal sinusoidal 1')
plt.xlabel('Timp (s)')
plt.ylabel('Amplitudine')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t, semnal_1, label='Semnal 1 (frecvență diferită)', linestyle='--', color='red')
plt.plot(t_eșantionat, semnal_eșantionat_1, 'go', markersize=4, label='Eșantionare cu frecvență mai mare decât Nyquist')
plt.title('Eșantionare cu frecvență mai mare decât Nyquist pentru semnal 1')
plt.xlabel('Timp (s)')
plt.ylabel('Amplitudine')
plt.legend()

plt.tight_layout()
plt.show()