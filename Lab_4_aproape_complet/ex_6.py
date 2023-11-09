import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa

# (a) Citirea semnalului audio dintr-un fișier
sample_rate, data = wavfile.read('C:\Users\Vladth01\OneDrive - unibuc.ro\An IV\Semestrul I\Procesarea semnalelor\Săptămâna_4\Lab_4\Vocale_inregistrate.wav')

# (b) Gruparea valorilor semnalului
window_size = int(0.01 * sample_rate)  # 1% din lungimea semnalului
overlap = int(window_size / 2)  # Suprapunere de 50%

# (c) Calculul FFT pentru fiecare grup
spectrogram = []
for i in range(0, len(data) - window_size, overlap):
    segment = data[i:i + window_size]
    fft_result = np.abs(np.fft.fft(segment))
    spectrogram.append(fft_result)

# (d) Pregătirea matricei
spectrogram_matrix = np.transpose(np.array(spectrogram))

# (e) Afișarea matricei într-o figură de tip spectogramă
plt.imshow(spectrogram_matrix, aspect='auto', cmap='viridis')
plt.colorbar(label='Amplitudine')
plt.xlabel('Timp')
plt.ylabel('Frecvență')
plt.title('Spectrograma')
plt.show()