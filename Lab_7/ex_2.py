import numpy as np
import matplotlib.pyplot as plt
from scipy import misc, ndimage

#Cream si afisam imaginea initiala cu ratonul
X = misc.face(gray=True)
plt.title('Imaginea initiala cu ratonul')
plt.imshow(X, cmap=plt.cm.gray)
plt.show()

req_cutoff = 120

Y_cutoff = Y.copy()
Y_cutoff[freq_db > freq_cutoff] = 0
X_cutoff = np.fft.ifft2(Y_cutoff)
X_cutoff = np.real(X_cutoff)    # avoid rounding erros in the complex domain,
                                # in practice use irfft2
plt.imshow(X_cutoff, cmap=plt.cm.gray)
plt.show()