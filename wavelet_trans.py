import numpy as np
import matplotlib.pyplot as plt

import pywt
import pywt.data


# Load image
original = Image.open("dataset/train/An Xiaoxiao/1.png").convert('L')

# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(original, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()

# Load image
original = Image.open("dataset/train/An Xiaoxiao/1.png").convert('L')
coeffs2 = pywt.dwt2(original, 'bior1.3')
LL, (LH, HL, HH) = coeffs2

plt.imshow(LL, interpolation="nearest", cmap=plt.cm.gray)
plt.xticks([])
plt.yticks([]);
plt.savefig("1.png")

image = Image.open("1.png")
image.tobytes()
