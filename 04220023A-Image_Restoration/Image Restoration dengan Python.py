import cv2
import numpy as np

# Load gambar yang ingin direstorasi
image = cv2.imread('Gambar/Mugiwara.jpg')

# Konversi gambar menjadi grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplikasikan teknik restorasi pada gambar grayscale

# 1. Denoising (Penghilangan Noise)
denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# 2. Deblurring (Penghilangan Kabur)
kernel = np.ones((5, 5), np.float32) / 25
deblurred = cv2.filter2D(denoised, -1, kernel)

# 3. Sharpening (Peningkatan Ketajaman)
sharpened = cv2.addWeighted(deblurred, 1.5, denoised, -0.5, 0)

# Menampilkan gambar asli, denoised, deblurred, dan sharpened
cv2.imshow("Gambar Asli", image)
cv2.imshow("Denoised", denoised)
cv2.imshow("Deblurred", deblurred)
cv2.imshow("Sharpened", sharpened)

