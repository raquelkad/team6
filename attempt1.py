import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen
image_path = image_path = "C:/Users/raque/Downloads/zdo2024/images/incision_couples/SA_20220801-104759_9xjxd9p268zh_incision_crop_0.jpg"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Convertir a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar umbralización
_, threshold_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)

# Encontrar contornos
contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar los contornos en la imagen original para visualización
contour_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

# Mostrar resultados
plt.figure(figsize=(12, 8))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB))
plt.title('Grayscale Image')
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(threshold_image, cv2.COLOR_GRAY2RGB))
plt.title('Threshold Image')
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.title('Contours Detected')
plt.show()

print(f"Number of stitches detected: {len(contours)}")
