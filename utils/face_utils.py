import cv2
import numpy as np
from PIL import Image
import io

def obtener_embeddings_lbp(imagen_bytes):
    try:
        # Cargar imagen y convertir a escala de grises
        imagen = Image.open(io.BytesIO(imagen_bytes)).convert('L')
        imagen_np = np.array(imagen)

        # Redimensionar a 128x128
        imagen_np = cv2.resize(imagen_np, (128, 128))

        # ✅ Ecualizar histograma para mejorar contraste
        imagen_np = cv2.equalizeHist(imagen_np)

        # Aplicar LBP
        lbp = np.zeros_like(imagen_np)

        for i in range(1, imagen_np.shape[0] - 1):
            for j in range(1, imagen_np.shape[1] - 1):
                centro = imagen_np[i, j]
                binario = ''
                binario += '1' if imagen_np[i-1, j-1] >= centro else '0'
                binario += '1' if imagen_np[i-1, j  ] >= centro else '0'
                binario += '1' if imagen_np[i-1, j+1] >= centro else '0'
                binario += '1' if imagen_np[i  , j+1] >= centro else '0'
                binario += '1' if imagen_np[i+1, j+1] >= centro else '0'
                binario += '1' if imagen_np[i+1, j  ] >= centro else '0'
                binario += '1' if imagen_np[i+1, j-1] >= centro else '0'
                binario += '1' if imagen_np[i  , j-1] >= centro else '0'
                lbp[i, j] = int(binario, 2)

        # Calcular histograma de LBP
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)  # Normalización

        return hist.tolist()

    except Exception as e:
        print("Error LBP:", e)
        return None
    
