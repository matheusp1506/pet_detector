"""
Captura um frame do Canal 1 (câmera composta do pátio) e salva
apenas a metade de baixo ("Principal"), já no mesmo sistema de
coordenadas usado por ROI e CROSS_LINE_X no pet_detector.py.

Uso:
    python capture_bottom_half.py

Gera "bottom_half.jpg". Abra em um editor de imagem que mostre
coordenadas de pixel (ex: Paint, GIMP) para calibrar ROI e
CROSS_LINE_X no pet_detector.py.
"""

import os
from dotenv import load_dotenv
import cv2

load_dotenv()

NVR_IP = os.getenv("NVR_IP", "")
NVR_PORT = int(os.getenv("NVR_PORT", "554"))

USERNAME = os.getenv("NVR_USERNAME", "")
PASSWORD = os.getenv("NVR_PASSWORD", "")

CHANNEL = 1
STREAM = 0

# Deve bater com COMPOSITE_SPLIT_Y no pet_detector.py.
COMPOSITE_SPLIT_Y = 1296


def build_rtsp_url():
    return (
        f"rtsp://{USERNAME}:{PASSWORD}@"
        f"{NVR_IP}:{NVR_PORT}"
        f"/user={USERNAME}&password={PASSWORD}"
        f"&channel={CHANNEL}&stream={STREAM}.sdp?real_stream"
    )


def main():
    url = build_rtsp_url()
    print("Conectando ao canal 1...")

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("ERRO: não foi possível conectar.")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("ERRO: não recebeu frame.")
        return

    print(f"Frame completo: {frame.shape[1]}x{frame.shape[0]}")

    bottom = frame[COMPOSITE_SPLIT_Y:, :]

    print(f"Metade de baixo: {bottom.shape[1]}x{bottom.shape[0]}")

    cv2.imwrite("bottom_half.jpg", bottom)
    print("Salvo em bottom_half.jpg")
    print()
    print("Abra a imagem e anote as coordenadas (x, y) dos")
    print("cantos da área do pátio para definir ROI, e a")
    print("coordenada X da linha de entrada/saída para")
    print("CROSS_LINE_X, no pet_detector.py.")


if __name__ == "__main__":
    main()
