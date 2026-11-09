"""
Ferramenta de calibração visual de ROI e CROSS_LINE_X.

Captura um frame do Canal 1, recorta a metade de baixo
(câmera composta) e desenha a ROI + linha de cruzamento em
cima, salvando "roi_check.jpg" para você conferir.

Ajuste ROI e CROSS_LINE_X abaixo, rode de novo, e repita até
o retângulo/linha caírem exatamente onde você quer. Depois,
copie os mesmos valores para pet_detector.py.

Uso:
    python roi_test.py

Para não reconectar à câmera toda hora enquanto ajusta só as
coordenadas, rode uma vez com USE_CACHED_FRAME = False, e nas
próximas com True (reaproveita bottom_half.jpg salvo).
"""

import os
from dotenv import load_dotenv
import cv2

# ============================================================
# CONEXÃO (só usada se USE_CACHED_FRAME = False)
# ============================================================

load_dotenv()

NVR_IP = os.getenv("NVR_IP", "")
NVR_PORT = int(os.getenv("NVR_PORT", "554"))

USERNAME = os.getenv("NVR_USERNAME", "")
PASSWORD = os.getenv("NVR_PASSWORD", "")

CHANNEL = 1
STREAM = 0

# ============================================================
# CÂMERA COMPOSTA
# ============================================================

COMPOSITE_SPLIT_Y = 1296

# ============================================================
# VALORES A CALIBRAR — ajuste e rode de novo
# ============================================================
# Coordenadas relativas à metade de baixo (pátio), mesmo
# sistema usado em pet_detector.py.

ROI = (1700, 670, 1900, 1080)
CROSS_LINE_X = 1800

# Linha horizontal opcional (topo/base). None para desativar.
CROSS_LINE_Y = 875

# ============================================================
# CONTROLE DE CACHE
# ============================================================
#
# True  -> reaproveita bottom_half.jpg já salvo (rápido, para
#          só ajustar coordenadas sem reconectar à câmera).
# False -> conecta de novo no NVR e captura um frame fresco.

USE_CACHED_FRAME = True


def build_rtsp_url():
    return (
        f"rtsp://{USERNAME}:{PASSWORD}@"
        f"{NVR_IP}:{NVR_PORT}"
        f"/user={USERNAME}&password={PASSWORD}"
        f"&channel={CHANNEL}&stream={STREAM}.sdp?real_stream"
    )


def capture_bottom_half():
    url = build_rtsp_url()
    print("Conectando ao canal 1...")

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise RuntimeError("Não foi possível conectar ao NVR.")

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Não recebeu frame do NVR.")

    bottom = frame[COMPOSITE_SPLIT_Y:, :]
    cv2.imwrite("bottom_half.jpg", bottom)

    return bottom


def main():
    if USE_CACHED_FRAME:
        frame = cv2.imread("bottom_half.jpg")
        if frame is None:
            print("bottom_half.jpg não encontrado, capturando frame novo...")
            frame = capture_bottom_half()
        else:
            print("Usando bottom_half.jpg em cache.")
    else:
        frame = capture_bottom_half()

    print(f"Frame: {frame.shape[1]}x{frame.shape[0]}")

    img = frame.copy()

    if ROI is not None:
        x1, y1, x2, y2 = ROI
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        line_y1, line_y2 = y1, y2
    else:
        line_y1, line_y2 = 0, img.shape[0]

    cv2.line(img, (CROSS_LINE_X, line_y1), (CROSS_LINE_X, line_y2), (255, 0, 255), 3)

    if CROSS_LINE_Y is not None:
        if ROI is not None:
            line_x1, line_x2 = ROI[0], ROI[2]
        else:
            line_x1, line_x2 = 0, img.shape[1]
        cv2.line(img, (line_x1, CROSS_LINE_Y), (line_x2, CROSS_LINE_Y), (255, 255, 0), 3)

    cv2.imwrite("roi_check.jpg", img)

    print()
    print(f"ROI:           {ROI}")
    print(f"CROSS_LINE_X:  {CROSS_LINE_X}")
    print(f"CROSS_LINE_Y:  {CROSS_LINE_Y}")
    print()
    print("Salvo em roi_check.jpg. Confira e ajuste ROI/")
    print("CROSS_LINE_X no topo deste arquivo, depois rode")
    print("de novo. Quando estiver bom, copie os valores para")
    print("pet_detector.py.")


if __name__ == "__main__":
    main()
