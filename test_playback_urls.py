"""
Testa formatos conhecidos de URL RTSP de PLAYBACK (gravações
antigas, não ao vivo) usados por NVRs da família Xmeye/iCSee e
clones. Sem garantia — cada firmware pode variar, e alguns
modelos simplesmente não expõem playback via RTSP (só pela
interface web/app proprietário).

Ajuste START_TIME/END_TIME para um período em que você tem
certeza que há gravação (ex: agora mesmo, ou um horário do dia
que sempre grava).

Uso:
    python test_playback_urls.py

Para cada formato testado, tenta abrir e ler um frame. Se
conseguir, salva um "playback_test_N.jpg" e imprime qual
formato funcionou.
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

# Formato: AAAAMMDDhhmmss. Ajuste para um período com gravação
# confirmada (ex: um evento que você sabe que aconteceu).
START_TIME = "20260905220839"
END_TIME = "20260905221339"


def build_candidate_urls():
    base = f"{USERNAME}:{PASSWORD}@{NVR_IP}:{NVR_PORT}"

    return [
        (
            "Xmeye-style query params (starttime/endtime)",
            f"rtsp://{base}/user={USERNAME}&password={PASSWORD}"
            f"&channel={CHANNEL}&stream={STREAM}.sdp?playback"
            f"&starttime={START_TIME}&endtime={END_TIME}",
        ),
        (
            "Xmeye-style com track= e looptime=",
            f"rtsp://{base}/user={USERNAME}&password={PASSWORD}"
            f"&channel={CHANNEL}&stream={STREAM}&playback=1"
            f"&starttime={START_TIME}&endtime={END_TIME}",
        ),
        (
            "Dahua-style (/cam/playback)",
            f"rtsp://{base}/cam/playback?channel={CHANNEL}&subtype={STREAM}"
            f"&starttime={START_TIME}&endtime={END_TIME}",
        ),
        (
            "Hikvision-style (Streaming/tracks)",
            f"rtsp://{base}/Streaming/tracks/{CHANNEL}01"
            f"?starttime={START_TIME}&endtime={END_TIME}",
        ),
    ]


def main():
    candidates = build_candidate_urls()

    print(f"Testando período {START_TIME} - {END_TIME}...")
    print()

    any_worked = False

    for i, (label, url) in enumerate(candidates, start=1):
        print(f"[{i}] {label}")
        print(f"    URL: {url}")

        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            print("    -> Não conectou.")
            cap.release()
            print()
            continue

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print("    -> Conectou, mas não recebeu frame.")
            print()
            continue

        filename = f"playback_test_{i}.jpg"
        cv2.imwrite(filename, frame)
        print(f"    -> SUCESSO! Frame salvo em {filename}")
        print()
        any_worked = True

    if not any_worked:
        print("Nenhum formato funcionou. Este NVR provavelmente só")
        print("expõe playback pela interface web/app proprietário,")
        print("não via RTSP direto. Siga com exportação manual +")
        print("extract_dataset_from_videos.py.")
    else:
        print("Confira os arquivos playback_test_N.jpg gerados para")
        print("ver qual realmente trouxe o vídeo do horário certo")
        print("(pode ter conectado mas devolvido o stream AO VIVO")
        print("em vez do gravado, se o NVR ignorar os parâmetros).")


if __name__ == "__main__":
    main()
