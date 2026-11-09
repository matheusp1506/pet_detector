"""
Testa múltiplos canais RTSP do NVR e salva um frame de cada,
para identificar visualmente qual CHANNEL corresponde a qual
câmera física (ex: qual é a que enquadra o pátio/quintal).

Uso:
    python test_channels.py

Gera arquivos "channel_1.jpg", "channel_2.jpg", etc. na mesma
pasta. Abra cada um e veja qual mostra a área que te interessa.
"""

import os
from dotenv import load_dotenv
import cv2

load_dotenv()

NVR_IP = os.getenv("NVR_IP", "")
NVR_PORT = int(os.getenv("NVR_PORT", "554"))

USERNAME = os.getenv("NVR_USERNAME", "")
PASSWORD = os.getenv("NVR_PASSWORD", "")

# 3 câmeras físicas (canais 1-3), cada uma com stream
# principal (0) e substream (1).
CHANNELS_TO_TEST = [1, 2, 3]
STREAMS_TO_TEST = [0, 1]


def build_rtsp_url(channel, stream):
    return (
        f"rtsp://{USERNAME}:{PASSWORD}@"
        f"{NVR_IP}:{NVR_PORT}"
        f"/user={USERNAME}&password={PASSWORD}"
        f"&channel={channel}&stream={stream}.sdp?real_stream"
    )


def main():
    for channel in CHANNELS_TO_TEST:
        for stream in STREAMS_TO_TEST:
            url = build_rtsp_url(channel, stream)
            label = f"channel{channel}_stream{stream}"
            print(f"Testando {label}...")

            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

            if not cap.isOpened():
                print(f"  {label}: não foi possível conectar.")
                cap.release()
                continue

            ret, frame = cap.read()

            if not ret or frame is None:
                print(f"  {label}: conectou, mas não recebeu frame.")
                cap.release()
                continue

            filename = f"{label}.jpg"
            cv2.imwrite(filename, frame)
            print(f"  {label}: OK -> salvo em {filename}")

            cap.release()

    print()
    print("Concluído. Abra os arquivos channel*_stream*.jpg gerados")
    print("e compare com o app para identificar cada câmera.")


if __name__ == "__main__":
    main()
