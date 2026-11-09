"""
Baixa gravações diretamente do NVR via protocolo DVRIP/Xmeye
(porta 34567), sem precisar exportar manualmente pela interface
web. Usa a biblioteca python-dvr (OpenIPC), instalada com:

    pip install git+https://github.com/OpenIPC/python-dvr.git

O arquivo bruto retornado pelo NVR usa um formato proprietário
(cabeçalhos privados intercalados no stream H264 — magic bytes
0x1fc/0x1fd/0x1fa/0x1f9). Este script extrai só os frames de
vídeo (ignora áudio) e remonta um .h264 Annex-B puro, que o
OpenCV/ffmpeg conseguem abrir normalmente.

Uso:
    python dvr_download.py

Ajuste START/END abaixo para o período que você quer baixar.
Gera os vídeos em dataset/videos/, prontos para
extract_dataset_from_videos.py processar.
"""

import os
import struct

from dotenv import load_dotenv
from dvrip import DVRIPCam

load_dotenv()

NVR_IP = os.getenv("NVR_IP", "")
USERNAME = os.getenv("NVR_USERNAME", "")
PASSWORD = os.getenv("NVR_PASSWORD", "")

CHANNEL = 0  # Canal 1 no app = channel 0 no protocolo (índice 0-based)

# Período a baixar (formato "YYYY-MM-DD HH:MM:SS").
START = "2026-09-05 00:00:00"
END = "2026-09-06 00:00:00"

RAW_DIR = "dataset/videos/_raw"
OUTPUT_DIR = "dataset/videos"


def strip_private_headers(raw_data):
    """
    Remove os cabeçalhos privados do NVR (Xiongmai/Xmeye) do
    arquivo bruto, mantendo só os frames de vídeo (I/P), e
    remonta um H264 Annex-B puro. Áudio (0x1fa) e info (0x1f9)
    são descartados.
    """

    output = bytearray()
    offset = 0
    n = len(raw_data)

    while offset + 4 <= n:
        (data_type,) = struct.unpack(">I", raw_data[offset:offset + 4])

        if data_type == 0x1FC or data_type == 0x1FE:
            if offset + 16 > n:
                break
            _, _, _, _, _, length = struct.unpack(
                "BBBBII", raw_data[offset + 4:offset + 16]
            )
            frame_start = offset + 16
            frame_end = frame_start + length
            output.extend(raw_data[frame_start:frame_end])
            offset = frame_end

        elif data_type == 0x1FD:
            if offset + 8 > n:
                break
            (length,) = struct.unpack("I", raw_data[offset + 4:offset + 8])
            frame_start = offset + 8
            frame_end = frame_start + length
            output.extend(raw_data[frame_start:frame_end])
            offset = frame_end

        elif data_type == 0x1FA:
            if offset + 8 > n:
                break
            _, _, length = struct.unpack("BBH", raw_data[offset + 4:offset + 8])
            offset = offset + 8 + length

        elif data_type == 0x1F9:
            if offset + 8 > n:
                break
            _, _, length = struct.unpack("BBH", raw_data[offset + 4:offset + 8])
            offset = offset + 8 + length

        else:
            # Formato inesperado a partir daqui; para não corromper
            # o restante, encerra o parsing aqui (mantém o que já
            # foi extraído até agora).
            break

    return bytes(output)


def download_period(start, end, channel=CHANNEL):
    """
    Conecta ao NVR, lista as gravações entre start/end (strings
    "YYYY-MM-DD HH:MM:SS") e baixa+converte cada uma para
    OUTPUT_DIR. Retorna o número de vídeos convertidos.
    """

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cam = DVRIPCam(NVR_IP, user=USERNAME, password=PASSWORD)

    print("Conectando ao NVR...")
    if not cam.login():
        print("ERRO: login falhou.")
        return 0

    print(f"Listando gravações de {start} a {end}...")
    files = cam.list_local_files(start, end, "h264", channel=channel)
    print(f"Encontrados {len(files)} arquivos.")

    if not files:
        print("Nenhuma gravação encontrada no período.")
        cam.close()
        return 0

    converted_count = 0

    for i, f in enumerate(files, start=1):
        begin = f["BeginTime"]
        file_end = f["EndTime"]
        remote_name = f["FileName"]

        safe_name = begin.replace(":", "").replace(" ", "_").replace("-", "")
        raw_path = os.path.join(RAW_DIR, f"{safe_name}.raw")
        output_path = os.path.join(OUTPUT_DIR, f"{safe_name}.h264")

        if os.path.exists(output_path):
            print(f"[{i}/{len(files)}] {begin} -> já existe, pulando.")
            continue

        print(f"[{i}/{len(files)}] Baixando {begin} - {file_end}...")

        try:
            cam.download_file(begin, file_end, remote_name, raw_path)
        except Exception as exc:
            print(f"  ERRO ao baixar: {exc}")
            continue

        with open(raw_path, "rb") as fh:
            raw_data = fh.read()

        video_data = strip_private_headers(raw_data)

        with open(output_path, "wb") as fh:
            fh.write(video_data)

        os.remove(raw_path)

        converted_count += 1
        print(f"  OK -> {output_path} ({len(video_data)} bytes)")

    cam.close()

    print()
    print(f"Concluído. {converted_count} vídeos convertidos em {OUTPUT_DIR}.")

    return converted_count


def main():
    download_period(START, END)
    print("Próximo passo: python extract_dataset_from_videos.py")


if __name__ == "__main__":
    main()
