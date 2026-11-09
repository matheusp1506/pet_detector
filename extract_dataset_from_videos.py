"""
Extrai frames para o dataset de fine-tuning a partir de vídeos
já baixados/exportados do NVR (via interface web), em vez de
esperar o pet_detector.py rodar ao vivo por dias.

Para cada vídeo em VIDEOS_DIR:
  1. Recorta a metade "Principal" (câmera composta), igual o
     pet_detector.py faz.
  2. Roda o YOLO genérico (COCO) na ROI, frame a frame.
  3. Salva em dataset/raw/ apenas os frames onde algo foi
     detectado (cat/dog), com cooldown para não repetir demais.

Como o modelo genérico ainda confunde cat/dog entre os 4 pets,
os frames salvos aqui ainda vão precisar ser anotados/corrigidos
manualmente no LabelImg com o nome certo do pet (ligeirinho,
gatao, mia, luke) — mas ao menos filtra automaticamente os
trechos "vazios" (sem nenhum pet), poupando trabalho manual.

Uso:
    1. Baixe os vídeos do NVR (playback -> exportar) e coloque
       em dataset/videos/ (qualquer formato que o OpenCV/ffmpeg
       leia: mp4, avi, etc).
    2. python extract_dataset_from_videos.py
"""

import os
import time

import cv2
from ultralytics import YOLO

VIDEOS_DIR = "dataset/videos"
OUTPUT_DIR = "dataset/raw"

MODEL_NAME = "yolo11m.pt"

# Classes COCO genéricas: 15 = cat, 16 = dog.
PET_CLASSES = [15, 16]
CONFIDENCE = 0.25

# Mesma câmera composta do pet_detector.py.
COMPOSITE_SPLIT_Y = 1296

# Mesma ROI calibrada do pet_detector.py.
ROI = (1700, 670, 1900, 1080)

# Intervalo mínimo (segundos de vídeo) entre capturas, para não
# salvar dezenas de frames quase idênticos do mesmo evento.
CAPTURE_COOLDOWN_SECONDS = 2


def extract_from_video(model, video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"  ERRO: não foi possível abrir {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 15
    frame_interval = max(1, int(fps * 0.5))  # avalia 2x por segundo

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    frame_idx = 0
    saved_count = 0
    last_capture_time = -CAPTURE_COOLDOWN_SECONDS

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        video_time = frame_idx / fps

        if COMPOSITE_SPLIT_Y is not None:
            frame = frame[COMPOSITE_SPLIT_Y:, :]

        if ROI is not None:
            x1, y1, x2, y2 = ROI
            detect_frame = frame[y1:y2, x1:x2]
        else:
            detect_frame = frame

        results = model.predict(
            detect_frame,
            classes=PET_CLASSES,
            conf=CONFIDENCE,
            imgsz=960,
            verbose=False
        )

        has_detection = len(results[0].boxes) > 0

        if has_detection and (video_time - last_capture_time) >= CAPTURE_COOLDOWN_SECONDS:
            last_capture_time = video_time

            timestamp_ms = int(video_time * 1000)
            filename = os.path.join(
                OUTPUT_DIR,
                f"{video_name}_{timestamp_ms:08d}.jpg"
            )
            cv2.imwrite(filename, detect_frame)
            saved_count += 1

        frame_idx += 1

    cap.release()
    return saved_count


def extract_all(model=None):
    """
    Processa todos os vídeos em VIDEOS_DIR, salvando em
    OUTPUT_DIR os frames com detecção. Reaproveita `model` se
    for passado (evita recarregar o YOLO quando chamado em
    conjunto com outras etapas do pipeline); caso contrário,
    carrega o modelo genérico configurado em MODEL_NAME.

    Retorna o total de frames salvos.
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(VIDEOS_DIR):
        print(f"Pasta {VIDEOS_DIR} não existe.")
        print("Crie-a e coloque os vídeos exportados do NVR lá.")
        return 0

    videos = [
        f for f in os.listdir(VIDEOS_DIR)
        if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".h264"))
    ]

    if not videos:
        print(f"Nenhum vídeo encontrado em {VIDEOS_DIR}.")
        return 0

    if model is None:
        print("Carregando modelo YOLO...")
        model = YOLO(MODEL_NAME)
        print("Modelo carregado.")
        print()

    total_saved = 0
    start = time.time()

    for video in videos:
        video_path = os.path.join(VIDEOS_DIR, video)
        print(f"Processando {video}...")

        saved = extract_from_video(model, video_path)
        total_saved += saved

        print(f"  {saved} frames salvos.")

    elapsed = time.time() - start

    print()
    print(f"Concluído em {elapsed:.1f}s.")
    print(f"Total de frames salvos: {total_saved}")
    print(f"Pasta: {OUTPUT_DIR}")

    return total_saved


def main():
    extract_all()
    print()
    print("Próximo passo: anote as imagens com LabelImg, usando")
    print("os nomes corretos dos pets (o modelo genérico usado")
    print("aqui só filtrou 'tem algo aqui', não sabe distinguir")
    print("ligeirinho/gatao/mia/luke).")


if __name__ == "__main__":
    main()
