"""
Fine-tuning do YOLO11 com o dataset anotado (dataset/), partindo
do modelo pré-treinado yolo11m.pt.

Pré-requisitos:
    1. Imagens capturadas em dataset/raw/ (pet_detector.py já
       faz isso automaticamente com DATASET_CAPTURE_ENABLED).
    2. Anotadas com LabelImg (formato YOLO), gerando um .txt
       para cada .jpg em dataset/raw/.
    3. Rodar dataset/split_dataset.py para organizar em
       images/train, images/val, labels/train, labels/val.

Uso:
    python train.py

Gera runs/detect/train/weights/best.pt — esse é o modelo
ajustado ao seu quintal/câmera. Para usá-lo no pet_detector.py,
troque MODEL_NAME para o caminho desse arquivo.
"""

from ultralytics import YOLO

BASE_MODEL = "yolo11m.pt"
DATA_YAML = "dataset/data.yaml"

EPOCHS = 100
IMG_SIZE = 960

# Batch pequeno por padrão — ajuste para cima se sua GPU/CPU
# aguentar mais sem estourar memória.
BATCH_SIZE = 8


def main():
    model = YOLO(BASE_MODEL)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=20,
        # workers=8 (padrão) usa multiprocessing para carregar
        # dados, o que no Windows tem overhead severo de
        # inicialização por worker (cada um reimporta o processo
        # do zero) — pode travar/demorar muito com paths longos
        # (ex: dentro de uma pasta do OneDrive). workers=0 roda
        # o carregamento na thread principal, mais lento por
        # imagem mas sem esse overhead de startup.
        workers=0,
    )

    print()
    print("Treino concluído.")
    print("Modelo ajustado salvo em: runs/detect/train/weights/best.pt")
    print()
    print("Para usar, edite pet_detector.py:")
    print('  MODEL_NAME = "runs/detect/train/weights/best.pt"')


if __name__ == "__main__":
    main()
