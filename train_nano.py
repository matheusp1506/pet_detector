"""
Fine-tuning do YOLO11n (nano) com o dataset anotado (dataset/),
para deploy em servidores sem GPU e com pouca RAM (ex: VPS
compartilhada rodando pet_detector_server.py).

yolo11n tem bem menos parâmetros que yolo11m (usado em train.py
para a estação local com GPU), então é mais rápido e leve em
CPU, ao custo de precisão um pouco menor — troca aceitável para
um servidor de produção com recursos limitados.

Pré-requisitos: mesmos de train.py (dataset já anotado e
organizado via dataset/split_dataset.py).

Uso:
    python train_nano.py

Gera runs/detect/trainN/weights/best.pt — copie esse arquivo
para o servidor (ver deploy/DEPLOY.md).
"""

from ultralytics import YOLO

BASE_MODEL = "yolo11n.pt"
DATA_YAML = "dataset/data.yaml"

EPOCHS = 100
IMG_SIZE = 960

BATCH_SIZE = 8


def main():
    model = YOLO(BASE_MODEL)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=20,
        workers=0,
    )

    print()
    print("Treino concluído.")
    print("Modelo nano salvo em: ver caminho 'Results saved to' acima.")
    print()
    print("Copie esse best.pt para o servidor (deploy/DEPLOY.md).")


if __name__ == "__main__":
    main()
