"""
Retoma um treino YOLO interrompido, a partir do último
checkpoint salvo (last.pt), continuando exatamente de onde
parou (mesma época, optimizer state, learning rate schedule).

Ajuste RUN_DIR abaixo se o treino estiver em uma pasta
runs/detect/trainN diferente.

Uso:
    python resume_train.py
"""

from ultralytics import YOLO

RUN_DIR = "runs/detect/train-3"


def main():
    checkpoint = f"{RUN_DIR}/weights/last.pt"

    print(f"Retomando treino a partir de {checkpoint}...")

    model = YOLO(checkpoint)
    model.train(resume=True)

    print()
    print("Treino concluído.")
    print(f"Modelo ajustado salvo em: {RUN_DIR}/weights/best.pt")


if __name__ == "__main__":
    main()
