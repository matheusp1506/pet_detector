"""
Pipeline integrado para coletar dataset de fine-tuning a partir
do histórico gravado no NVR, em 3 fases:

  1. DOWNLOAD  — baixa as gravações do período via protocolo
                 DVRIP (porta 34567), sem precisar exportar
                 manualmente pela interface web.
  2. CONVERSÃO — já embutida na fase de download: remove os
                 cabeçalhos privados do NVR e remonta um .h264
                 Annex-B puro, legível pelo OpenCV/ffmpeg.
  3. EXTRAÇÃO  — percorre os vídeos convertidos, roda o YOLO
                 genérico (COCO) na ROI calibrada, e salva em
                 dataset/raw/ só os frames onde algo foi
                 detectado (cat/dog).

Período padrão: apenas hoje (00:00 até agora). Para baixar um
dia específico (ex: um dia de cada vez ao longo da última
semana), passe a data como argumento:

    python build_dataset.py 2026-08-29

Isso baixa o dia inteiro (00:00 até 23:59:59) dessa data.

Uso:
    python build_dataset.py              # hoje, até agora
    python build_dataset.py 2026-08-29   # dia inteiro especificado

Para um período com hora específica, importe
download_period()/extract_all() diretamente.

Ao final, os frames captados ainda precisam ser anotados
manualmente no LabelImg com o nome correto de cada pet
(ligeirinho, gatao, mia, luke) — o modelo genérico usado aqui só
filtra "tem algo aqui", não distingue os indivíduos.
"""

import sys
import time
from datetime import datetime, timedelta

from ultralytics import YOLO

from dvr_download import download_period
from extract_dataset_from_videos import extract_all, MODEL_NAME


def compute_date_range(day_arg=None):
    """
    Sem argumento: hoje, 00:00 até agora.
    Com argumento "YYYY-MM-DD": esse dia inteiro, 00:00 até
    23:59:59.
    """

    if day_arg is None:
        now = datetime.now()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return (
            midnight.strftime("%Y-%m-%d %H:%M:%S"),
            now.strftime("%Y-%m-%d %H:%M:%S"),
        )

    day = datetime.strptime(day_arg, "%Y-%m-%d")
    start = day.replace(hour=0, minute=0, second=0)
    end = start + timedelta(days=1) - timedelta(seconds=1)

    return (
        start.strftime("%Y-%m-%d %H:%M:%S"),
        end.strftime("%Y-%m-%d %H:%M:%S"),
    )


START_DATE, END_DATE = compute_date_range(
    sys.argv[1] if len(sys.argv) > 1 else None
)


def main():
    print("=" * 60)
    print("FASE 1/3: Download das gravações do NVR")
    print("=" * 60)
    print(f"Período: {START_DATE} até {END_DATE}")
    print()

    start_time = time.time()

    downloaded = download_period(START_DATE, END_DATE)

    if downloaded == 0:
        print()
        print("Nenhum vídeo baixado. Encerrando (nada para extrair).")
        return

    download_elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("FASE 2/3: Conversão")
    print("=" * 60)
    print("Já concluída junto com o download (cabeçalhos privados")
    print("removidos automaticamente em dvr_download.py).")

    print()
    print("=" * 60)
    print("FASE 3/3: Extração de frames com detecção")
    print("=" * 60)

    print("Carregando modelo YOLO...")
    model = YOLO(MODEL_NAME)
    print("Modelo carregado.")
    print()

    extract_start = time.time()
    saved = extract_all(model=model)
    extract_elapsed = time.time() - extract_start

    total_elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("RESUMO")
    print("=" * 60)
    print(f"Vídeos baixados:      {downloaded}")
    print(f"Frames salvos:        {saved}")
    print(f"Tempo download:       {download_elapsed:.1f}s")
    print(f"Tempo extração:       {extract_elapsed:.1f}s")
    print(f"Tempo total:          {total_elapsed:.1f}s")
    print()
    print("Próximo passo: anote as imagens em dataset/raw/ com")
    print("LabelImg, usando os nomes corretos dos pets (ligeirinho,")
    print("gatao, mia, luke). Depois rode dataset/split_dataset.py")
    print("e train.py.")


if __name__ == "__main__":
    main()
