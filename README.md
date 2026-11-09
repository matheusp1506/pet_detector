# Pet Detector

🇬🇧 English · [🇧🇷 Português](#-português)

## Motivation

I was about to travel, and my cats are used to going outside. I was worried that one of them might not come back while I was away. I built this software so I could keep an eye on them remotely: it watches a camera at the cats' usual entry and exit point and sends me a Telegram photo every time one of them goes out or comes in. If a cat has been out for a suspiciously long time, I can ask a family member to go look for it.

## What it does

Watches an NVR camera stream with a fine-tuned YOLO model, tracks pets (two cats and a dog) and sends a Telegram alert with a photo when one crosses a virtual line in a monitored corridor ("ENTROU" = came in, "SAIU" = went out).

The software reports crossings only. It does not raise an alarm by itself when a pet stays away for too long; you read the alerts and judge the elapsed time yourself.

## Layout

| Path | Purpose |
|---|---|
| `pet_detector.py` | Desktop version (OpenCV window, optional dataset capture) |
| `pet_detector_server.py` | Headless service: auto-reconnect, logging, memory monitor |
| `dvr_download.py`, `extract_dataset_from_videos.py`, `build_dataset.py` | Build a training set from NVR recordings |
| `train.py`, `train_nano.py`, `resume_train.py` | YOLO fine-tuning (medium / nano / resume) |
| `roi_test.py`, `capture_bottom_half.py`, `test_*.py` | Camera calibration helpers |
| `deploy/` | systemd service + install guide ([DEPLOY.md](deploy/DEPLOY.md)) |

## Setup

```bash
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in Telegram + NVR credentials
```

All credentials are read from `.env` (never committed). Model weights (`*.pt`), datasets, training runs and camera snapshots are git-ignored; train your own with `train.py` or copy a `best.pt` next to the script (`MODEL_PATH`).

## Run

```bash
python pet_detector_server.py   # headless
python pet_detector.py          # with preview window
```

The ROI, crossing lines and composite-camera split are calibrated for one specific camera; edit the constants at the top of the scripts for yours.

---

# 🇧🇷 Português

[🇬🇧 English](#pet-detector) · 🇧🇷 Português

## Motivação

Eu ia viajar e meus gatos estão acostumados a sair de casa. Fiquei preocupado com a possibilidade de algum deles não voltar enquanto eu estivesse fora. Criei este software para acompanhá-los à distância: ele monitora uma câmera no ponto por onde os gatos costumam entrar e sair e me envia uma foto pelo Telegram toda vez que um deles sai ou volta. Se um gato ficar fora por um tempo muito longo, posso pedir a um familiar que vá procurá-lo.

## O que faz

Monitora o stream de uma câmera ligada a um NVR com um modelo YOLO ajustado (fine-tuned), rastreia os pets (dois gatos e um cachorro) e envia um alerta no Telegram, com foto, quando um deles cruza uma linha virtual em um corredor monitorado ("ENTROU" / "SAIU").

O software avisa apenas os cruzamentos. Ele não dispara um alarme sozinho quando um pet demora demais para voltar; quem lê os alertas e avalia o tempo decorrido é você.

## Estrutura

| Caminho | Função |
|---|---|
| `pet_detector.py` | Versão desktop (janela OpenCV, captura opcional de dataset) |
| `pet_detector_server.py` | Serviço sem interface gráfica: reconexão automática, logs, monitor de memória |
| `dvr_download.py`, `extract_dataset_from_videos.py`, `build_dataset.py` | Montam um dataset de treino a partir das gravações do NVR |
| `train.py`, `train_nano.py`, `resume_train.py` | Fine-tuning do YOLO (médio / nano / retomar) |
| `roi_test.py`, `capture_bottom_half.py`, `test_*.py` | Utilitários de calibração da câmera |
| `deploy/` | Serviço systemd + guia de instalação ([DEPLOY.md](deploy/DEPLOY.md)) |

## Instalação

```bash
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # preencha as credenciais do Telegram e do NVR
```

Todas as credenciais são lidas do `.env` (nunca versionado). Pesos de modelo (`*.pt`), datasets, resultados de treino e fotos da câmera estão no `.gitignore`; treine o seu com `train.py` ou copie um `best.pt` para junto do script (`MODEL_PATH`).

## Execução

```bash
python pet_detector_server.py   # sem interface gráfica
python pet_detector.py          # com janela de visualização
```

A ROI, as linhas de cruzamento e o corte da câmera composta foram calibrados para uma câmera específica; edite as constantes no topo dos scripts para a sua.
