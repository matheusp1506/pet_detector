#!/usr/bin/env bash
#
# Instala o pet_detector como serviço systemd em um servidor
# Ubuntu sem interface gráfica.
#
# Uso (a partir da pasta deploy/, com o projeto já copiado):
#   sudo ./install.sh
#
# Pré-requisitos: copie a pasta do projeto (incluindo
# pet_detector_server.py, best.pt e .env) para /opt/pet_detector
# no servidor antes de rodar este script. Exemplo, a partir da
# sua máquina de treino:
#
#   rsync -avz --exclude 'dataset/raw' --exclude 'dataset/videos' \
#     ./ usuario@servidor:/tmp/pet_detector_deploy/
#
#   # no servidor:
#   sudo mkdir -p /opt/pet_detector
#   sudo mv /tmp/pet_detector_deploy/* /opt/pet_detector/
#   cd /opt/pet_detector/deploy && sudo ./install.sh

set -euo pipefail

INSTALL_DIR="/opt/pet_detector"
SERVICE_USER="petdetector"

if [ "$EUID" -ne 0 ]; then
  echo "Rode como root (sudo ./install.sh)." >&2
  exit 1
fi

if [ ! -f "$INSTALL_DIR/pet_detector_server.py" ]; then
  echo "ERRO: $INSTALL_DIR/pet_detector_server.py não encontrado." >&2
  echo "Copie o projeto para $INSTALL_DIR antes de rodar este script." >&2
  exit 1
fi

echo "== Instalando dependências do sistema =="
apt-get update -qq
apt-get install -y --no-install-recommends \
  python3 python3-venv python3-pip ffmpeg libgl1 libglib2.0-0

echo "== Criando usuário de serviço (sem login, sem home) =="
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
  useradd --system --no-create-home --shell /usr/sbin/nologin "$SERVICE_USER"
fi

echo "== Ajustando dono dos arquivos =="
mkdir -p "$INSTALL_DIR/.config/Ultralytics"
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"

echo "== Criando ambiente virtual Python =="
if [ ! -d "$INSTALL_DIR/venv" ]; then
  sudo -u "$SERVICE_USER" python3 -m venv "$INSTALL_DIR/venv"
fi

echo "== Instalando dependências Python =="
sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install --upgrade pip -q

# ultralytics declara opencv-python (com GUI) como dependência
# obrigatória, mesmo em servidor headless. Instalamos tudo
# normalmente e depois forçamos a versão headless por cima, sem
# deixar o pip reintroduzir a versão com GUI via resolução de
# dependências (--no-deps).
sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install -q \
  -r "$INSTALL_DIR/deploy/requirements.txt"
sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" uninstall -y -q opencv-python opencv-python-headless
sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install -q --no-deps opencv-python-headless

if [ ! -f "$INSTALL_DIR/best.pt" ]; then
  echo "AVISO: $INSTALL_DIR/best.pt não encontrado — copie o" >&2
  echo "modelo treinado para lá antes de iniciar o serviço." >&2
fi

if [ ! -f "$INSTALL_DIR/.env" ]; then
  echo "AVISO: $INSTALL_DIR/.env não encontrado — copie um .env" >&2
  echo "com TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, NVR_PASSWORD," >&2
  echo "etc, antes de iniciar o serviço." >&2
fi

echo "== Instalando serviço systemd =="
cp "$INSTALL_DIR/deploy/pet-detector.service" /etc/systemd/system/pet-detector.service
systemctl daemon-reload
systemctl enable pet-detector.service

echo
echo "Instalação concluída."
echo
echo "Para iniciar agora:"
echo "  sudo systemctl start pet-detector"
echo
echo "Para acompanhar os logs:"
echo "  sudo journalctl -u pet-detector -f"
echo
echo "O serviço já está habilitado para iniciar no boot."
