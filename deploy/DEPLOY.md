# Deploy do Pet Detector em servidor Ubuntu (headless)

Roda `pet_detector_server.py` como serviço systemd: inicia no
boot, reinicia sozinho se cair, sem GUI, sem salvar imagens em
disco — só monitora e envia alertas via Telegram.

## 1. Preparar os arquivos no servidor

No servidor Ubuntu, como usuário com sudo:

```bash
sudo mkdir -p /opt/pet_detector
```

Da sua máquina de treino, copie o necessário (sem o dataset
bruto nem os vídeos, que não fazem sentido em produção):

```bash
rsync -avz \
  --exclude 'dataset/' \
  --exclude 'runs/' \
  --exclude '*.mp4' --exclude '*.h264' \
  ./pet_detector_server.py \
  ./deploy \
  usuario@servidor:/tmp/pet_detector_deploy/
```

Copie também o modelo treinado (só o `best.pt` final, não a
pasta `runs/` inteira):

```bash
scp runs/detect/train-5/weights/best.pt \
  usuario@servidor:/tmp/pet_detector_deploy/best.pt
```

No servidor:

```bash
sudo mv /tmp/pet_detector_deploy/* /opt/pet_detector/
```

## 2. Configurar credenciais

```bash
sudo cp /opt/pet_detector/deploy/.env.example /opt/pet_detector/.env
sudo nano /opt/pet_detector/.env
```

Preencha `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` e
`NVR_PASSWORD` no mínimo.

## 3. Instalar

```bash
cd /opt/pet_detector/deploy
sudo ./install.sh
```

Isso instala as dependências do sistema (ffmpeg, libs de vídeo
headless), cria um usuário de serviço sem privilégios
(`petdetector`), monta um virtualenv Python, e registra o
serviço systemd habilitado para iniciar no boot.

## 4. Iniciar e verificar

```bash
sudo systemctl start pet-detector
sudo systemctl status pet-detector
sudo journalctl -u pet-detector -f
```

Você deve ver as linhas de log: carregamento do modelo, conexão
com o NVR, e depois os alertas de entrada/saída conforme os
pets passarem.

## Operação do dia a dia

- **Reiniciar após mudar `.env` ou o modelo**:
  `sudo systemctl restart pet-detector`
- **Parar**: `sudo systemctl stop pet-detector`
- **Desabilitar do boot**: `sudo systemctl disable pet-detector`
- **Ver logs recentes**: `sudo journalctl -u pet-detector -n 200`
- **Ver logs ao vivo**: `sudo journalctl -u pet-detector -f`

## Auto-recuperação — o que já está coberto

- **Processo crasha** (exceção não tratada, etc): o systemd
  reinicia automaticamente (`Restart=always`, `RestartSec=10`).
- **Servidor reinicia**: o serviço sobe sozinho no boot
  (`systemctl enable` já foi feito pelo `install.sh`).
- **RTSP cai ou trava** (sem frame novo por 30s): o próprio
  script detecta e reconecta sozinho, sem precisar do systemd.
- **Falha de configuração** (ex: senha errada — o processo sai
  imediato com erro): o systemd tenta de novo a cada 10s; se o
  problema for persistente, os logs (`journalctl`) vão mostrar
  o mesmo erro se repetindo — é o sinal de que precisa de
  intervenção manual (corrigir `.env` e `restart`).

## Atualizando o modelo depois de re-treinar

```bash
scp runs/detect/train-N/weights/best.pt usuario@servidor:/tmp/best.pt
ssh usuario@servidor 'sudo mv /tmp/best.pt /opt/pet_detector/best.pt && \
  sudo chown petdetector:petdetector /opt/pet_detector/best.pt && \
  sudo systemctl restart pet-detector'
```
