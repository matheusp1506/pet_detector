"""
Versão headless (sem GUI) do pet_detector.py, para rodar como
serviço em um servidor Linux sem ambiente gráfico.

Diferenças em relação ao pet_detector.py original:
  - Sem cv2.imshow/namedWindow/waitKey — não depende de X11.
  - Sem captura de dataset em disco (DATASET_CAPTURE_ENABLED
    removido) — este servidor só monitora e alerta, não coleta
    dados de treino.
  - Log estruturado via módulo `logging`, para o systemd/
    journalctl capturar corretamente (em vez de print com \\r).
  - Reconecta automaticamente ao RTSP se a conexão cair (loop
    externo em torno de connect_and_run), em vez de encerrar.

Uso:
    python3 pet_detector_server.py

Para rodar como serviço systemd (start no boot + auto-restart
em caso de crash), veja pet-detector.service e o guia de
instalação em DEPLOY.md.
"""

import logging
import os
import signal
import sys
import threading
import time
from collections import deque

import cv2
import requests
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("pet_detector")

# Taxa mínima de leitura do stream. A fila guarda até
# QUEUE_SECONDS segundos de frames nessa taxa; se o consumidor
# (loop de detecção) for mais lento que isso por tempo demais,
# os frames mais ANTIGOS da fila são descartados primeiro.
#
# Cada frame é a imagem completa da câmera composta (~18MB antes
# do split/crop de ROI). Em CPU (sem GPU), a inferência é mais
# lenta que a chegada de frames, então a fila enche e fica cheia
# — QUEUE_SECONDS alto vira um buffer de memória permanente, não
# um amortecedor ocasional. Mantido baixo por padrão (1 frame,
# equivalente a "sempre pegar o mais recente"); ajustável via env
# var apenas se a máquina tiver memória de sobra.
STREAM_FPS = 15
QUEUE_SECONDS = float(os.getenv("QUEUE_SECONDS", "0.1"))
QUEUE_MAXLEN = max(1, int(STREAM_FPS * QUEUE_SECONDS))

# Se a conexão RTSP cair ou não vier frame algum por esse tempo
# (segundos), o processo reconecta do zero.
STREAM_STALL_TIMEOUT = 30

# Intervalo entre tentativas de reconexão após uma falha.
RECONNECT_DELAY_SECONDS = 5


class StreamReader:
    """
    Lê o stream RTSP em uma thread separada e enfileira os
    frames capturados, para que o loop de detecção processe
    todos eles (sem pular nenhum), em vez de só o mais recente.
    """

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.queue = deque(maxlen=QUEUE_MAXLEN)
        self.ret = False
        self.last_frame_time = time.time()
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        min_interval = 1.0 / STREAM_FPS

        while self.running:
            start = time.time()

            ret, frame = self.cap.read()

            with self.lock:
                self.ret = ret
                if ret:
                    self.queue.append(frame)
                    self.last_frame_time = time.time()

            if not ret:
                time.sleep(0.1)
                continue

            elapsed = time.time() - start
            remaining = min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def read(self):
        with self.lock:
            if self.queue:
                return True, self.queue.popleft()
            return self.ret, None

    def seconds_since_last_frame(self):
        with self.lock:
            return time.time() - self.last_frame_time

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)


# ============================================================
# CONFIGURAÇÃO DO NVR
# ============================================================

NVR_IP = os.getenv("NVR_IP", "")
NVR_PORT = int(os.getenv("NVR_PORT", "554"))

USERNAME = os.getenv("NVR_USERNAME", "")
PASSWORD = os.getenv("NVR_PASSWORD", "")

CHANNEL = int(os.getenv("NVR_CHANNEL", "1"))
STREAM = int(os.getenv("NVR_STREAM", "0"))


def build_rtsp_url():
    return (
        f"rtsp://{USERNAME}:{PASSWORD}@"
        f"{NVR_IP}:{NVR_PORT}"
        f"/user={USERNAME}&password={PASSWORD}"
        f"&channel={CHANNEL}&stream={STREAM}.sdp?real_stream"
    )


# ============================================================
# CONFIGURAÇÃO DO YOLO
# ============================================================

# Caminho do modelo fine-tuned. No servidor, copie best.pt para
# junto deste script (ou ajuste o caminho / defina via env var
# MODEL_PATH).
MODEL_NAME = os.getenv("MODEL_PATH", "best.pt")

# Classes monitoradas, por índice do modelo carregado.
#   0 = ligeirinho (gato frajola)
#   1 = gatao (gato laranja com branco)
#   2 = mia (gata preta) — EXCLUÍDA: sem exemplos de treino.
#   3 = luke (cachorro caramelo pequeno)
PET_CLASSES = [0, 1, 3]

# Resolução de entrada da inferência. Servidor roda em CPU com
# pouca RAM (VPS compartilhada) — 640 usa bem menos memória e
# processa mais rápido que 960 (usado na estação de treino com
# GPU), ao custo de um pouco de precisão em objetos pequenos ou
# distantes. Ajustável via env var se precisar recalibrar.
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))

CONFIDENCE = float(os.getenv("CONFIDENCE", "0.6"))

# ============================================================
# CÂMERA COMPOSTA (DUAS IMAGENS EMPILHADAS)
# ============================================================

COMPOSITE_SPLIT_Y = 1296

# ============================================================
# REGIÃO DE INTERESSE (ROI)
# ============================================================

ROI = (1700, 670, 1900, 1080)

# ============================================================
# LINHA DE CRUZAMENTO (ALERTA)
# ============================================================

CROSS_LINE_X = 1800
CROSS_LINE_Y = 875
CROSS_MARGIN = 5
ALERT_COOLDOWN_SECONDS = 15


# ============================================================
# CONFIGURAÇÃO DO TELEGRAM
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


# Largura máxima da imagem enviada nos alertas do Telegram
# (mantém proporção). O frame original é bem maior que o
# necessário para visualização em um celular; reduzir também
# deixa o envio mais rápido em conexões lentas. Defina como 0
# para desativar o redimensionamento e enviar em resolução
# original.
ALERT_IMAGE_MAX_WIDTH = int(os.getenv("ALERT_IMAGE_MAX_WIDTH", "800"))


def resize_for_alert(image):
    if ALERT_IMAGE_MAX_WIDTH <= 0:
        return image

    height, width = image.shape[:2]
    if width <= ALERT_IMAGE_MAX_WIDTH:
        return image

    scale = ALERT_IMAGE_MAX_WIDTH / width
    new_size = (ALERT_IMAGE_MAX_WIDTH, int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


# ============================================================
# MONITORAMENTO DE MEMÓRIA
# ============================================================
#
# O processo (PyTorch + Ultralytics + OpenCV em CPU) tem um
# patamar de uso de memória de regime de ~650-700MB medido via
# cgroup (memory.current — a mesma métrica que o systemd usa
# para aplicar MemoryHigh=900M/MemoryMax=1000M, ver
# deploy/pet-detector.service). Este monitor avisa via Telegram
# se esse uso ultrapassar um limiar, sinal de que algo mudou
# (vazamento, pico de carga) e vale investigar antes do systemd
# matar o processo por MemoryMax.

MEMORY_CHECK_INTERVAL_SECONDS = int(
    os.getenv("MEMORY_CHECK_INTERVAL_SECONDS", "60")
)
MEMORY_ALERT_THRESHOLD_MB = int(
    os.getenv("MEMORY_ALERT_THRESHOLD_MB", "850")
)
# Não repete o alerta de memória alta com mais frequência que
# isso, mesmo que a checagem periódica continue detectando o
# problema — evita floodar o Telegram enquanto a causa não é
# resolvida.
MEMORY_ALERT_COOLDOWN_SECONDS = int(
    os.getenv("MEMORY_ALERT_COOLDOWN_SECONDS", "1800")
)

# Ignora checagens de memória nos primeiros segundos após o
# processo subir, para não confundir o pico natural de
# carregamento/aquecimento do modelo com um problema real.
MEMORY_ALERT_GRACE_PERIOD_SECONDS = int(
    os.getenv("MEMORY_ALERT_GRACE_PERIOD_SECONDS", "120")
)


def get_rss_mb():
    """
    Lê o uso de memória do processo em MB, na mesma métrica que
    o systemd usa para aplicar MemoryMax/MemoryHigh (cgroup v2
    memory.current) — não o VmRSS de /proc/self/status.

    Os dois discordam bastante em produção (VmRSS soma páginas
    residentes compartilhadas/mapeadas de bibliotecas do
    PyTorch/OpenCV que ficam bem acima do que o cgroup cobra do
    serviço), o que fazia esse monitor alertar com um número que
    nunca refletia o risco real do systemd matar o processo.
    Cai de volta para VmRSS se o cgroup não estiver disponível
    (ex: rodando fora do Linux, ou sem cgroup v2).
    """

    try:
        with open("/proc/self/cgroup", "r") as f:
            cgroup_path = f.read().strip().split(":")[-1]

        memory_current_path = f"/sys/fs/cgroup{cgroup_path}/memory.current"
        with open(memory_current_path, "r") as f:
            return int(f.read().strip()) / 1024 / 1024
    except (FileNotFoundError, ValueError, IndexError, OSError):
        pass

    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024
    except (FileNotFoundError, ValueError, IndexError):
        return None

    return None


def memory_monitor_loop():
    """
    Roda em background durante toda a vida do processo,
    checando periodicamente o uso de memória e alertando via
    Telegram se ultrapassar MEMORY_ALERT_THRESHOLD_MB.
    """

    # Logo após subir, o processo tem um pico natural de memória
    # (carregamento do modelo YOLO, primeira inferência aquecendo
    # buffers internos do PyTorch/OpenCV) que pode passar de 1GB
    # por ~1 minuto antes de assentar no patamar de regime
    # (~600-700MB). Sem essa carência, o primeiro check já cai
    # nesse pico e dispara um alerta de falso positivo toda vez
    # que o serviço reinicia.
    started_at = time.time()
    last_alert_time = 0

    while not _shutdown_requested:
        time.sleep(MEMORY_CHECK_INTERVAL_SECONDS)

        if time.time() - started_at < MEMORY_ALERT_GRACE_PERIOD_SECONDS:
            continue

        rss_mb = get_rss_mb()
        if rss_mb is None:
            continue

        log.debug(f"Uso de memória atual: {rss_mb:.0f}MB")

        if rss_mb < MEMORY_ALERT_THRESHOLD_MB:
            continue

        now = time.time()
        if now - last_alert_time < MEMORY_ALERT_COOLDOWN_SECONDS:
            continue

        last_alert_time = now

        message = (
            f"⚠️ Uso de memória alto: {rss_mb:.0f}MB "
            f"(limiar: {MEMORY_ALERT_THRESHOLD_MB}MB). "
            f"Verifique o servidor (journalctl -u pet-detector)."
        )
        log.warning(message)
        send_telegram_alert(message)


# Rajadas de detecção (vários track IDs cruzando a linha em
# poucos minutos, ex: o pet saindo e voltando do campo de visão
# repetidamente) podiam disparar muitas threads de envio ao
# Telegram ao mesmo tempo, cada uma seguranco sua própria cópia
# do frame + buffer JPEG na memória até o POST completar (até
# ALERT_SEND_TIMEOUT_SECONDS). Isso causava picos de memória
# correlacionados com essas rajadas. Limitamos quantos envios
# rodam ao mesmo tempo; o excesso apenas espera a vez.
ALERT_SEND_TIMEOUT_SECONDS = 10
_alert_send_semaphore = threading.Semaphore(2)


def send_telegram_alert(message, image=None):
    """
    Envia uma notificação para o Telegram em uma thread
    separada, para não travar o loop de detecção.
    """

    if not TELEGRAM_ENABLED:
        return

    def _send():
        with _alert_send_semaphore:
            try:
                if image is not None:
                    resized = resize_for_alert(image)
                    ok, buffer = cv2.imencode(".jpg", resized)
                    if ok:
                        url = (
                            f"https://api.telegram.org/bot"
                            f"{TELEGRAM_BOT_TOKEN}/sendPhoto"
                        )
                        requests.post(
                            url,
                            data={
                                "chat_id": TELEGRAM_CHAT_ID,
                                "caption": message
                            },
                            files={
                                "photo": (
                                    "alerta.jpg",
                                    buffer.tobytes(),
                                    "image/jpeg"
                                )
                            },
                            timeout=ALERT_SEND_TIMEOUT_SECONDS
                        )
                        return

                url = (
                    f"https://api.telegram.org/bot"
                    f"{TELEGRAM_BOT_TOKEN}/sendMessage"
                )
                requests.post(
                    url,
                    data={
                        "chat_id": TELEGRAM_CHAT_ID,
                        "text": message
                    },
                    timeout=ALERT_SEND_TIMEOUT_SECONDS
                )

            except requests.RequestException as exc:
                log.error(f"Erro ao enviar alerta Telegram: {exc}")

    threading.Thread(target=_send, daemon=True).start()


def check_crossing(
    axis,
    side,
    track_id,
    class_name,
    last_side,
    last_alert_time,
    annotated_crop
):
    """
    Verifica se o pet cruzou a linha de um eixo ("x" ou "y") e,
    se sim (respeitando o cooldown), dispara o alerta Telegram.
    """

    key = (track_id, axis)
    previous_side = last_side.get(key)

    if (
        side is not None
        and previous_side is not None
        and side != previous_side
    ):

        now = time.time()
        last_time = last_alert_time.get(key, 0)

        if now - last_time >= ALERT_COOLDOWN_SECONDS:

            last_alert_time[key] = now

            if axis == "x":
                direction = "ENTROU" if previous_side == "left" else "SAIU"
            else:
                direction = "SAIU" if previous_side == "top" else "ENTROU"

            message = (
                f"{class_name.capitalize()} "
                f"(ID {track_id}) {direction} "
                f"pela área monitorada."
            )

            log.info(message)

            send_telegram_alert(message, image=annotated_crop)

    if side is not None:
        last_side[key] = side


# Sinalizador global para desligamento limpo via SIGTERM
# (usado pelo systemd ao parar o serviço).
_shutdown_requested = False


def _handle_shutdown_signal(signum, frame):
    global _shutdown_requested
    log.info(f"Sinal {signum} recebido, encerrando...")
    _shutdown_requested = True


def connect_and_run(model):
    """
    Conecta ao NVR e roda o loop de detecção até a conexão
    cair, o stream travar (sem frames por STREAM_STALL_TIMEOUT
    segundos), ou um shutdown ser solicitado. Retorna
    normalmente em qualquer desses casos, para o chamador
    decidir se reconecta.
    """

    rtsp_url = build_rtsp_url()

    log.info(f"Conectando ao NVR {NVR_IP}:{NVR_PORT} (canal {CHANNEL})...")

    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        log.error("Não foi possível abrir o stream RTSP.")
        cap.release()
        return

    log.info("Câmera conectada.")

    stream = StreamReader(cap)

    last_side = {}
    last_alert_time = {}

    try:
        while not _shutdown_requested:

            if stream.seconds_since_last_frame() > STREAM_STALL_TIMEOUT:
                log.warning(
                    f"Nenhum frame recebido nos últimos "
                    f"{STREAM_STALL_TIMEOUT}s. Reconectando..."
                )
                return

            ret, frame = stream.read()

            if not ret:
                time.sleep(0.1)
                continue

            if frame is None:
                time.sleep(0.01)
                continue

            if COMPOSITE_SPLIT_Y is not None:
                frame = frame[COMPOSITE_SPLIT_Y:, :]

            if ROI is not None:
                x1, y1, x2, y2 = ROI
                detect_frame = frame[y1:y2, x1:x2]
            else:
                x1, y1 = 0, 0
                detect_frame = frame

            results = model.track(
                detect_frame,
                persist=True,
                classes=PET_CLASSES,
                conf=CONFIDENCE,
                imgsz=IMG_SIZE,
                verbose=False
            )

            result = results[0]

            # Recorte anotado (com caixas do YOLO), usado apenas
            # para anexar aos alertas do Telegram — não é salvo
            # em disco nem exibido em janela nenhuma. Envia só a
            # área da ROI (bem menor que o frame composto
            # completo) para manter o custo de memória por
            # alerta baixo, especialmente sob rajadas de
            # detecção com vários envios simultâneos.
            annotated_crop = result.plot()

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]

                    if box.id is not None:
                        track_id = int(box.id[0])
                    else:
                        track_id = -1

                    log.debug(
                        f"Pet: {class_name} ID: {track_id} "
                        f"Conf: {confidence:.2f}"
                    )

                    if track_id != -1:

                        bx1, by1, bx2, by2 = box.xyxy[0].tolist()

                        center_x = x1 + (bx1 + bx2) / 2
                        center_y = y1 + (by1 + by2) / 2

                        if center_x < CROSS_LINE_X - CROSS_MARGIN:
                            side_x = "left"
                        elif center_x > CROSS_LINE_X + CROSS_MARGIN:
                            side_x = "right"
                        else:
                            side_x = None

                        check_crossing(
                            "x", side_x, track_id, class_name,
                            last_side, last_alert_time, annotated_crop
                        )

                        if CROSS_LINE_Y is not None:

                            if center_y < CROSS_LINE_Y - CROSS_MARGIN:
                                side_y = "top"
                            elif center_y > CROSS_LINE_Y + CROSS_MARGIN:
                                side_y = "bottom"
                            else:
                                side_y = None

                            check_crossing(
                                "y", side_y, track_id, class_name,
                                last_side, last_alert_time, annotated_crop
                            )

    finally:
        stream.stop()
        cap.release()


def main():
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    signal.signal(signal.SIGINT, _handle_shutdown_signal)

    if not PASSWORD:
        log.error(
            "NVR_PASSWORD não definida (via .env ou variável de "
            "ambiente). Encerrando."
        )
        sys.exit(1)

    log.info("Carregando modelo YOLO...")
    model = YOLO(MODEL_NAME)
    log.info(f"Modelo carregado. Classes: {model.names}")

    if TELEGRAM_ENABLED:
        log.info("Alertas Telegram: ATIVADOS")
    else:
        log.warning(
            "Alertas Telegram: DESATIVADOS "
            "(defina TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID)"
        )

    threading.Thread(target=memory_monitor_loop, daemon=True).start()

    while not _shutdown_requested:
        try:
            connect_and_run(model)
        except Exception:
            log.exception("Erro inesperado no loop de detecção.")

        if _shutdown_requested:
            break

        log.info(
            f"Tentando reconectar em {RECONNECT_DELAY_SECONDS}s..."
        )
        time.sleep(RECONNECT_DELAY_SECONDS)

    log.info("Encerrado.")


if __name__ == "__main__":
    main()
