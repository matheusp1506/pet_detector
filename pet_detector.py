import os
import cv2
import threading
import time
import requests
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# Taxa mínima de leitura do stream. A fila guarda até
# QUEUE_SECONDS segundos de frames nessa taxa; se o consumidor
# (loop de detecção) for mais lento que isso por tempo demais,
# os frames mais ANTIGOS da fila são descartados primeiro, para
# não perder pets que passam rápido perto do "agora" e para o
# atraso não crescer sem limite.
STREAM_FPS = 15
QUEUE_SECONDS = 3
QUEUE_MAXLEN = STREAM_FPS * QUEUE_SECONDS


class StreamReader:
    """
    Lê o stream RTSP em uma thread separada e enfileira os
    frames capturados, para que o loop de detecção processe
    todos eles (sem pular nenhum), em vez de só o mais recente.

    A fila tem tamanho limitado (QUEUE_MAXLEN): se a detecção
    for mais lenta que a chegada de frames, os mais antigos vão
    sendo descartados para manter o atraso máximo previsível.
    """

    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.queue = deque(maxlen=QUEUE_MAXLEN)
        self.ret = False
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

            if not ret:
                time.sleep(0.1)
                continue

            elapsed = time.time() - start
            remaining = min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def read(self):
        """
        Retorna o próximo frame da fila (o mais antigo ainda não
        consumido), em ordem, ou (False, None) se não houver
        nenhum disponível no momento.
        """
        with self.lock:
            if self.queue:
                return True, self.queue.popleft()
            return self.ret, None

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

CHANNEL = 1

# Stream principal
STREAM = 0


# ============================================================
# URL RTSP
# ============================================================

RTSP_URL = (
    f"rtsp://{USERNAME}:{PASSWORD}@"
    f"{NVR_IP}:{NVR_PORT}"
    f"/user={USERNAME}&password={PASSWORD}"
    f"&channel={CHANNEL}&stream={STREAM}.sdp?real_stream"
)


# ============================================================
# CONFIGURAÇÃO DO YOLO
# ============================================================

# Modelo fine-tuned (treinado em train.py com o dataset dos 4
# pets — ver dataset/data.yaml). Substitui o COCO genérico.
MODEL_NAME = "runs/detect/train-5/weights/best.pt"

# Classes monitoradas, por índice do modelo carregado.
#
# Modelo genérico COCO (yolo11m.pt): classes fixas do dataset.
#   15 = cat, 16 = dog (não distingue indivíduos)
#
# Modelo fine-tuned (best.pt, ver dataset/data.yaml): classes
# por indivíduo.
#   0 = ligeirinho (gato frajola)
#   1 = gatao (gato laranja com branco)
#   2 = mia (gata preta) — EXCLUÍDA: sem exemplos no dataset de
#       treino, o modelo nunca aprendeu a reconhecê-la.
#   3 = luke (cachorro caramelo pequeno)
#
# Ajuste PET_CLASSES conforme o modelo em uso acima.

PET_CLASSES = [0, 1, 3]  # modelo fine-tuned (mia excluída)

# Confiança mínima
#
# 0.15 gerava muitos falsos positivos com o modelo COCO
# genérico. Com o modelo fine-tuned (best.pt), o dataset ainda é
# pequeno e desbalanceado (72% das caixas são do Gatão, 0 da
# Mia), o que causa overfitting: objetos inanimados na área
# monitorada às vezes são classificados como "gatao" com
# confiança alta. Subido para 0.6 como mitigação temporária;
# a correção real é ampliar o dataset (mais dias, incluindo
# negativos) e re-treinar do zero.
CONFIDENCE = 0.6

# ============================================================
# CÂMERA COMPOSTA (DUAS IMAGENS EMPILHADAS)
# ============================================================
#
# Esta câmera é composta: cada frame RTSP traz duas imagens
# empilhadas verticalmente (dois sensores/lentes em um só
# corpo). A metade de baixo ("Principal") é a que enquadra o
# pátio; a de cima é outra vista que não usamos.
#
# Frame completo: 2304x2592. Split exato na metade: y=1296.
#
# Defina COMPOSITE_SPLIT_Y = None se a câmera não for composta
# (frame único, sem split).

COMPOSITE_SPLIT_Y = 1296

# ============================================================
# REGIÃO DE INTERESSE (ROI)
# ============================================================
#
# Recorta apenas a área do pátio/garagem antes de rodar o
# YOLO. Isso permite manter alta resolução relativa ao
# tamanho do gato, sem pagar o custo computacional de
# processar a imagem inteira (ruas, telhados, etc).
#
# Coordenadas em pixels, já relativas à metade de baixo do
# frame (ou seja, y=0 aqui corresponde a y=COMPOSITE_SPLIT_Y
# no frame original). Ajuste conforme o enquadramento do
# pátio na câmera.
#
# Defina ROI = None para desativar e usar a metade inteira.
#
# Calibrado a partir de bottom_half.jpg: corredor estreito
# entre o pátio e a parede de tijolos, próximo à entrada da
# casa — caminho que os gatos usam para pular no muro.

ROI = (1700, 670, 1900, 1080)

# ============================================================
# LINHA DE CRUZAMENTO (ALERTA)
# ============================================================
#
# Linha vertical no canto direito da ROI. Quando um pet
# (identificado pelo tracker) cruza essa linha, um alerta é
# disparado via Telegram.
#
# Coordenada X em pixels da imagem ORIGINAL (mesmo sistema
# da ROI). Deve estar entre ROI[0] e ROI[2].
#
# Cruzar da esquerda para a direita = "entrou"
# Cruzar da direita para a esquerda = "saiu"
#
# Linha no meio do corredor estreito (ROI acima), entre o
# pátio (esquerda) e a passagem lateral/parede (direita).

CROSS_LINE_X = 1800

# Linha horizontal opcional, no mesmo corredor, para detectar
# também o movimento no outro eixo (topo/base) — útil quando o
# pet anda ao longo do corredor em vez de atravessá-lo de lado
# a lado. Coordenada Y em pixels, mesmo sistema da ROI.
#
# Cruzar de cima para baixo = "saiu"
# Cruzar de baixo para cima = "entrou"
#
# Defina CROSS_LINE_Y = None para monitorar só a linha vertical.

CROSS_LINE_Y = 875

# Distância mínima (em pixels) que o centro do pet precisa
# se mover entre frames para confirmarmos que realmente
# cruzou a linha (evita alertas falsos por ruído/tremor
# da caixa delimitadora bem em cima da linha).
CROSS_MARGIN = 5

# Tempo mínimo (segundos) entre alertas do mesmo track ID,
# para não floodar o Telegram caso o pet fique andando
# para lá e para cá perto da linha.
ALERT_COOLDOWN_SECONDS = 15


# ============================================================
# COLETA DE DATASET (FINE-TUNING)
# ============================================================
#
# Salva automaticamente o recorte cru da ROI (sem caixas
# desenhadas) sempre que um pet for detectado, para depois
# anotar manualmente (ex: LabelImg) e treinar um modelo
# ajustado a este quintal/câmera específicos.
#
# Defina DATASET_CAPTURE_ENABLED = False para desativar.

DATASET_CAPTURE_ENABLED = True
DATASET_DIR = "dataset/raw"

# Confiança mínima para salvar o frame (independente do
# CONFIDENCE usado para o alerta — aqui vale a pena manter
# mais alto, para não poluir o dataset com falsos positivos).
DATASET_MIN_CONFIDENCE = 0.35

# Intervalo mínimo (segundos) entre capturas do mesmo track ID,
# para não salvar centenas de frames quase idênticos do pet
# parado no mesmo lugar.
DATASET_CAPTURE_COOLDOWN_SECONDS = 5


# ============================================================
# CONFIGURAÇÃO DO TELEGRAM
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TELEGRAM_ENABLED = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


# Largura máxima da imagem enviada nos alertas do Telegram
# (mantém proporção). Defina como 0 para desativar o
# redimensionamento e enviar em resolução original.
ALERT_IMAGE_MAX_WIDTH = 800


def resize_for_alert(image):
    if ALERT_IMAGE_MAX_WIDTH <= 0:
        return image

    height, width = image.shape[:2]
    if width <= ALERT_IMAGE_MAX_WIDTH:
        return image

    scale = ALERT_IMAGE_MAX_WIDTH / width
    new_size = (ALERT_IMAGE_MAX_WIDTH, int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


# Rajadas de detecção (vários track IDs cruzando a linha em
# poucos minutos) podiam disparar muitas threads de envio ao
# Telegram ao mesmo tempo, cada uma segurando sua própria cópia
# do frame + buffer JPEG até o POST completar. Limitamos quantos
# envios rodam ao mesmo tempo; o excesso apenas espera a vez.
_alert_send_semaphore = threading.Semaphore(2)


def send_telegram_alert(message, image=None):
    """
    Envia uma notificação para o Telegram em uma thread
    separada, para não travar o loop de detecção enquanto
    espera a resposta da API.
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
                            timeout=10
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
                    timeout=10
                )

            except requests.RequestException as exc:
                print(f"\nErro ao enviar alerta Telegram: {exc}")

    threading.Thread(target=_send, daemon=True).start()


def check_crossing(
    axis,
    side,
    track_id,
    class_name,
    last_side,
    last_alert_time,
    annotated_frame
):
    """
    Verifica se o pet cruzou a linha de um eixo ("x" ou "y") e,
    se sim (respeitando o cooldown), dispara o alerta Telegram.

    side: "left"/"right" para eixo x, "top"/"bottom" para eixo y.
    Retorna nada; atualiza last_side/last_alert_time in-place.
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
                # Esquerda -> direita = ENTROU; direita -> esquerda = SAIU.
                direction = "ENTROU" if previous_side == "left" else "SAIU"
            else:
                direction = "SAIU" if previous_side == "top" else "ENTROU"

            message = (
                f"{class_name.capitalize()} "
                f"(ID {track_id}) {direction} "
                f"pela área monitorada."
            )

            print(f"\n{message}")

            send_telegram_alert(
                message,
                image=annotated_frame
            )

    if side is not None:
        last_side[key] = side


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    if DATASET_CAPTURE_ENABLED:
        os.makedirs(DATASET_DIR, exist_ok=True)

    print("Carregando modelo YOLO...")

    model = YOLO(MODEL_NAME)

    print("Modelo carregado.")
    print()
    print(f"NVR:      {NVR_IP}:{NVR_PORT}")
    print(f"Canal:    {CHANNEL}")
    print(f"Stream:   {STREAM}")
    print()

    print("Conectando ao NVR...")

    cap = cv2.VideoCapture(
        RTSP_URL,
        cv2.CAP_FFMPEG
    )

    if not cap.isOpened():
        print()
        print("ERRO: não foi possível abrir o stream RTSP.")
        print()
        print("Verifique:")
        print("  - IP do NVR")
        print("  - usuário")
        print("  - senha")
        print("  - canal")
        print("  - URL RTSP")
        return

    print("Câmera conectada.")

    stream = StreamReader(cap)

    # ========================================================
    # INFORMAÇÕES DO STREAM
    # ========================================================

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    print()
    print("Informações do vídeo:")
    print(f"  Resolução: {width}x{height}")
    print(f"  FPS:       {fps}")
    print()

    print("Pressione Q ou ESC para sair.")
    print()

    cv2.namedWindow(
        "Pet Detector",
        cv2.WINDOW_NORMAL
    )

    if TELEGRAM_ENABLED:
        print("Alertas Telegram: ATIVADOS")
    else:
        print(
            "Alertas Telegram: DESATIVADOS "
            "(configure TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID no .env)"
        )
    print()

    # Último lado conhecido de cada track ID em relação a cada
    # linha de cruzamento. Chave: (track_id, eixo), onde eixo é
    # "x" (linha vertical) ou "y" (linha horizontal). Valor:
    # "left"/"right" para eixo x, "top"/"bottom" para eixo y.
    last_side = {}

    # Timestamp do último alerta disparado para cada
    # (track_id, eixo), para não floodar o Telegram.
    last_alert_time = {}

    # Timestamp da última captura de dataset por track ID.
    last_dataset_capture_time = {}

    # ========================================================
    # LOOP
    # ========================================================

    try:

        while True:

            ret, frame = stream.read()

            if not ret:

                print(
                    "Falha ao receber frame."
                )

                time.sleep(0.1)

                continue

            if frame is None:

                # Fila vazia no momento: a captura está mais
                # rápida que o consumo, ou ainda não chegou
                # nenhum frame. Espera um pouco e tenta de novo,
                # sem tratar como falha.

                time.sleep(0.01)

                continue

            # =================================================
            # RECORTE DA METADE ÚTIL (CÂMERA COMPOSTA)
            # =================================================
            #
            # A partir daqui, "frame" passa a representar apenas
            # a metade de baixo (pátio). Todo o resto do loop
            # (ROI, desenho, cruzamento de linha) trabalha nesse
            # sistema de coordenadas.

            if COMPOSITE_SPLIT_Y is not None:
                frame = frame[COMPOSITE_SPLIT_Y:, :]

            # =================================================
            # RECORTE DA REGIÃO DE INTERESSE (ROI)
            # =================================================

            if ROI is not None:
                x1, y1, x2, y2 = ROI
                detect_frame = frame[y1:y2, x1:x2]
            else:
                x1, y1 = 0, 0
                detect_frame = frame

            # =================================================
            # YOLO + TRACKING
            # =================================================
            #
            # persist=True é importante:
            #
            # O tracker mantém a identidade dos objetos
            # entre os frames.
            #
            # Assim podemos ter:
            #
            #   cachorro ID 1
            #   cachorro ID 1
            #   cachorro ID 1
            #
            # mesmo que a posição dele mude.
            #

            results = model.track(
                detect_frame,
                persist=True,
                classes=PET_CLASSES,
                conf=CONFIDENCE,
                imgsz=960,
                verbose=False
            )

            result = results[0]

            # =================================================
            # DESENHAR RESULTADO
            # =================================================
            #
            # As detecções são feitas em detect_frame (recorte
            # da ROI), então o result.plot() só desenha nesse
            # recorte. Para monitorar visualmente e para anexar
            # aos alertas do Telegram, colamos o recorte
            # anotado de volta no frame completo, com o
            # retângulo da ROI marcado.

            annotated_crop = result.plot()

            annotated_frame = frame.copy()

            annotated_frame[
                y1:y1 + annotated_crop.shape[0],
                x1:x1 + annotated_crop.shape[1]
            ] = annotated_crop

            # Cópia sem os contornos de ROI/linha de cruzamento,
            # usada nos alertas do Telegram — mostra o frame
            # inteiro com as caixas de detecção do YOLO, mas sem
            # a marcação da área monitorada.
            alert_frame = annotated_frame.copy()

            if ROI is not None:
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (ROI[2], ROI[3]),
                    (0, 0, 255),
                    2
                )

            cv2.line(
                annotated_frame,
                (CROSS_LINE_X, y1),
                (CROSS_LINE_X, ROI[3] if ROI is not None else frame.shape[0]),
                (255, 0, 255),
                2
            )

            if CROSS_LINE_Y is not None:
                cv2.line(
                    annotated_frame,
                    (x1, CROSS_LINE_Y),
                    (ROI[2] if ROI is not None else frame.shape[1], CROSS_LINE_Y),
                    (255, 255, 0),
                    2
                )

            # =================================================
            # MOSTRAR DETECÇÕES NO TERMINAL
            # =================================================

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(
                        box.cls[0]
                    )

                    confidence = float(
                        box.conf[0]
                    )

                    class_name = model.names[
                        class_id
                    ]

                    # ID fornecido pelo tracker
                    if box.id is not None:
                        track_id = int(
                            box.id[0]
                        )
                    else:
                        track_id = -1

                    print(
                        f"\r"
                        f"Pet: {class_name:<5} "
                        f"ID: {track_id:<3} "
                        f"Conf: {confidence:.2f}",
                        end=""
                    )

                    # =========================================
                    # COLETA DE DATASET (FINE-TUNING)
                    # =========================================
                    #
                    # Salva o recorte cru da ROI (detect_frame,
                    # sem caixas desenhadas), respeitando um
                    # cooldown por track ID para não acumular
                    # centenas de frames repetidos do mesmo pet
                    # parado.

                    if (
                        DATASET_CAPTURE_ENABLED
                        and confidence >= DATASET_MIN_CONFIDENCE
                    ):

                        capture_key = track_id if track_id != -1 else id(box)
                        now = time.time()
                        last_capture = last_dataset_capture_time.get(
                            capture_key, 0
                        )

                        if (
                            now - last_capture
                            >= DATASET_CAPTURE_COOLDOWN_SECONDS
                        ):

                            last_dataset_capture_time[capture_key] = now

                            # Nome só com timestamp — não inclui a
                            # classe prevista pelo modelo, para não
                            # confundir com uma anotação real durante
                            # a revisão manual (pre_annotate.py gera
                            # a anotação de verdade, no .txt).
                            timestamp = datetime.now().strftime(
                                "%Y%m%d_%H%M%S_%f"
                            )
                            filename = os.path.join(
                                DATASET_DIR,
                                f"{timestamp}.jpg"
                            )

                            cv2.imwrite(filename, detect_frame)

                    # =========================================
                    # DETECÇÃO DE CRUZAMENTO DE LINHA
                    # =========================================
                    #
                    # As coordenadas da caixa (box.xyxy) estão
                    # no sistema do detect_frame (recorte da
                    # ROI). Convertemos para o sistema da
                    # imagem original somando o deslocamento
                    # da ROI (x1, y1), que é o mesmo sistema
                    # de CROSS_LINE_X.

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
                            "x",
                            side_x,
                            track_id,
                            class_name,
                            last_side,
                            last_alert_time,
                            alert_frame
                        )

                        if CROSS_LINE_Y is not None:

                            if center_y < CROSS_LINE_Y - CROSS_MARGIN:
                                side_y = "top"
                            elif center_y > CROSS_LINE_Y + CROSS_MARGIN:
                                side_y = "bottom"
                            else:
                                side_y = None

                            check_crossing(
                                "y",
                                side_y,
                                track_id,
                                class_name,
                                last_side,
                                last_alert_time,
                                alert_frame
                            )

            cv2.imshow(
                "Pet Detector",
                annotated_frame
            )

            # =================================================
            # TECLADO
            # =================================================

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:

        print()
        print()
        print("Interrompido pelo usuário.")

    finally:

        print()
        print()
        print("Encerrando...")

        stream.stop()
        cap.release()

        cv2.destroyAllWindows()

        # Garante processamento dos eventos
        # pendentes das janelas OpenCV.

        for _ in range(5):
            cv2.waitKey(1)

        print("Finalizado.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()