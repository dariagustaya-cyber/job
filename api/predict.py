import os
import json
import zipfile
import requests
import tensorflow as tf
from keras.layers import TFSMLayer

# Ссылка на zip из GitHub Releases (у тебя уже правильная)
MODEL_ZIP_URL = "https://github.com/dariagustaya-cyber/job/releases/download/v1.0.0/jobshield_web_model.zip"

# На Vercel можно писать только в /tmp
TMP_DIR = "/tmp"
MODEL_DIR = os.path.join(TMP_DIR, "jobshield_web_model")
MODEL_ZIP_PATH = os.path.join(TMP_DIR, "jobshield_web_model.zip")

_model = None  # кэш модели в памяти (на время жизни контейнера)


def ensure_model_downloaded():
    """Скачать и распаковать модель в /tmp, если её ещё нет."""
    # Проверяем, что распаковка уже есть (saved_model.pb должен существовать)
    saved_model_pb = os.path.join(MODEL_DIR, "saved_model.pb")
    if os.path.exists(saved_model_pb):
        return

    # Скачиваем zip
    r = requests.get(MODEL_ZIP_URL, stream=True, timeout=60, allow_redirects=True)
    r.raise_for_status()

    with open(MODEL_ZIP_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    # Распаковываем
    os.makedirs(MODEL_DIR, exist_ok=True)
    with zipfile.ZipFile(MODEL_ZIP_PATH, "r") as z:
        z.extractall(MODEL_DIR)

    # Частый кейс: zip содержит папку jobshield_web_model внутри.
    # Тогда фактический путь будет /tmp/jobshield_web_model/jobshield_web_model/saved_model.pb
    nested = os.path.join(MODEL_DIR, "jobshield_web_model", "saved_model.pb")
    if os.path.exists(nested):
        # переопределим MODEL_DIR на вложенную папку
        return "nested"

    return "root"


def get_model():
    global _model
    if _model is not None:
        return _model

    where = ensure_model_downloaded()

    model_path = MODEL_DIR
    if where == "nested":
        model_path = os.path.join(MODEL_DIR, "jobshield_web_model")

    # call_endpoint у тебя работал как "serve"
    _model = TFSMLayer(model_path, call_endpoint="serve")
    return _model


def handler(request):
    # CORS (чтобы фронтенд мог дергать API)
    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }

    if request.method == "OPTIONS":
        return {"statusCode": 200, "headers": cors_headers, "body": ""}

    if request.method != "POST":
        return {
            "statusCode": 405,
            "headers": cors_headers,
            "body": json.dumps({"error": "Only POST allowed"})
        }

    try:
        body = json.loads(request.body or "{}")
        text = (body.get("text") or "").strip()

        if not text:
            return {
                "statusCode": 400,
                "headers": cors_headers,
                "body": json.dumps({"error": "Text is required"})
            }

        model = get_model()

        x = tf.constant([text])
        prob = float(model(x).numpy()[0][0])

        return {
            "statusCode": 200,
            "headers": {**cors_headers, "Content-Type": "application/json"},
            "body": json.dumps({
                "fraud_probability": prob,
                "is_fraud": prob >= 0.5
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": cors_headers,
            "body": json.dumps({"error": str(e)})
        }
