import os
import json
import urllib.request
import zipfile
import tensorflow as tf
from keras.layers import TFSMLayer

# 1) ССЫЛКА НА ZIP ИЗ RELEASE (вставь свою)
MODEL_ZIP_URL = "https://github.com/dariagustaya-cyber/job/releases/download/v1.0.0/jobshield_web_model.zip"

# 2) Куда распаковываем модель на Vercel (временная папка доступна)
CACHE_DIR = "/tmp/jobshield_model"
MODEL_DIR = os.path.join(CACHE_DIR, "jobshield_web_model")
ZIP_PATH = os.path.join(CACHE_DIR, "jobshield_web_model.zip")

_model = None

def _ensure_model_downloaded():
    os.makedirs(CACHE_DIR, exist_ok=True)

    # если уже распаковано — ничего не делаем
    if os.path.exists(os.path.join(MODEL_DIR, "saved_model.pb")):
        return

    # скачиваем zip
    urllib.request.urlretrieve(MODEL_ZIP_URL, ZIP_PATH)

    # распаковываем
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(CACHE_DIR)

    # иногда в zip бывает вложенная папка — проверим
    # ожидаем структуру: /tmp/jobshield_model/jobshield_web_model/saved_model.pb
    if not os.path.exists(os.path.join(MODEL_DIR, "saved_model.pb")):
        # если zip распаковался не так, попробуем найти папку с saved_model.pb
        for root, dirs, files in os.walk(CACHE_DIR):
            if "saved_model.pb" in files:
                # root — папка модели
                # сделаем symlink/копирование пути через переменную
                # просто переопределим MODEL_DIR глобально нельзя, поэтому делаем fallback:
                return root

        raise FileNotFoundError("saved_model.pb not found after unzip")

    return MODEL_DIR

def _get_model():
    global _model
    if _model is not None:
        return _model

    model_path = _ensure_model_downloaded()

    # IMPORTANT: call_endpoint должен совпадать с тем, что внутри SavedModel.
    # У тебя работало 'serve', значит оставляем.
    _model = TFSMLayer(model_path, call_endpoint="serve")
    return _model

def handler(request):
    if request.method != "POST":
        return {
            "statusCode": 405,
            "body": json.dumps({"error": "Only POST allowed"})
        }

    try:
        body = json.loads(request.body or "{}")
        text = (body.get("text") or "").strip()

        if not text:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Text is required"})
            }

        model = _get_model()
        x = tf.constant([text])
        prob = float(model(x).numpy()[0][0])

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "fraud_probability": prob,
                "is_fraud": prob >= 0.5
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }
