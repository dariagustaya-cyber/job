import os
import json
import tensorflow as tf
from keras.layers import TFSMLayer

# Путь к папке SavedModel внутри репозитория
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "jobshield_web_model")

# Ленивая загрузка (загрузим один раз на холодном старте)
_model = None

def get_model():
    global _model
    if _model is None:
        # endpoint почти всегда "serve" (как у тебя в Colab)
        _model = TFSMLayer(MODEL_DIR, call_endpoint="serve")
    return _model

def handler(request):
    # CORS (чтобы фронт мог дергать API)
    headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }

    # Preflight
    if request.method == "OPTIONS":
        return (204, headers, "")

    if request.method != "POST":
        return (405, headers, json.dumps({"error": "Use POST"}))

    try:
        body = request.json if hasattr(request, "json") else None
        if body is None:
            body = json.loads(request.body.decode("utf-8"))

        text = (body.get("text") or "").strip()
        if not text:
            return (400, headers, json.dumps({"error": "Field 'text' is required"}))

        model = get_model()

        x = tf.constant([text])  # batch из 1 строки
        y = model(x)             # (1,1)
        prob = float(y.numpy().reshape(-1)[0])

        return (200, headers, json.dumps({"probability": prob}))
    except Exception as e:
        return (500, headers, json.dumps({"error": str(e)}))
