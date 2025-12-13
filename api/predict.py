import os
import json
import tensorflow as tf
from keras.layers import TFSMLayer

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "jobshield_web_model"
)

# Загружаем модель ОДИН РАЗ (очень важно для Vercel)
model = TFSMLayer(
    MODEL_DIR,
    call_endpoint="serve"
)

def handler(request):
    if request.method != "POST":
        return {
            "statusCode": 405,
            "body": json.dumps({"error": "Only POST allowed"})
        }

    try:
        body = json.loads(request.body)
        text = body.get("text", "")

        if not text:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Text is required"})
            }

        x = tf.constant([text])
        prob = float(model(x).numpy()[0][0])

        return {
            "statusCode": 200,
            "body": json.dumps({
                "fraud_probability": prob,
                "is_fraud": prob >= 0.5
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
