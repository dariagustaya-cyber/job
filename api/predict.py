from http.server import BaseHTTPRequestHandler
import json
import tensorflow as tf
from keras.layers import TFSMLayer
import requests
from bs4 import BeautifulSoup

model = TFSMLayer(
    "jobshield_web_model",
    call_endpoint="serve"
)

def extract_text_from_url(url):
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text(separator=" ")
    return text[:20000]  # защита

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        body = self.rfile.read(content_length)
        data = json.loads(body)

        url = data.get("url")
        if not url:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing URL")
            return

        text = extract_text_from_url(url)

        prob = model(tf.constant([text])).numpy()[0][0]

        response = {
            "probability_fake": float(prob),
            "label": "FAKE" if prob > 0.67 else "REAL"
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
