import os, json, time, random, logging
import telebot
import numpy as np, faiss
import torch
from sentence_transformers import SentenceTransformer
from flask import Flask, request

# ---------- ЭКОНОМИЯ ПАМЯТИ ----------
torch.set_num_threads(1)  # один поток для PyTorch

TOKEN = "8279028319:AAH5QZLWXHfL2gNnYFQTTt0ehKgDrU7hXqg"
DB_PATH = "persona_pure.json"
INDEX_PATH = "faiss_index.bin"
MODEL_NAME = "all-MiniLM-L6-v2"
PORT = int(os.environ.get("PORT", 7860))

app = Flask(__name__)
bot = telebot.TeleBot(TOKEN, threaded=False)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RenderBot")

# ---------- ЗАГРУЗКА ДАННЫХ (оптимизировано) ----------
logger.info("Загрузка базы...")
with open(DB_PATH, "r", encoding="utf-8") as f:
    memory = json.load(f)
if len(memory) > 0 and isinstance(memory[0], dict):
    memory = [item.get("text", item.get("answer", str(item))) for item in memory]

logger.info("Загрузка FAISS‑индекса (mmap)...")
# IO_FLAG_MMAP = 2 – индекс будет отображён в память, не загружаясь полностью
index = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP)

logger.info("Загрузка модели...")
model = SentenceTransformer(MODEL_NAME, device="cpu")
logger.info("Модель готова")

def get_rag_answer(query):
    try:
        vec = model.encode([query]).astype("float32")
        D, I = index.search(vec, 1)
        idx = int(I[0][0])
        return memory[idx]
    except:
        return random.choice(memory)

# ---------- ОБРАБОТЧИКИ ----------
@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    text = message.text
    if not text:
        return
    logger.info(f"Сообщение: {text[:30]}")
    is_mentioned = any(name in text.lower() for name in ["олег", "олеж", "бот"])
    is_lucky = random.random() < 0.10
    if is_mentioned or is_lucky:
        time.sleep(random.randint(2, 5))
        ans = get_rag_answer(text)
        bot.reply_to(message, ans)

@app.route("/", methods=["POST"])
def webhook():
    if request.headers.get("content-type") == "application/json":
        update = telebot.types.Update.de_json(request.get_json())
        bot.process_new_updates([update])
        return "OK"
    return "BAD REQUEST", 400

@app.route("/health")
def health():
    return "OK", 200

# ---------- УСТАНОВКА ВЕБХУКА ----------
def set_webhook():
    public_url = f"https://{os.environ.get('RENDER_EXTERNAL_HOSTNAME')}/"
    bot.remove_webhook()
    bot.set_webhook(url=public_url)
    logger.info(f"Вебхук установлен на {public_url}")

if __name__ == "__main__":
    set_webhook()
    app.run(host="0.0.0.0", port=PORT)
