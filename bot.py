import os, json, time, logging
import telebot
import numpy as np, faiss
from sentence_transformers import SentenceTransformer
from flask import Flask, request
from google import genai
import torch

torch.set_num_threads(1)

# ---------- КОНФИГ ----------
TOKEN = "8279028319:AAE-EMzU4eUC-lnVf3OANZ5cc0glOXlK8q8"
DB_PATH = "persona_pure.json"
INDEX_PATH = "faiss_index.bin"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "ВАШ_КЛЮЧ_ЗДЕСЬ")
GEMINI_MODEL = "gemini-2.5-flash"
TRIGGER_WORDS = ["олег", "олеж", "дед"]
PORT = int(os.environ.get("PORT", 7860))

# ---------- ИНИЦИАЛИЗАЦИЯ ----------
app = Flask(__name__)
bot = telebot.TeleBot(TOKEN, threaded=False)

print(">>> Загрузка базы и модели...")
with open(DB_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)
if raw and isinstance(raw[0], dict):
    static_texts = [item.get("text", item.get("answer", str(item))) for item in raw]
else:
    static_texts = raw

index = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP)   # экономия RAM
model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Готово: {len(static_texts)} текстов")

gemini_client = None
if GEMINI_API_KEY and GEMINI_API_KEY != "ВАШ_КЛЮЧ_ЗДЕСЬ":
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print("✅ Gemini подключён")

def retrieve_examples(query, k=5):
    qvec = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
    D, I = index.search(qvec.reshape(1, -1), k)
    return [static_texts[i] for i in I[0] if i >= 0]

def generate_response(user_text):
    if not gemini_client:
        return random.choice(static_texts)  # fallback
    examples = retrieve_examples(user_text)
    example_block = "\n".join([f"Пример: {ex}" for ex in examples])
    prompt = (
        "Ты — Олег, общаешься в Telegram-чате. Твой стиль: простой, разговорный, иногда с юмором. "
        "Отвечай одной репликой, без пояснений.\n"
        f"Примеры твоих настоящих ответов:\n{example_block}\n\n"
        f"Собеседник: {user_text}\nТвой ответ:"
    )
    try:
        resp = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return resp.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return random.choice(static_texts)

# ---------- ОБРАБОТЧИК TELEGRAM ----------
@bot.message_handler(func=lambda m: True)
def handle_message(message):
    text = message.text
    if not text:
        return

    # проверяем, нужно ли отвечать
    reply_to_bot = (message.reply_to_message and
                    message.reply_to_message.from_user.id == bot.get_me().id)
    mentioned = any(w in text.lower() for w in TRIGGER_WORDS)
    if not (mentioned or reply_to_bot):
        return   # не реагируем

    print(f">>> {text[:60]}")
    answer = generate_response(text)
    bot.reply_to(message, answer)

# ---------- ВЕБХУК FLASK ----------
@app.route("/", methods=["POST"])
def webhook():
    if request.headers.get("content-type") == "application/json":
        update = telebot.types.Update.de_json(request.get_json())
        bot.process_new_updates([update])
        return "OK"
    return "BAD REQUEST", 400

@app.route("/health")
def health():
    return "OK"

# ---------- СТАРТ ----------
if __name__ == "__main__":
    bot.remove_webhook()
    time.sleep(1)
    bot.set_webhook(url=f"https://{os.environ.get('RENDER_EXTERNAL_HOSTNAME')}/")
    print(">>> Бот запущен")
    app.run(host="0.0.0.0", port=PORT)