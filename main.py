import logging
import os
import openai
import time
from flask import Flask, request
from telebot import TeleBot, types
from dotenv import load_dotenv

load_dotenv()

# APP SET UP
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
BOT_TOKEN = os.environ.get("BOT_TOKEN")
URL = os.environ.get("URL")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET")

bot = TeleBot(BOT_TOKEN, threaded=True)
bot.remove_webhook()
time.sleep(1)
bot.set_webhook(url=f"{URL}/{WEBHOOK_SECRET}")


@app.route(f"/{WEBHOOK_SECRET}", methods=["POST"])
def webhook():
    """Webhook to handle incoming updates from Telegram."""
    update = types.Update.de_json(request.data.decode("utf8"))
    bot.process_new_updates([update])
    return "ok", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0")
