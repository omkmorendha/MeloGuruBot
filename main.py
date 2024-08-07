import json
import logging
import os
import time
from flask import Flask, request
from telebot import TeleBot, types
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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

questions_loader = TextLoader('./questions.json')
raw_questions = questions_loader.load()

# Load answers JSON and extract DEFAULT_RESPONSE
with open('./answers.json', 'r') as f:
    answers_data = json.load(f)
DEFAULT_RESPONSE = answers_data.get('default_response', "I apologize, but I don't have enough information to answer that question.")

answers_loader = TextLoader('./answers.json')
raw_answers = answers_loader.load()

raw_documents = raw_questions + raw_answers
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings())

bot = TeleBot(BOT_TOKEN, threaded=True)
# bot.remove_webhook()
# time.sleep(1)
# bot.set_webhook(url=f"{URL}/{WEBHOOK_SECRET}")


# Create the retrieval chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

template = """
1. Answer the question based only on the following context, only reply with the answer:
{context}

2. Answer precisely from the context and then provide additional relevantinformation if available. 
3. Answer in a friendly, positive, and appreciative tone. 
4. Stay brief and only answer with information relevant to the question
5. If you cannot find a relevant answer in the context, respond with: {default_response}

Question: 
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough(), "default_response": lambda _: DEFAULT_RESPONSE}
    | prompt
    | llm
    | StrOutputParser()
)

@app.route(f"/{WEBHOOK_SECRET}", methods=["POST"])
def webhook():
    """Webhook to handle incoming updates from Telegram."""
    update = types.Update.de_json(request.data.decode("utf8"))
    bot.process_new_updates([update])
    return "ok", 200


@bot.message_handler(commands=["start", "restart"])
def start(message):
    """Handle /start and /restart commands."""
    chat_id = message.chat.id

    message_to_send = "Welcome to Melospeech! I'm MeloGuru AI Assistant, I'm here to assist you with any questions or needs you might have. I can make mistakes. Always check with your guru if you're in doubt."    
    bot.send_message(message.chat.id, message_to_send, parse_mode="Markdown")


@bot.message_handler(func=lambda message: True)
def respond_to_message(message):
    query = message.text.strip().lower()
    try:
        response = chain.invoke(query)
        if not response or response.strip() == "":
            response = DEFAULT_RESPONSE
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        response = DEFAULT_RESPONSE
    
    bot.reply_to(message, response, parse_mode='Markdown')
    return


if __name__ == "__main__":
    app.run(host="0.0.0.0")
