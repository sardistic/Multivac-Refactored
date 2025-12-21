import os
from flask import Flask
from concurrent.futures import ThreadPoolExecutor
import discord
from discord_bot import bot
from config import DISCORD_TOKEN

app = Flask(__name__)


@app.route('/')
def home():
    return "Discord bot is running."


def run_flask_app():
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)


def run_discord_bot():
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    executor = ThreadPoolExecutor(max_workers=2)
    executor.submit(run_flask_app)
    executor.submit(run_discord_bot)
