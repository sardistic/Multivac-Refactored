from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from flask import Flask

from config import DISCORD_TOKEN
from discord_bot import bot

app = Flask(__name__)


@app.route("/")
def home():
    return "Discord bot is running."


def run_flask_app():
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)


def run_discord_bot():
    bot.run(DISCORD_TOKEN)


def run_combined():
    executor = ThreadPoolExecutor(max_workers=2)
    executor.submit(run_flask_app)
    executor.submit(run_discord_bot)
