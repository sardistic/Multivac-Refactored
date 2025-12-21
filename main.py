import sys
import time
from logger_setup import configure_logging

# Set up logging before importing any bot modules
configure_logging('--verbose' in sys.argv)

from discord_bot import bot
from config import DISCORD_TOKEN

if __name__ == "__main__":
    print(">>> Starting Discord bot...")
    time.sleep(1)  # slight delay for visibility/logs
    bot.run(DISCORD_TOKEN)
