# progress.py (fixed version)

import asyncio
import random

FULL_BLOCK = '█'
PARTIAL_BLOCKS = [
    (0.00, ' '),
    (0.125, '▏'),
    (0.25, '▎'),
    (0.375, '▍'),
    (0.5, '▌'),
    (0.625, '▋'),
    (0.75, '▊'),
    (0.875, '▉'),
    (1.00, '█')
]
FADE_BLOCKS = ['.', ':', '-', '░', '▒', '▓']

async def start_progress_bar(message, task: asyncio.Task, action_label="Working", emoji="💬", duration_estimate=40):
    width = 24
    animation_update_interval = 0.1     # Local animation refresh (every 100ms)
    discord_edit_interval = 1.5          # Only push .edit() every 1.5s to Discord

    last_discord_edit = 0
    start_time = task._loop.time()

    try:
        while not task.done():
            now = task._loop.time()
            elapsed = now - start_time
            progress = min(elapsed / duration_estimate, 1.0)

            bar = build_progress_bar(progress, width, fancy=True)
            render = f"{emoji} {action_label} {bar}"

            if now - last_discord_edit >= discord_edit_interval:
                try:
                    await message.edit(content=render)
                    last_discord_edit = now
                except Exception:
                    pass

            await asyncio.sleep(animation_update_interval)

        # FINAL forced full-bar once done
        bar = build_progress_bar(1.0, width, fancy=False)
        final_render = f"{emoji} {action_label} {bar}"
        try:
            await message.edit(content=final_render)
        except Exception:
            pass

    except Exception:
        pass

def build_progress_bar(progress: float, width: int = 24, fancy=True) -> str:
    progress = max(0.0, min(1.0, progress))
    filled_blocks = int(progress * width)
    partial_ratio = (progress * width) - filled_blocks

    bar = ''

    for i in range(width):
        if i < filled_blocks:
            bar += FULL_BLOCK
        elif i == filled_blocks:
            bar += select_partial_block(partial_ratio)
        else:
            if fancy:
                bar += random.choice(FADE_BLOCKS)
            else:
                bar += random.choice(['░', '▒', '▓'])

    return f"[{bar}]"

def select_partial_block(ratio: float) -> str:
    for threshold, char in PARTIAL_BLOCKS:
        if ratio <= threshold:
            return char
    return FULL_BLOCK
