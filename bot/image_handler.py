import asyncio
import base64
import io
import json
import logging
import mimetypes
import re

import discord

from bot.chat_context import build_chat_context
from gemini_utils import generate_gemini_image
from openai_utils import generate_openai_messages_response_with_tools, get_openai_client
from stability_utils import handle_image_generation
from weather_utils import get_location_details, get_weather_data

logger = logging.getLogger("discord_bot")


async def handle_generate_image_intent(
    *,
    message,
    prompt: str,
    duration_estimate: int,
    stream_ok: bool,
    live_status_with_progress,
):
    weather_match = re.search(r"imagine\s+weather\s+(.*)", prompt, flags=re.IGNORECASE)
    if weather_match:
        loc_query = weather_match.group(1).strip()
        if not loc_query:
            await message.reply("❌ Please specify a location, e.g. `imagine weather Tokyo`.")
            return

        async def _generate_weather_widget():
            try:
                loc = await get_location_details(loc_query)
                units = "imperial" if "US" in loc.get("name", "") else "metric"
                data = await get_weather_data(loc["lat"], loc["lon"], units=units)

                current = data.get("current", {})
                main = current.get("main", {})
                wind = current.get("wind", {})

                temp = main.get("temp", "?")
                feels_like = main.get("feels_like", "?")
                humidity = main.get("humidity", "?")
                temp_min = main.get("temp_min", "?")
                temp_max = main.get("temp_max", "?")
                pressure = main.get("pressure", "?")
                wind_speed = wind.get("speed", "?")
                visibility = current.get("visibility", "?")
                clouds = current.get("clouds", {}).get("all", "?")
                cond = (current.get("weather") or [{}])[0].get("description", "unknown")

                forecast_data = data.get("forecast", {})
                forecast_cur = forecast_data.get("current", {})
                uvi = forecast_cur.get("uvi", "?")
                pop = forecast_data.get("daily", [{}])[0].get("pop", 0) * 100

                sys_data = current.get("sys", {})
                sunrise_raw = sys_data.get("sunrise")
                sunset_raw = sys_data.get("sunset")

                import datetime

                tz_offset = current.get("timezone", 0)
                local_dt = datetime.datetime.utcnow() + datetime.timedelta(seconds=tz_offset)
                time_str = local_dt.strftime("%I:%M %p")

                sr_str = datetime.datetime.utcfromtimestamp(sunrise_raw + tz_offset).strftime("%I:%M %p") if sunrise_raw else "?"
                ss_str = datetime.datetime.utcfromtimestamp(sunset_raw + tz_offset).strftime("%I:%M %p") if sunset_raw else "?"

                if isinstance(visibility, (int, float)):
                    vis_str = f"{round(visibility / 1609.34, 1)} mi" if units == "imperial" else f"{round(visibility / 1000, 1)} km"
                else:
                    vis_str = "?"

                widget_prompt = (
                    f"A professional, high-density data-maximalist 3D weather station dashboard layout in a WIDESCREEN 16:9 cinematic format. "
                    f"THE PRIMARY FOCUS is the hero weather block: Large '{round(float(temp))}°' and '{loc['name']}' in HUGE, BOLD, HIGH-CONTRAST typography. \n"
                    f"COMPREHENSIVE DATA GRID (crisp, clear, modern labels): \n"
                    f"- Today's Range: {round(float(temp_min))}° - {round(float(temp_max))}° \n"
                    f"- Feels Like: {round(float(feels_like))}° | Humidity: {humidity}% \n"
                    f"- Rain Chance: {round(pop)}% | UV Index: {uvi} \n"
                    f"- Pressure: {pressure} hPa | Visibility: {vis_str} \n"
                    f"- Wind: {wind_speed} {'mph' if units=='imperial' else 'm/s'} | Clouds: {clouds}% \n"
                    f"- Sunrise: {sr_str} | Sunset: {ss_str} \n"
                    f"- Local Time: {time_str} \n"
                    f"The layout is a modern tech interface with transparent elements. "
                    f"The background is a cinematic, expansive widescreen shot of {loc['name']} "
                    f"reflecting current {cond} skies and {'nighttime' if local_dt.hour < 6 or local_dt.hour > 18 else 'daytime'} lighting. "
                    f"Premium Apple / SF Pro typography / iOS 17 Weather app aesthetic, 8k hyper-detailed text."
                )
                logger.info(f"Generating extreme weather widget: {widget_prompt}")
                return await asyncio.to_thread(generate_gemini_image, widget_prompt, 1600, 900)
            except Exception as e:
                logger.error(f"Weather widget failed: {e}")
                return None

        status_msg, image_data = await live_status_with_progress(
            message,
            action_label="Building Widget",
            emoji="🌦️",
            coro=_generate_weather_widget(),
            duration_estimate=15,
            summarizer=(lambda: "Fetching live data... Rendering widget...") if stream_ok else None,
        )

        if image_data:
            image_data.seek(0)
            await status_msg.reply(file=discord.File(image_data, filename="weather_widget.png"))
            await status_msg.edit(content=f"✅ Weather Widget for **{loc_query}**")
        else:
            await status_msg.edit(content="❌ Failed to generate weather widget.")
        return

    status_msg, image_data = await live_status_with_progress(
        message,
        action_label="Generating",
        emoji="🎨",
        coro=handle_image_generation(message, prompt),
        duration_estimate=duration_estimate,
        summarizer=(lambda: "Rendering image… adding details…") if stream_ok else None,
    )
    if image_data:
        await status_msg.edit(content="✅ Image generated")
        await message.channel.send(file=discord.File(image_data, "generated_image.png"))
    else:
        await status_msg.edit(content="❌ Image generation failed.")


async def handle_edit_image_intent(
    *,
    message,
    prompt: str,
    image_urls,
    prompt_for_image_selection,
    live_status_with_progress,
):
    images_to_edit = image_urls
    if len(image_urls) > 1:
        selection = await prompt_for_image_selection(message, len(image_urls))
        if selection != "all":
            images_to_edit = [image_urls[selection]]

    async def _do_single_edit(img_url: str):
        edit_instruction = f"You must edit this image. {prompt}. Apply the changes to the image."
        response = await get_openai_client().responses.create(
            model="gpt-4.1",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": edit_instruction},
                        {"type": "input_image", "image_url": img_url},
                    ],
                }
            ],
            tools=[{"type": "image_generation", "action": "edit"}],
        )

        image_calls = [o for o in response.output if o.type == "image_generation_call"]
        if image_calls and image_calls[0].result:
            return io.BytesIO(base64.b64decode(image_calls[0].result))
        return None

    edited_count = 0
    for idx, img_url in enumerate(images_to_edit):
        label = f"Editing ({idx+1}/{len(images_to_edit)})" if len(images_to_edit) > 1 else "Editing"
        status_msg, image_data = await live_status_with_progress(
            message,
            action_label=label,
            emoji="🔧",
            coro=_do_single_edit(img_url),
            duration_estimate=30,
        )

        if image_data:
            edited_count += 1
            await status_msg.edit(content=f"✅ Image {idx+1} edited" if len(images_to_edit) > 1 else "✅ Image edited")
            await message.channel.send(file=discord.File(image_data, f"edited_{idx+1}.png"))
        else:
            await status_msg.edit(content=f"❌ Image {idx+1} failed" if len(images_to_edit) > 1 else "❌ Edit failed")

    if len(images_to_edit) > 1 and edited_count > 0:
        await message.channel.send(f"✅ Done! Edited {edited_count}/{len(images_to_edit)} images.")


async def handle_describe_image_intent(
    *,
    message,
    prompt: str,
    image_urls,
    ref_msg,
    is_reply_to_bot: bool,
    duration_estimate: int,
    stream_ok: bool,
    live_status_with_progress,
    send_or_edit_with_truncation,
):
    describe_injection = (
        "When asked to describe an image:\n"
        "- Identify the setting, subjects, and any visible text (transcribe briefly).\n"
        "- If humor/irony/meme is implied, explain *why* it's funny or incongruous.\n"
        "- Point to 2–3 specific visual cues that support your explanation.\n"
        "- Keep it concise and concrete."
    )
    if ref_msg and (ref_msg.content or "").strip():
        if is_reply_to_bot:
            reply_context = f"You are responding to your previous message:\n---\n{ref_msg.content.strip()}\n---"
        else:
            reply_context = f"User is replying to this message:\n---\nFrom: {ref_msg.author.display_name}\n{ref_msg.content.strip()}\n---"
    else:
        reply_context = ""

    async def _describe():
        msgs = [
            {"role": "system", "content": "Describe these images concisely. If text exists, transcribe it."},
            {"role": "system", "content": describe_injection},
        ]
        if reply_context:
            msgs.append({"role": "system", "content": reply_context})
        msgs.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}] + [{"type": "image_url", "image_url": {"url": u}} for u in image_urls],
        })
        return await generate_openai_messages_response_with_tools(msgs, tools=[])

    status_msg, response = await live_status_with_progress(
        message,
        action_label="Describing",
        emoji="🖼️",
        coro=_describe(),
        duration_estimate=duration_estimate,
        summarizer=(lambda: "Looking at visual elements… noting layout/text…") if stream_ok else None,
    )

    if not response:
        await status_msg.edit(content="❌ Generation failed.")
        return

    if isinstance(response, tuple):
        text_resp, artifacts = response
    else:
        text_resp, artifacts = response, []

    if text_resp and text_resp.strip():
        await send_or_edit_with_truncation(
            text_resp,
            target_msg=status_msg,
            original_message=message,
            model="gpt-4o-vision",
        )

    if artifacts:
        files = []
        for i, (data, mime) in enumerate(artifacts):
            ext = mimetypes.guess_extension(mime) or ".png"
            f = io.BytesIO(data)
            files.append(discord.File(f, filename=f"artifact_{i}{ext}"))
        if files:
            try:
                await status_msg.reply(files=files)
            except Exception as e:
                logger.error(f"Failed to send artifacts: {e}")
                await status_msg.reply("⚠️ Failed to upload generated artifacts.")


async def send_debug_context(message, bot_user):
    try:
        msgs = build_chat_context(
            message=message,
            user_id=str(message.author.id),
            raw_prompt=message.content.replace("/debug_context", "").strip() or "DEBUG",
            ref_msg=message.reference.resolved if message.reference else None,
            is_reply_to_bot=(message.reference.resolved.author.id == bot_user.id) if message.reference and message.reference.resolved else False,
        )
        f = io.BytesIO(json.dumps(msgs, indent=2, default=str).encode("utf-8"))
        await message.reply("Here is the exact context I would send to OpenAI:", file=discord.File(f, filename="context_debug.json"))
    except Exception as e:
        await message.reply(f"Failed to build context: {e}")
