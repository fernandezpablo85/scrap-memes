import os
import httpx
import io
import logging
import logger
from PIL import Image

logger.setup_logging()


def send_message(photo: Image, caption, link_url, chat_id=None):

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if chat_id is None:
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    caption = f"[{caption}]({link_url})"

    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    photo.save(img_byte_arr, format="JPEG")
    img_byte_arr = img_byte_arr.getvalue()

    files = {"photo": ("image.jpg", img_byte_arr, "image/jpeg")}
    payload = {
        "chat_id": chat_id,
        "caption": caption,
        "parse_mode": "Markdown",
    }

    response = httpx.post(url, data=payload, files=files)
    if response.status_code == 200:
        logging.info("Photo posted successfully to {chat_id}")
    else:
        logging.error(f"Failed to post photo: {response.status_code}, {response.text}")
