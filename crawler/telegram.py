import os
import httpx
import io
from dotenv import load_dotenv
from PIL import Image


def send_message(photo: Image, caption, link_url):
    # Load .env from parent directory
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(dotenv_path)

    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
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
        print("Photo posted successfully")
    else:
        print(f"Failed to post photo: {response.status_code}, {response.text}")
