from PIL import Image
import httpx
import logger
import logging
import io

logger.setup_logging()

VOXES_URL = "https://api.devox.me/voxes/getVoxes"

# make a low-effort attempt to look like a browser
COMMON_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def fetch():
    # unsure why they send this, is the same thing the browser sends.
    req_body = {"count": 20, "oldCount": 0}
    res = httpx.post(VOXES_URL, json=req_body, headers=COMMON_HEADERS)
    return res.json()


def get_image(post_id: str):
    image_path = f"https://media.devox.re/file/posts-media/{post_id}.jpeg"
    img_response = httpx.get(image_path, headers=COMMON_HEADERS)
    valid_image = img_response.status_code == 200 and img_response.headers.get(
        "Content-Type", ""
    ).startswith("image/")
    if not valid_image:
        logging.error(f"Error while PIL loading image {post_id}")
        return None
    img_bytes = img_response.content
    try:
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        logging.error(f"Error while PIL loading image {post_id}: {e}")
        return None
