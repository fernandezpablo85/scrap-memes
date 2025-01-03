import httpx

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
    return httpx.get(image_path, headers=COMMON_HEADERS)
