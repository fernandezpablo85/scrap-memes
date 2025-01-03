import httpx
import os
import classifier
import telegram
from PIL import Image
import argparse
import io
import voxes


def main(no_post=False):
    vxs = voxes.fetch()
    # Skip if we've already seen this vox
    with open(os.path.join(os.path.dirname(__file__), "seen.txt"), "r") as f:
        seen = f.read().splitlines()

    for v in vxs["voxes"]:
        if v["filename"] in seen:
            continue
        post_id = v["filename"]
        img_response = voxes.get_image(post_id)
        img_bytes = img_response.content
        filename = os.path.basename(f"{post_id}.jpg")
        img_local_path = f"dataset/unclassified/{filename}"
        # Use the content for processing (e.g., open as an image)
        try:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            print(f"error while PIL loading image {v['filename']}")

        if img_response.status_code == 200 and img_response.headers.get(
            "Content-Type", ""
        ).startswith("image/"):
            os.makedirs("dataset/unclassified", exist_ok=True)
            with open(img_local_path, "wb") as f:
                f.write(img_bytes)
            print(f"https://devox.me/FOO/{post_id}")

            with open("seen.txt", "a") as f:
                f.write(post_id + "\n")
            pred, ypred = classifier.classify(image)
            if ypred[0] >= 0.8 and not no_post:
                telegram.send_message(
                    photo=image,
                    caption=v["title"],
                    link_url=f"https://devox.me/FOO/{post_id}",
                )
            print(pred)
            print(ypred)
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-post", action="store_true", help="Do not post to Telegram"
    )
    args = parser.parse_args()
    main(no_post=args.no_post)
