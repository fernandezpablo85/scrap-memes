import os
import classifier
import telegram
from PIL import Image
import argparse
import io
import voxes
import logger
import logging
import duplicates as dups

# Setup logging
logger.setup_logging()


def is_close(prob: float):
    return prob >= 0.4 and prob <= 0.6


def is_certain(prob: float):
    return prob > 0.8


def main(no_post=False):
    logging.info("Fetching vox data.")
    vxs = voxes.fetch()
    unique = set(v["filename"] for v in vxs["voxes"])
    logging.info(f"fetched {len(vxs['voxes'])} voxes, {len(unique)} unique")

    for v in vxs["voxes"]:
        vox_id = v["filename"]
        if dups.already_seen(vox_id):
            logging.info(f"Skipping already seen vox: '{vox_id}'")
            continue
        logging.info(f"vox '{vox_id}' not seen, processing...")
        img_response = voxes.get_image(vox_id)
        img_bytes = img_response.content
        filename = os.path.basename(f"{vox_id}.jpg")
        img_local_path = f"dataset/unclassified/{filename}"
        # Use the content for processing (e.g., open as an image)
        try:
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            logging.error(f"Error while PIL loading image {vox_id}: {e}")
            continue

        if img_response.status_code == 200 and img_response.headers.get(
            "Content-Type", ""
        ).startswith("image/"):
            os.makedirs("dataset/unclassified", exist_ok=True)
            with open(img_local_path, "wb") as f:
                f.write(img_bytes)
            logging.info(f"Image saved: https://devox.me/FOO/{vox_id}")

            # mark as seen.
            dups.mark_already_seen(vox_id)

            pred, ypred = classifier.classify(image)

            # close to threshold, active learning opportunity.
            if is_close(ypred[0]):
                logging.info(f"😲 Close to threshold: {vox_id}")
                logging.info(f"Prediction: {pred}, Confidence: {ypred[0]:.4f}")

            # high precision, post message.
            if is_certain(ypred[0]) and not no_post:
                telegram.send_message(
                    photo=image,
                    caption=v["title"],
                    link_url=f"https://devox.me/FOO/{vox_id}",
                )
            logging.info(f"Prediction: {pred}, Confidence: {ypred[0]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-post", action="store_true", help="Do not post to Telegram"
    )
    args = parser.parse_args()
    if args.no_post:
        logging.info("Running in no-post mode - will not post to Telegram")
    main(no_post=args.no_post)
