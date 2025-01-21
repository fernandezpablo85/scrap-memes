from pathlib import Path
import classifier
import telegram
import argparse
import voxes
import logger
import logging
import duplicates as dups
from dotenv import load_dotenv
from io import BytesIO

# Setup logging
logger.setup_logging()

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent


def is_close(prob: float):
    return prob >= 0.4 and prob <= 0.6


def is_certain(prob: float):
    return prob > 0.8


def main(no_post=False, no_store=False):
    # Load .env from parent directory
    dotenv_path = PARENT_DIR / ".env"
    load_dotenv(dotenv_path)

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
        image = voxes.get_image(vox_id)
        if not image:
            continue

        pred, ypred = classifier.classify(image)

        # close to threshold, active learning opportunity.
        if is_close(ypred[0]):
            logging.info(f"😲 Close to threshold: {vox_id}")
            logging.info(f"Prediction: {pred}, Confidence: {ypred[0]:.4f}")

        # mark as seen.
        dups.mark_already_seen(vox_id, float(ypred[0]))

        # high precision, post message.
        if is_certain(ypred[0]) and not no_post:
            telegram.send_message(
                photo=image,
                caption=v["title"],
                link_url=f"https://devox.me/FOO/{vox_id}",
            )
        logging.info(f"Prediction: {pred}, Confidence: {ypred[0]:.4f}")

        if not no_store:
            triage_dir = CURRENT_DIR / "triage"
            triage_dir.mkdir(exist_ok=True)
            folder = "class1" if ypred[0] > 0.5 else "class2"
            (triage_dir / folder).mkdir(exist_ok=True)
            image_path = triage_dir / folder / f"{vox_id}.jpg"
            with open(image_path, "wb") as f:
                img_bytes = BytesIO()
                image.save(img_bytes, format="JPEG")
                f.write(img_bytes.getvalue())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-post", action="store_true", help="Do not post to Telegram"
    )
    parser.add_argument(
        "--no-store", action="store_true", help="Do not save files to local disk"
    )
    args = parser.parse_args()
    if args.no_post:
        logging.info("Running in no-post mode - will not post to Telegram")
    if args.no_store:
        logging.info("Running in no-store mode - will not save files to local disk")
    main(no_post=args.no_post, no_store=args.no_store)
