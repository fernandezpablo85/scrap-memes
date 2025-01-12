import voxes
import duplicates as dups
import logging
import logger
import telegram
import os
from dotenv import load_dotenv


logger.setup_logging()


def is_gil_laburante_or_nini(vox_title: str):
    return any(word in vox_title.lower() for word in ["laburante", "nini"])


def main():
    # Load .env from parent directory
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(dotenv_path)

    vs = voxes.fetch()
    for v in vs["voxes"]:
        vox_id = v["filename"]
        if dups.already_seen(vox_id):
            logging.info(f"Skipping already seen vox: '{vox_id}'")
            continue
        if is_gil_laburante_or_nini(v["title"]):
            image = voxes.get_image(vox_id)
            telegram.send_message(
                photo=image,
                caption=v["title"],
                link_url=f"https://devox.me/FOO/{vox_id}",
                chat_id=os.getenv("TELEGRAM_LABURANTES"),
            )


if __name__ == "__main__":
    main()
