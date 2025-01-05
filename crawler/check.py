import argparse
import sys
from classifier import classify
from PIL import Image
import logging
import logger

logger.setup_logging()


def classify_url(url):
    import httpx
    import io

    # Get image from URL
    response = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"})
    img_bytes = response.content

    # Create PIL Image from bytes
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return classify(image)


def main():
    parser = argparse.ArgumentParser(description="Check an image")
    parser.add_argument("path", help="URL or local file path to check")

    args = parser.parse_args()

    if not args.path:
        logging.error("Error: Path is required")
        sys.exit(1)

    # Check if input is URL or local file
    if args.path.startswith(("http://", "https://")):
        pred, ypred = classify_url(args.path)
    else:
        try:
            image = Image.open(args.path).convert("RGB")
            pred, ypred = classify(image)
        except FileNotFoundError:
            logging.error(f"Error: File not found: {args.path}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error loading image: {e}")
            sys.exit(1)

    logging.info(f"{pred} - {ypred[0] * 100:.2f}%")


if __name__ == "__main__":
    main()
