import argparse
import sys
from classifier import classify
import telegram
from PIL import Image


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
    parser = argparse.ArgumentParser(description="Check a URL")
    parser.add_argument("url", help="URL to check")
    parser.add_argument("--post", action="store_true", help="Post flag")

    args = parser.parse_args()

    if not args.url:
        print("Error: URL is required")
        sys.exit(1)

    pred, ypred = classify_url(args.url)
    print(f"{pred} - {ypred[0] * 100:.2f}%")


if __name__ == "__main__":
    main()
