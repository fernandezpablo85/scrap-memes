from PIL import Image, ImageDraw, ImageFont


def rounded_rectangle(draw, xy, radius, fill):
    x1, y1, x2, y2 = xy
    draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill)
    draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill)
    draw.ellipse([x1, y1, x1 + radius * 2, y1 + radius * 2], fill=fill)
    draw.ellipse([x2 - radius * 2, y1, x2, y1 + radius * 2], fill=fill)
    draw.ellipse([x1, y2 - radius * 2, x1 + radius * 2, y2], fill=fill)
    draw.ellipse([x2 - radius * 2, y2 - radius * 2, x2, y2], fill=fill)


def add_tag_and_title(img: Image, title: str, font_size=24):
    padding = 10
    font = ImageFont.truetype("fonts/Lato-Regular.ttf", font_size)

    # Title with text wrapping and shadow
    draw = ImageDraw.Draw(img)
    max_width = img.width - (padding * 2)

    lines = []
    words = title.split()
    current_line = []
    current_width = 0

    for word in words:
        word_width = font.getbbox(word)[2]
        if current_width + word_width <= max_width:
            current_line.append(word)
            current_width += word_width + font.getbbox(" ")[2]
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_width = word_width + font.getbbox(" ")[2]

    if current_line:
        lines.append(" ".join(current_line))

    line_height = font.getbbox("Aj")[3] + 5
    total_height = line_height * len(lines)
    y = img.height - total_height - 15

    for i, line in enumerate(lines):
        # Shadow position offset slightly
        draw.text((22, y + 2), line, font=font, fill=(0, 0, 0, 90))
        draw.text((20, y), line, font=font, fill="white")
        y += line_height

    return img


if __name__ == "__main__":
    result = add_tag_and_title("pepe-kikiriki.jpg", "A LEVANTARSE GIL LABURANTE")
    result.save("output-pepe.jpg")
