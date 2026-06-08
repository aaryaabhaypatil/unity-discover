from PIL import Image, ImageDraw, ImageFont
from imagehash import phash

def text_to_image(text, width=258, height=350, font_path="/Library/Fonts/Arial.ttf", font_position = 0.5, font_size=33, bg=(255,255,255, 255), fg=(0,0,0,255)):
    font = ImageFont.truetype(font_path, font_size)
    img = Image.new("RGBA", (width, height), bg)
    draw = ImageDraw.Draw(img)

    # Word wrap
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        if draw.textlength(test, font=font) <= width:
            current = test
        else:
            lines.append(current)
            current = word
    lines.append(current)

    # Measure total text block height
    line_height = font.getbbox("A")[3]*1.20
    y = height*font_position

    for line in lines:
        line_width = draw.textlength(line, font=font)
        x = (width - line_width) // 2  # horizontal center
        draw.text((x, y), line, font=font, fill=fg)
        y += line_height

    return img, y

def generate_cell(icon_path, label, bg = (255,255,255,255)):
    padding = 14

    icon_image = Image.open(icon_path)
    icon_image.convert("RGBA")

    img_cd, y = text_to_image(label, font_size=36, font_position=0.058, bg=bg)

    max_width = img_cd.width - 2 * padding
    max_height = img_cd.height - y - padding

    i_width = icon_image.width
    i_height = icon_image.height

    i_scale = min(max_width/i_width, max_height/i_height)
    icon_image = icon_image.resize((int(i_width*i_scale), int(i_height*i_scale)))
    i_width, i_height = icon_image.size

    # place icon at bottom center with padding
    iy = y + max_height/2 - i_height/2
    ix = (img_cd.width - i_width)/2

    # place icon on image
    img_cd.paste(icon_image, (int(ix), int(iy)), icon_image)

    return img_cd

def test(cell_path, icon_path, label):
    cell_image = Image.open(cell_path)
    cell_gen = generate_cell(icon_path, label)
    ci_hash = phash(cell_image)
    cg_hash = phash(cell_gen)
    return ci_hash - cg_hash