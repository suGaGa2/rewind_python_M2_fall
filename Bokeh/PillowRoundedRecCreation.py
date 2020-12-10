from PIL import Image, ImageDraw, ImageFont
import requests
import os


def round_corner(radius, fill):
    """Draw a round corner"""
    corner = Image.new('RGBA', (radius, radius), (255, 0, 0, 0))
    draw = ImageDraw.Draw(corner)
    draw.pieslice((0, 0, radius * 2, radius * 2), 180, 270, fill=fill)
    return corner


def round_rectangle(size, radius, fill):
    """Draw a rounded rectangle"""
    width, height = size
    rectangle = Image.new('RGBA', size, fill)
    corner = round_corner(radius, fill)
    rectangle.paste(corner, (0, 0))
    rectangle.paste(corner.rotate(90), (0, height - radius))  # Rotate the corner and paste it
    rectangle.paste(corner.rotate(180), (width - radius, height - radius))
    rectangle.paste(corner.rotate(270), (width - radius, 0))
    return rectangle

def word_image_creation(word, word_size, color, folder_path):
    dummy   = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
    draw_d  = ImageDraw.Draw(dummy)

    ttfontname = "./logotypejp_mp_m_1.1.ttf"
    fontsize = word_size
    font = ImageFont.truetype(ttfontname, fontsize)
    textRGB = (255,255,255)
    text = word
    textWidth, textHeight = draw_d.textsize(text,font=font)
    MARGIN = 5
    campus = Image.new('RGBA', (textWidth + MARGIN, textHeight + MARGIN), (0, 0, 0, 0))
    draw = ImageDraw.Draw(campus)

    if color ==  "RED":
        rgba = (200, 0, 0, 255)
    if color == "BLUE":
        rgba = (0, 0, 200, 255)
    if color == "PURPLE":
        rgba = (200, 0, 200, 255)
    if color ==  "NO":
        rgba = (0, 0, 0, 0)
    img = round_rectangle((textWidth + MARGIN, textHeight + MARGIN), 10, rgba)
    campus.paste( img, (0,0) )
    #draw.rectangle((0, 0, textWidth, textHeight), fill=(0, 192, 192))
    draw.text(( MARGIN/2, MARGIN/2), text, fill=textRGB, font=font)
    save_path = folder_path + "/" + word + ".png"
    campus.save(save_path, quality=95)