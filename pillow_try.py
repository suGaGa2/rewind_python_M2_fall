from PIL import Image, ImageDraw
import requests
import os


ttfontname = "./logotypejp_mp_m_1.1.ttf"
fontsize = 4
font = ImageFont.truetype(ttfontname, fontsize)

width = 691.0448113144344 - 574.0448113144346
campus = Image.new('RGBA', (), （0, 0, 0)
im2 = Image.open(filename_image).resize((20, 20))
textRGB = (20, 20, 20)
text = "Avicii"
draw.text((text_position_x, text_position_y), text, fill=textRGB, font=font)

draw = ImageDraw.Draw(im)
draw.rectangle((200, 100, 300, 200), fill=(0, 192, 192), outline=(255, 255, 255))

im.save('pillow_imagedraw.jpg', quality=95)