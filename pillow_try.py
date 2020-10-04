from PIL import Image, ImageDraw
import requests
import os

url = 'http://img.youtube.com/vi/E8gipfw9BWw/hqdefault.jpg'

response = requests.get(url)
filename_image = os.path.basename(url)
with open(filename_image, 'wb') as f:
    f.write(response.content)

im = Image.new('RGB', (1000, 1000), (128, 128, 128))
im2 = Image.open(filename_image).resize((20, 20))
im.paste(im2, (100, 50))

draw = ImageDraw.Draw(im)
draw.rectangle((200, 100, 300, 200), fill=(0, 192, 192), outline=(255, 255, 255))

im.save('pillow_imagedraw.jpg', quality=95)