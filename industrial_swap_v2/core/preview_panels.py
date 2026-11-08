"""Layouts for saved supervision and same-camera renders; no image synthesis."""
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

def pixels(t):
    a=t.detach().cpu().numpy() if torch.is_tensor(t) else t
    if a.ndim==3 and a.shape[0]==3: a=a.transpose(1,2,0)
    return np.round(a.clip(0,1)*255).astype(np.uint8)


def board(images,labels,zoom=None):
    arrays=[pixels(a) for a in images]; h,w=arrays[0].shape[:2]
    header=max(34,round(w*.045))
    try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',max(14,round(w*.022)))
    except OSError: font=ImageFont.load_default()
    canvas=Image.new('RGB',(w*len(arrays),h+header+(h//2+30 if zoom else 0)),'white'); d=ImageDraw.Draw(canvas)
    for index,(array,label) in enumerate(zip(arrays,labels)):
        image=Image.fromarray(array); canvas.paste(image,(index*w,header)); d.text((index*w+12,10),label,fill='black',font=font)
        if zoom:
            crop=image.crop(zoom); scale=min(w/crop.width,(h//2)/crop.height)
            crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.LANCZOS)
            canvas.paste(crop,(index*w+(w-crop.width)//2,h+header+30))
    return canvas


def panel(path, images, labels, subtitles, width, title, footer):
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font = ImageFont.truetype(font_path, 23)
    small = ImageFont.truetype(font_path, 18)
    large = ImageFont.truetype(font_path, 28)
    height = round(images[0].height * width / images[0].width)
    gap, top, header = 16, 70, 70
    board = Image.new('RGB', (width * 2 + gap * 3, top + (height + header + gap) * 2 + 70), '#eef1f4')
    draw = ImageDraw.Draw(board)
    draw.text((gap, 15), title, font=large, fill='#152333')
    for i, (image, label, subtitle) in enumerate(zip(images, labels, subtitles)):
        x = gap + (i % 2) * (width + gap)
        y = top + (i // 2) * (height + header + gap)
        draw.rectangle((x, y, x + width, y + header + height), fill='white')
        draw.text((x + 10, y + 6), label, font=font, fill='#152333')
        draw.text((x + 10, y + 38), subtitle, font=small, fill='#455468')
        board.paste(image.convert('RGB').resize((width, height), Image.Resampling.NEAREST if image.mode == 'L' else Image.Resampling.LANCZOS), (x, y + header))
    draw.text((gap, board.height - 58), footer, font=small, fill='#243347')
    board.save(path)
