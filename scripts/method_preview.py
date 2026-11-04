"""Actual RGB target crops for visual inspection; left source, right edited."""
import os,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
root=Path(os.environ['ROOT']);method=sys.argv[1]
frames=[61,91,151,161,281]
out=root/'outputs'/method/'quality_preview.png'
sheet=Image.new('RGB',(760,5*400+30),'white');draw=ImageDraw.Draw(sheet)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
draw.text((10,3),'Unedited source',font=font,fill='black');draw.text((390,3),method+' actual RGB',font=font,fill='black')
for i,frame in enumerate(frames):
 name=f'frame_{frame:06d}.png'
 mask=np.asarray(Image.open(root/'outputs/evaluation_masks/wide'/name))>127
 y,x=np.where(mask);cx=(x.min()+x.max())/2;cy=(y.min()+y.max())/2
 side=max(100,max(np.ptp(x),np.ptp(y))*1.55)
 box=(round(cx-side/2),round(cy-side/2),round(cx+side/2),round(cy+side/2))
 for j,m in enumerate(['reconstruction',method]):
  image=Image.open(root/f'outputs/{m}/common_render/wide'/name).convert('RGB').crop(box).resize((380,380),Image.Resampling.LANCZOS)
  sheet.paste(image,(j*380,30+i*400))
 draw.text((10,410+i*400),f'Camera{frame:03d}',font=font,fill='black')
sheet.save(out)
print(out)
