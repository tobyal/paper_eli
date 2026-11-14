"""Shared vector typography and 300-DPI PDF/PNG export (system Python)."""
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageOps
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli')
OUT=ROOT/'industrial_swap_v2/outputs/paper_figures_v3'
pdfmetrics.registerFont(TTFont('PaperSans','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('PaperBold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))


def read(p):return json.loads(Path(p).read_text())
def save(p,value):Path(p).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def source(p):
    p=Path(p);return dict(path=str(p),resolved=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size)


def grid(dest,title,rows,column_titles=None,row_titles=None,cell_labels=None,footer=None,width=510.24,left=0,image_ratio=1.,fontsize=8.,title_fontsize=10.):
    """Rows are native-resolution raster panels; layout never changes image data."""
    dest=Path(dest);cols=len(rows[0]);gap=5.;margin=10.
    cw=(width-2*margin-left-(cols-1)*gap)/cols;ch=cw*image_ratio
    head=28.+(26. if column_titles else 0.)
    labelh=25. if cell_labels else 0.
    rowh=ch+labelh+10.
    height=head+len(rows)*rowh+10.+(16. if footer else 0.)
    c=canvas.Canvas(str(dest.with_suffix('.pdf')),pagesize=(width,height),pageCompression=1)
    c.setTitle(title);c.setAuthor('paper_eli - existing experimental renders')
    c.setFillColorRGB(0,0,0);c.setFont('PaperBold',title_fontsize)
    assert pdfmetrics.stringWidth(title,'PaperBold',title_fontsize)<=width-2*margin,title
    c.drawString(margin,height-17,title)
    def lines(text,x,y,w,size=fontsize,bold=False):
        name='PaperBold' if bold else 'PaperSans';c.setFont(name,size)
        for i,line in enumerate(text.split('\n')):
            assert pdfmetrics.stringWidth(line,name,size)<=w+1,(line,w)
            c.drawString(x,y-i*(size+2),line)
    if column_titles:
        for j,text in enumerate(column_titles):lines(text,margin+left+j*(cw+gap),height-37,cw,bold=True)
    for i,row in enumerate(rows):
        top=height-head-i*rowh
        if row_titles:lines(row_titles[i],margin,top-18,left-5,fontsize,bold=True)
        for j,p in enumerate(row):
            x=margin+left+j*(cw+gap)
            if cell_labels:lines(cell_labels[i][j],x,top-8,cw,fontsize,bold=True)
            y=top-labelh-ch
            im=Image.open(p);iw,ih=im.size
            scale=min(cw/iw,ch/ih);dw,dh=iw*scale,ih*scale
            c.drawImage(str(p),x+(cw-dw)/2,y+(ch-dh)/2,dw,dh,mask='auto')
    if footer:lines(footer,margin,10,width-2*margin,6.5)
    c.showPage();c.save()
    subprocess.run(['pdftoppm','-singlefile','-r','300','-png',str(dest.with_suffix('.pdf')),str(dest)],check=True)
    im=Image.open(dest.with_suffix('.png'));im.save(dest.with_suffix('.png'),dpi=(300,300))
    preview=ImageOps.contain(im,(1600,2600));preview.save(dest.parent/(dest.name+'_preview.jpg'),quality=94)
    return dict(width_points=width,height_points=height,dpi=300,pixels=list(im.size),font='DejaVu Sans, embedded vector labels',panel_count=len(rows)*cols)
