"""One-row F10 with the five panels selected in the user's marked reference.

Only layout/copy operations; no Gaussian rendering or image-content changes.
"""
import json
import shutil
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image
from relayout_f06_f07 import V3, V4, WIDTH, canvas_for, title, export, sha, record, dump

OLD=V3/'F10_multiview'
OUT=V4/'F10_selected_views'
SELECTION=[
    ('a_front','equipment_detail','(a) Front View'),
    ('b_left_oblique','full_scene','(b) Left Oblique View'),
    ('c_right_side','full_scene','(c) Right-side View'),
    ('d_rear','equipment_detail','(d) Rear View'),
    ('e_elevated','equipment_detail','(e) Elevated View'),
]


def main():
    pan=OUT/'panels';pan.mkdir(parents=True,exist_ok=True)
    protected=[p for root in [V3,V4] for p in root.glob('*/figure_*')
               if p.is_file() and p.parent!=OUT]
    before={str(p):sha(p) for p in protected}
    prior=json.loads((OLD/'sources.json').read_text())
    cameras=json.loads((OLD/'cameras.json').read_text())
    view_records={v['view']:v for v in cameras['views']}
    records=[]
    for view,group,label in SELECTION:
        src=OLD/group/(view+'.png');dst=pan/(view+'.png')
        shutil.copy2(src,dst)
        assert sha(src)==sha(dst)
        im=Image.open(dst)
        assert im.width==im.height
        camera=view_records[view]
        crop=camera['detail_crop_xyxy'] if group=='equipment_detail' else [0,0,camera['camera']['width'],camera['camera']['height']]
        records.append(dict(view=view,label=label,selected_original_row=group,
            source=record(src),output=record(dst),size=list(im.size),byte_identical=True,
            original_camera_record=camera,existing_display_crop=crop,new_crop=False))
    margin,gap,heading=3.,3.,17.
    cell=(WIDTH-2*margin-4*gap)/5
    height=2*margin+heading+cell
    stem=OUT/'figure_F10_revised'
    c=canvas_for(stem,height)
    c.setTitle('Multi-view results of the reconfigured industrial scene')
    for i,(view,group,label) in enumerate(SELECTION):
        x=margin+i*(cell+gap)
        title(c,label,x,height-margin-8,cell,size=7.4)
        c.drawImage(str(pan/(view+'.png')),x,margin,cell,cell)
    layout=export(c,stem)
    layout.update(rows=1,columns=5,width_mm=180,height_mm=height/72*25.4,
        cell_width_mm=cell/72*25.4,heading_font_pt=7.4,no_overall_title=True,
        no_row_headers=True,no_footer=True,selection_rectangles_not_in_output=True)
    caption=('Fig. 10. Multi-view results of the reconfigured industrial scene for Case00. '
        'The selected views are (a) front, (b) left oblique, (c) right side, (d) rear, '
        'and (e) elevated. Panels (a), (d), and (e) reuse the original equipment-detail '
        'crops; panels (b) and (c) reuse the original full-scene images, as selected '
        'in the marked review figure. Thus the five panels intentionally retain '
        'their different original display scales. All images show the same saved '
        'Gaussian scene used by the original F05 and F10. Original camera parameters, '
        'crop bounds, foreground occlusion, and reconstruction artifacts are unchanged. '
        'Red rectangles in the review image indicated panel selection and are not '
        'part of the experimental outputs. No re-rendering, training, image retouching, '
        'or scene modification was performed.\n')
    (OUT/'caption.txt').write_text(caption,encoding='utf-8')
    dump(OUT/'sources.json',dict(source_manifest=record(OLD/'sources.json'),
        source_cameras=record(OLD/'cameras.json'),original_sources=prior,
        selection=records,layout=layout,script=record(Path(__file__)),
        gaussian_render_calls=0,new_generation_calls=0,training_steps=0,
        camera_changes=0,new_crops=0,selection_order_confirmed='One row, five columns'))
    assert before=={str(p):sha(p) for p in protected},'An existing figure changed'
    dump(OUT/'validation.json',dict(old_figures_unchanged=True,protected_sha256=before,
        selected_panels_byte_identical=True,panel_count=5,png=layout['png'],pdf=layout['pdf']))
    (OUT/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>F10 Selected Views</title>
<style>body{font-family:Arial,sans-serif;max-width:1700px;margin:24px auto;padding:0 20px;color:#222}img{width:100%;height:auto}a{color:#185b94}</style>
<h1>F10 · 五视角横排版</h1><p>按红框选择：Front 细节、Left Oblique 全景、Right-side 全景、Rear 细节、Elevated 细节。</p>
<p><a href="figure_F10_revised.png">300 DPI PNG</a> · <a href="figure_F10_revised.pdf">PDF</a> · <a href="panels/">五张高清子图</a> · <a href="caption.txt">Caption</a> · <a href="../F10_selected_views.zip">完整包</a></p>
<img src="figure_F10_revised_preview.jpg">''',encoding='utf-8')
    with ZipFile(V4/'F10_selected_views.zip','w',ZIP_DEFLATED,compresslevel=4) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT))
    print(json.dumps(dict(output=str(OUT),pixels=layout['pixels'],rows=1,columns=5,
                         height_mm=layout['height_mm'])),flush=True)


if __name__=='__main__':main()
