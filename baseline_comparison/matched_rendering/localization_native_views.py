"""Supplement red localization with each method's saved native reference view."""
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
import localization_red as lr

r=lr.r


def native_camera(row, method):
    if method=='gaussianeditor':
        path=r.BASE/'gaussianeditor_native'/row['case']/'anchor_camera.json'
        record=r.read(path)
        key='native selected localization/Add camera'
    elif method=='dge':
        path=r.BASE/'dge_native'/row['case']/'records/localization.json'
        record=r.read(path)['review_cameras'][0]
        key='review_cameras[0]'
    else:
        path=(r.Path(row['target_record']) if method=='ours' else
              r.BASE/'gsle_native'/row['case']/'target_bbox.json')
        record=r.read(path)['reference_camera']
        key='reference_camera'
    cam=r.cpu_camera(record)
    scale=1024/max(cam.width,cam.height)
    # Preserve native aspect ratio and camera field of view; figure panels use
    # white letterboxing, never anisotropic image stretching or geometry edits.
    return cam.resize(round(cam.width*scale),round(cam.height*scale)),path,key


@torch.no_grad()
def main():
    torch.set_num_threads(6);torch.cuda.set_per_process_memory_fraction(.45)
    spec=r.read(r.BASE/'prompts.json');before=r.sha(spec['source_scene'])
    scene=r.c.GaussianData.load(spec['source_scene'])
    for row in spec['cases']:
        case=row['case'];out=lr.OUT/case;folder=out/'views/native_localization';folder.mkdir(parents=True,exist_ok=True)
        report=r.read(out/'localization_report.json');panels=[('Original',None,None)]
        for method,label in zip(lr.METHODS,lr.LABELS):
            path=lr.index_path(row,method);assert r.sha(path)==report['methods'][method]['indices_sha256']
            idx=torch.as_tensor(np.load(path),device='cuda')
            camera,source,key=native_camera(row,method)
            colors=scene.colors(camera.center)
            r.c.save_image(folder/('original_'+method+'.png'),r.c.raster(camera,scene,colors))
            colors[idx]=colors.new_tensor([1.,0.,0.])
            r.c.save_image(folder/(method+'.png'),r.c.raster(camera,scene,colors))
            flags=torch.zeros_like(colors);flags[idx]=1
            contrib=r.c.raster(camera,scene,flags)[0]
            r.c.save_image(folder/(method+'_visible_selection.png'),contrib[None])
            report['methods'][method]['views']['native_localization']=dict(camera=camera.record(),
                source=str(source),source_key=key,source_sha256=r.sha(source),
                visible_selection=r.alpha_stats(contrib.cpu().numpy()))
            panels.append((label,folder/(method+'.png'),None))
            print(case,method,'native',report['methods'][method]['views']['native_localization']['visible_selection']['pixels'],flush=True)
        panels[0]=('Original',folder/'original_ours.png',None)
        lr.board(out/'comparison_native_localization.png',[panels],case+' | Native localization reference cameras',
                 'Each method uses its own saved localization view | Red = all selected Gaussians')
        r.save(out/'localization_report.json',report)
        page=(out/'index.html').read_text(encoding='utf-8')
        page=page.replace('</html>','<section><h2>各方法原生定位参考视角</h2><p>使用各方法记录中的定位相机，观察方向可以不同；保留原始宽高比。Original 使用 Ours 的 M1 参考相机。</p><a href="comparison_native_localization.png"><img src="comparison_native_localization.png"></a></section></html>')
        (out/'index.html').write_text(page,encoding='utf-8')
    assert r.sha(spec['source_scene'])==before
    overview=Image.new('RGB',(2500,90+540*len(lr.CASES)),'white');d=ImageDraw.Draw(overview)
    d.text((16,12),'Native localization in red | Per-method reference cameras',font=ImageFont.truetype('DejaVuSans.ttf',27),fill='black')
    d.text((16,50),'Original | GaussianEditor | DGE | GSLE | Ours - Native viewpoints differ; complete selections retained.',font=ImageFont.truetype('DejaVuSans.ttf',22),fill='#444444')
    for i,case in enumerate(lr.CASES):
        im=Image.open(lr.OUT/case/'comparison_native_localization.png')
        overview.paste(im.crop((0,80,2500,620)),(0,90+i*540))
        d.text((12,90+i*540+38),case,font=ImageFont.truetype('DejaVuSans.ttf',24),fill='black',stroke_width=2,stroke_fill='white')
    overview.save(lr.OUT/'overview_native_localization.png',dpi=(300,300))
    ImageOps.contain(overview,(1600,1900)).save(lr.OUT/'overview_native_localization_web.jpg',quality=94)
    p=lr.OUT/'index.html';page=p.read_text(encoding='utf-8')
    page=page.replace('</html>','<h2>各方法原生定位视角</h2><p>观察方向不同，用于检查完整定位区域。</p><a href="overview_native_localization.png"><img src="overview_native_localization_web.jpg"></a></html>')
    p.write_text(page,encoding='utf-8')
    # Package the completed matching views and this native-view supplement.
    import zipfile
    archive=lr.OUT/'localization_red_review.zip'
    temp=archive.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for item in sorted(lr.OUT.rglob('*')):
            if item.is_file() and item.suffix in ('.png','.jpg','.json','.html','.txt'):
                z.write(item,item.relative_to(lr.OUT))
    temp.replace(archive)
    print('COMPLETE native views and package',archive.stat().st_size,flush=True)


if __name__=='__main__':main()
