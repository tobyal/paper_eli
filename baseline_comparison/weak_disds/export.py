"""Render and verify the actual color-only DI-SDS checkpoint, without training."""
import shutil
from PIL import ImageDraw, ImageOps
from common import *
from policy import FROZEN


def montage(path, files, labels, columns=3):
    image=Image.new("RGB",(512*columns,544*((len(files)+columns-1)//columns)),"white")
    draw=ImageDraw.Draw(image)
    for i,(file,label) in enumerate(zip(files,labels)):
        x,y=(i%columns)*512,(i//columns)*544
        draw.text((x+8,y+8),label,fill="black")
        img=ImageOps.contain(Image.open(file).convert("RGB"),(512,512))
        image.paste(img,(x+(512-img.width)//2,y+32+(512-img.height)//2))
    image.save(path)


def main():
    cfg=config(); source=Path(cfg["weak_disds"]["source"])
    coarse_state=torch.load(source/"coarse/final.pth",map_location="cpu")
    final_state=torch.load(OUT/"refine/final.pth",map_location="cpu")
    frozen={key:torch.equal(coarse_state["state"][key],final_state["state"][key]) for key in FROZEN}
    assert all(frozen.values()),frozen
    assert coarse_state["transform"]==final_state["transform"]
    n=len(coarse_state["state"]["xyz"])
    assert len(final_state["state"]["xyz"])==n
    color_delta=(coarse_state["state"]["color_logit"].sigmoid()-final_state["state"]["color_logit"].sigmoid()).abs()
    record=dict(case=source.name,iterations=final_state["step"],gaussians=n,
        sds_weight=cfg["refine"]["sds_weight"],color_lr=cfg["optimizer"]["color_lr"],
        frozen_parameters_bitwise_equal=frozen,transform_identical=True,
        trainable_parameters=["color_logit"],color_rgb_mean_abs_change=float(color_delta.mean()),
        color_rgb_max_abs_change=float(color_delta.max()),views=[])
    del coarse_state,final_state,color_delta
    anchor=Camera.from_record(json.loads((OUT/"avp/anchor.json").read_text())["camera"])
    cameras=[anchor]+[Camera.from_record(x) for x in json.loads((OUT/"localization/review_cameras.json").read_text())]
    coarse=GaussianData.load(source/"coarse/final_object_world.ply")
    weak=GaussianData.load(OUT/"refine/final_object_world.ply")
    scene=GaussianData.load(OUT/"final/scene.ply")
    coarse_scene=None
    white=torch.ones(3,device="cuda")
    with torch.no_grad():
        for i,cam in enumerate(cameras):
            before=render(cam,coarse,background=white)
            after=render(cam,weak,background=white)
            mask=before["alpha"]>0.01
            iou=float(((before["alpha"]>0.5)&(after["alpha"]>0.5)).sum()/((before["alpha"]>0.5)|(after["alpha"]>0.5)).sum().clamp_min(1))
            row=dict(camera=cam.record(),alpha_max_abs_change=float((before["alpha"]-after["alpha"]).abs().max()),
                depth_max_abs_change=float((before["depth"]-after["depth"]).abs().max()),silhouette_iou=iou,
                rendered_rgb_mean_abs_change=float((before["rgb"]-after["rgb"]).abs()[mask.expand_as(before["rgb"])].mean()))
            record["views"].append(row)
            save_image(OUT/f"coarse_view_{i}.png",before["rgb"])
            save_image(OUT/f"weak_view_{i}.png",after["rgb"])
            save_image(OUT/f"weak_scene_view_{i}.png",render(cam,scene,aux=False)["rgb"])
            if i==0:
                gt=torch.as_tensor(np.array(Image.open(OUT/"foreground/reference_white.png").convert("RGB")).copy()/255,device="cuda",dtype=torch.float32).permute(2,0,1)
                record["anchor_rgb_mse"]={"coarse":float((before["rgb"]-gt).square().mean()),"weak":float((after["rgb"]-gt).square().mean())}
                save_image(OUT/"coarse_alpha.png",before["alpha"])
                save_image(OUT/"weak_alpha.png",after["alpha"])
                for tag,result in [("coarse",before),("weak",after)]:
                    rgb=result["rgb"]
                    saturation=(rgb.max(0).values-rgb.min(0).values)/rgb.max(0).values.clamp_min(1e-6)
                    record[tag+"_anchor_high_saturation_fraction"]=float((saturation[mask.squeeze(0)]>0.8).float().mean())
        del scene
        coarse_scene=GaussianData.load(source/"composition/scene.ply")
        save_image(OUT/"coarse_scene.png",render(anchor,coarse_scene,aux=False)["rgb"])
    for src,dst in [("coarse_final.png","coarse_reference.png"),("refine_final.png","original_disds_reference.png"),
        ("scene_final.png","original_disds_scene.png"),("foreground_white.png","foreground_reference.png")]:
        shutil.copy2(source/src,OUT/dst)
    montage(OUT/"comparison.png",[OUT/"coarse_view_0.png",OUT/"original_disds_reference.png",OUT/"weak_view_0.png"],
        ["Coarse 6500","Original DI-SDS 2000",f"Weak DI-SDS {record['iterations']}"])
    montage(OUT/"scene_comparison.png",[OUT/"coarse_scene.png",OUT/"original_disds_scene.png",OUT/"weak_scene_view_0.png"],
        ["Coarse composed scene","Original DI-SDS composed scene","Weak DI-SDS composed scene"])
    montage(OUT/"multiview.png",[OUT/f"{stage}_view_{i}.png" for i in range(1,len(cameras)) for stage in ("coarse","weak")],
        [f"{stage} / original camera {i}" for i in range(1,len(cameras)) for stage in ("coarse","weak")],columns=2)
    montage(OUT/"scene_multiview.png",[OUT/f"weak_scene_view_{i}.png" for i in range(1,len(cameras))],
        [f"Weak scene / original camera {i}" for i in range(1,len(cameras))])
    step_images=[OUT/"coarse_view_0.png"]+[OUT/f"refine/previews/reference_{step:06d}.png" for step in (100,500,1000) if step<=record["iterations"]]
    montage(OUT/"progress.png",step_images,["Coarse 6500"]+[f"Weak {step}" for step in (100,500,1000) if step<=record["iterations"]],columns=len(step_images))
    composition=json.loads((OUT/"final/composition.json").read_text())
    record["background_bitwise_identical"]=composition["retained_background_bitwise_identical"]
    record["geometry_checks_passed"]=all(v["alpha_max_abs_change"]==0 and v["depth_max_abs_change"]==0 for v in record["views"])
    assert record["geometry_checks_passed"]
    assert record["background_bitwise_identical"]
    record["outputs"]={"object_local":str(OUT/"refine/final_object_local.ply"),"object_world":str(OUT/"refine/final_object_world.ply"),"scene":str(OUT/"final/scene.ply"),"checkpoint":str(OUT/"refine/final.pth")}
    save_json(OUT/"records/verification.json",record)
    cards="".join(f'<h2>{label}</h2><a href="{file}"><img src="{file}" loading="lazy"></a>' for file,label in [("progress.png","Coarse / weak 100 / 500 / 1000"),("comparison.png","Coarse / original DI-SDS / weak DI-SDS"),("scene_comparison.png","Same anchor scene comparison"),("multiview.png","Three original cameras: Coarse vs weak"),("scene_multiview.png","Weak final scene: three original cameras")])
    (OUT/"index.html").write_text('<!doctype html><html><meta charset="utf-8"><title>Weak DI-SDS '+source.name+'</title><style>body{font:16px/1.6 system-ui;max-width:1550px;margin:auto;padding:24px;background:#f4f6f8}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>'+source.name+' / Weak DI-SDS</h1><p>Independent conservative experiment. Frozen xyz, scale, rotation, opacity. Color LR 0.001. SDS weight '+str(cfg['refine']['sds_weight'])+' . Native diffusion, depth and inpainting conditions unchanged.</p><p><a href="records/verification.json">Verification and exact PLY paths</a> | <a href="config.json">Effective config</a> | <a href="records/status.json">Status</a></p>'+cards+'<h2>Server outputs</h2><pre>'+json.dumps(record["outputs"],indent=2)+'</pre></html>')
    print(json.dumps(record),flush=True)


if __name__=="__main__":main()
