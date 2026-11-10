"""Camera-only GE framing with simultaneous localization/generated visibility."""
import math
import numpy as np
import torch
import render as r


@torch.no_grad()
def choose(original, final, obj, object_flags, previous_camera, object_center, up, out):
    base=r.BASE/'target_centered_views'
    selection=r.read(base/'case04_view_adjust_v2/ge_camera_search.json')
    localization_center=np.asarray(selection['localization_center'])
    focus_ids=np.load(base/'case04_view_adjust_v2/ge_localization_framing_indices.npy')
    focus=original.subset(torch.as_tensor(focus_ids,device='cuda'))
    loc_flags=torch.zeros_like(original.xyz);loc_flags[:,1]=1
    loc_flags[torch.as_tensor(focus_ids,device='cuda'),0]=1
    ours=r.c.Camera.from_record(r.read(base/'case04/camera_params.json')['methods']['ours']['views']['main']['final_camera'])
    old_up=r.orientation(-ours.R[2],up)
    current_dir=r.orientation(-previous_camera.R[2],up)
    initial=float(np.linalg.norm(previous_camera.center-object_center))
    trials=[]
    def evaluate(cam,kind):
        loc=r.candidate_metrics(cam,focus,original,loc_flags,res=224)
        gen=r.candidate_metrics(cam,obj,final,object_flags,res=224)
        visible=r.c.raster(cam.resize(224,224),original,loc_flags)[0].cpu().numpy()
        actual=r.alpha_stats(visible)
        theta=r.orientation(-cam.R[2],up)
        yaw=r.delta_angle(theta['yaw'],old_up['yaw']);pitch=theta['pitch']-old_up['pitch']
        # Both footprints must have image margins; a visibly tiny red fragment
        # cannot satisfy the constraint merely by being inside the frustum.
        margin=[]
        for stats in [loc,gen]:
            if stats['bbox']:
                x0,y0,x1,y1=stats['bbox'];margin.append(min(x0,y0,224-x1,224-y1))
            else:margin.append(-224)
        feasible=(min(margin)>=8 and loc['capture']>=.98 and gen['capture']>=.98
                  and actual['pixels']>=220 and gen['visibility']>=.55
                  and loc['size']>=.18 and gen['size']>=.25)
        score=40*(max(0,.98-loc['capture'])+max(0,.98-gen['capture']))
        score+=8*max(0,.45-loc['visibility'])+8*max(0,.82-gen['visibility'])
        score+=1.5*abs(math.log(max(loc['size'],.001)/.4))+1.7*abs(math.log(max(gen['size'],.001)/.5))
        score+=4*max(0,1-actual['pixels']/400)+sum(3*max(0,1-m/10) for m in margin)
        score+=.012*abs(yaw)+.025*abs(pitch)+max(0,.65-gen['scene_coverage'])
        rec=dict(camera=cam.record(),kind=kind,score=score,simultaneously_framed=feasible,
            localization=loc,visible_localization=actual,generated_object=gen,
            image_margins_224=margin,ours_yaw_difference=yaw,ours_pitch_difference=pitch)
        trials.append(rec)
        return rec
    # Explore views around both real regions. A camera aimed at the new object
    # is accepted only if the localized old target is also visible and framed.
    for blend in [0.,.35,.5]:
        center=object_center*(1-blend)+localization_center*blend
        for factor in [1.,1.35,1.7]:
            for yaw in [-48,-36,-24,-12,0,12,24]:
                cam=r.camera_at(previous_camera,object_center,center,initial*factor,up,yaw,0,'case04_ge_joint_framing')
                evaluate(cam,f'joint_center_blend_{blend:.2f}')
    # Native Add and its earlier centered render are established views that can
    # naturally align old and new targets along depth, instead of hiding either.
    native=r.cpu_camera(r.read(r.BASE/'gaussianeditor_native/case04/anchor_camera.json')).resize(1024,1024)
    evaluate(native,'native_add_camera')
    old=r.c.Camera.from_record(r.read(base/'case04/camera_params.json')['methods']['gaussianeditor']['views']['main']['final_camera'])
    evaluate(old,'previous_native_object_centered')
    eligible=[x for x in trials if x['simultaneously_framed']]
    best=min(eligible or trials,key=lambda x:x['score'])
    assert eligible, 'No candidate shows both regions adequately; do not publish a failed framing.'
    main_camera=r.c.Camera.from_record(best['camera'])
    pivot=(localization_center+object_center)/2
    auxiliary={}
    for n,angle in [(1,-8),(2,8)]:
        options=[]
        for factor in [1.,1.15,1.3]:
            ac=r.camera_at(main_camera,pivot,pivot,np.linalg.norm(main_camera.center-pivot)*factor,up,angle,0,f'ge_joint_aux{n}')
            options.append(evaluate(ac,f'aux{n}'))
        valid=[x for x in options if x['simultaneously_framed']]
        assert valid,f'Auxiliary {n} does not frame both targets'
        auxiliary[f'aux{n}']=min(valid,key=lambda x:x['score'])
    top=sorted(eligible,key=lambda x:x['score'])[:4]
    all_ids=np.load(r.BASE/'gaussianeditor_native/case04/localization/selected_indices.npy')
    idx=torch.as_tensor(all_ids,device='cuda')
    for n,rec in enumerate(top):
        cam=r.c.Camera.from_record(rec['camera']);colors=original.colors(cam.center)
        colors[idx]=.4*colors[idx]+.6*colors.new_tensor([1.,0.,0.])
        r.c.save_image(out/f'joint_candidate_{n}_localization.png',r.c.raster(cam,original,colors))
        r.c.save_image(out/f'joint_candidate_{n}_final.png',r.c.render(cam,final,aux=False)['rgb'])
    result=dict(selected=best,candidates=trials,feasible_count=len(eligible),
        auxiliary_views=auxiliary,
        localization_center=localization_center.tolist(),generated_center=object_center.tolist(),
        framing_indices_source=str(base/'case04_view_adjust_v2/ge_localization_framing_indices.npy'),
        policy='Both actual localized old target and generated object must be visibly inside the same camera, with margins. All native localization indices remain in red renders; no model movement.')
    r.save(out/'joint_framing_search.json',result)
    return r.c.Camera.from_record(best['camera']),result
