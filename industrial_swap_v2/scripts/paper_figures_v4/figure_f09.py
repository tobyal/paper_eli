from figure_common import *
import shutil

def main():
    d=folder('F09_text_control')
    refined=d/'m4_2000_v2'
    if (refined/'optimization_results.json').exists() and (refined/'caption.txt').exists():
        # Prefer the completed correction; never silently regress to placement-only renders.
        for name in ('figure_F09.png','figure_F09.pdf','preview_F09.jpg','caption.txt','sources.json','data.csv'):
            shutil.copy2(refined/name,d/name)
        shutil.copytree(refined/'panels',d/'panels',dirs_exist_ok=True)
        print('F09 reused actual dual-mask 2000-step publication',flush=True)
        return
    results=read(d/'placement_results.json');fig,ax=plt.subplots(2,4,figsize=(10,6.5));fig.subplots_adjust(left=.025,right=.98,bottom=.15,top=.87,wspace=.035,hspace=.12)
    fig.suptitle('Text-Controllable Industrial Equipment Replacement',y=.97,fontsize=14,fontweight='bold')
    labels=['Six-Axis Robot','Requested SCARA Robot','CNC Machining Center','Roller Conveyor'];rows=[]
    for i,r in enumerate(results['variants']):
        p=d/'variants'/r['variant']
        for j,name in enumerate(['main_scene.png','detail.png']):
            ax[j,i].imshow(Image.open(p/name));ax[j,i].axis('off')
            if j==0:ax[j,i].set_title(labels[i],fontsize=9)
        t=r['transform'];rows.append(dict(variant=r['variant'],gaussians=r['gaussians'],scale=t['scale'],yaw_degrees=t['yaw_degrees'],
            support_center_world=str(t['support_center_world']),mask_iou=t.get('metrics',{}).get('mask_iou','NA'),
            quality_gate_passed=t.get('quality_gate_passed',True),M4_background_steps=0,scene_sha256=r['scene_sha256'],background_sha256=r['background_sha256']))
    fig.text(.025,.06,'Identical camera, crop, installation center, support plane and retained M1 background. No background refinement.\nB is the actual result of the SCARA request; its kinematic structure is not verified. C retains the observed foreground-extraction defect.',fontsize=8)
    caption='Figure 9. Text-controlled replacements at one fixed Case00 workstation. Columns show the existing yellow six-axis robot asset and three newly generated Qwen/TRELLIS.2/PBR Gaussian assets requested as a blue-white SCARA robot, light-gray CNC machining center, and blue-frame roller conveyor. All four objects use the same retained M1 background, exact installation center/support height, allowed volume and scene camera; the bottom row uses one identical pixel crop across columns. New assets use 36 PBR views and 4000 Gaussian-training iterations; only scale and yaw are fitted for placement. No background refinement or floor completion is performed. A reuses the unchanged original robot object/pose, recomposed with the shared M1 background for this experiment; the existing F05/F10 scenes are untouched. B has an articulated-arm appearance and does not establish correct SCARA kinematics. C preserves the actual incomplete control-display foreground extraction. Results are not manually repaired or replaced by 2D images. Per-variant alignment quality gates and exact sources are retained in the data and manifest.'
    adaptation=read(d/'variants/C_cnc/records/export_resource_adaptation.json')
    caption+=' CNC export uses the official decimation parameter at 250,000 target faces after the million-face UV attempt stalled; the full native TRELLIS mesh is retained, with no repeated diffusion sampling.'
    export(fig,d,'F09',caption,dict(placement=results,fixed_installation=read(d/'fixed_installation_space.json'),generation=read(d/'generation_status.json'),CNC_export_resource_adaptation=adaptation,same_camera=True,original_paper_figures_unchanged=True),rows,ax.ravel())

if __name__=='__main__':main()
