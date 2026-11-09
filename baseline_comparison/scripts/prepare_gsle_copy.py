"""Case-parameterized snapshot of existing GSLE; core SDS code unchanged."""
from pathlib import Path
import shutil,difflib,json,hashlib
ROOT=Path('/mnt/sda/ubuntu/lkx/papers/paper_eli');BASE=ROOT/'baseline_comparison';source=ROOT/'gsle';dest=BASE/'vendor/gsle'
assert not dest.exists();shutil.copytree(source,dest,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
def edit(name,old,new):
 p=dest/name;t=p.read_text();assert old in t,(name,old);p.write_text(t.replace(old,new))
for name in ['reference_proposals.py','query_language.py','rank_inpainting.py','sam_foreground.py']:
 edit(name,'(root / "gsle/config.json").read_text()','Path(os.environ["GSLE_CONFIG"]).read_text()')
 if name=='reference_proposals.py':edit(name,'root / "outputs/gsle/localization_proposals"','Path(os.environ["GSLE_OUTPUT"]) / "localization_proposals"')
 else:edit(name,'out = root / "outputs/gsle"','out = Path(os.environ["GSLE_OUTPUT"])')
for name,filename in [('localize.py','localization.json'),('avp.py','avp.json'),('prepare_edit_region.py','edit_region.json')]:
 edit(name,'ROOT / "records/gsle/'+filename+'"','RECORDS / "'+filename+'"')
edit('query_language.py','out / f"langsplat_{level}/chkpnt30000.pth"','root / f"outputs/gsle/langsplat_{level}/chkpnt30000.pth"')
edit('reference_proposals.py','texts = [\n    cfg["target_text"],\n    "a black automotive assembly on a white metal support base",\n    "an industrial machine on a white cart",\n]','texts = [cfg["target_text"]]')
edit('reference_proposals.py','records.sort(key=lambda x: x["rank"], reverse=True)',
'''records.sort(key=lambda x: x["rank"], reverse=True)
# The user-confirmed original-photo SAM region selects the physical instance;
# its 3D extent is still freshly determined by the LangSplat query and depth.
selected_mask=np.load(cfg["reference_sam_mask"]).astype(bool)
yy,xx=np.where(selected_mask)
chosen={"level":cfg["reference_sam_level"],"feature_index":cfg["reference_sam_ids"][0],
    "feature_indices":cfg["reference_sam_ids"],"area_fraction":float(selected_mask.mean()),
    "centroid":[float(xx.mean()/width),float(yy.mean()/height)],
    "bbox":[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())],
    "text_scores":scores[cfg["reference_sam_ids"]].mean(0).tolist(),
    "semantic_score":float(scores[cfg["reference_sam_ids"]].mean()),
    "rank":float(scores[cfg["reference_sam_ids"]].mean()),"association":"confirmed original-photo SAM instance"}
records.insert(0,chosen)''')
edit('localize.py','reference_mask = segments[proposal["level"]] == proposal["feature_index"]',
 'reference_mask = np.isin(segments[proposal["level"]], proposal.get("feature_indices", [proposal["feature_index"]]))')
edit('rank_inpainting.py','''positives = (
    cfg["replacement_text"],
    "a yellow articulated industrial robot arm on a pedestal",
    "a single six-axis robotic manipulator with an elbow and gripper",
)
negatives = (
    "a yellow roller machine",
    "a long horizontal frame with wheels",
    "two separate industrial robot arms",
    "a disconnected mechanical assembly",
    "texture",
)''','''positives = (cfg["replacement_text"],)
negatives = ("a disconnected mechanical assembly", "multiple separate machines", "texture", "background", "floor")''')
edit('rank_inpainting.py','mean pretrained OpenCLIP relevance over three robot-arm descriptions against explicit competing machine descriptions; visual QA remains necessary','pretrained OpenCLIP relevance of the exact replacement text against generic failure/background descriptions; visual QA remains necessary')
edit('sam_foreground.py','from sam_prompts import yellow_object_prompts','')
edit('sam_foreground.py','''points, labels, prompt_record = yellow_object_prompts(
    image, masks[int(scores.argmax())]
)
refined, refined_scores, _ = predictor.predict(
    box=box, point_coords=points, point_labels=labels, multimask_output=True
)
masks = np.concatenate((masks, refined))
scores = np.concatenate((scores, refined_scores))''','''points=np.empty((0,2),np.float32);labels=np.empty((0,),np.int32)
prompt_record={"method":"projected editing-region box and exact replacement CLIP text", "box":box.tolist(),"color_rules":None}''')
edit('sam_foreground.py','positive_coverage = float(sampled[labels == 1].mean())','positive_coverage = float(sampled[labels == 1].mean()) if (labels == 1).any() else 1.0')
edit('sam_foreground.py','valid = positive_coverage >= 0.9 and negative_coverage == 0.0','valid = containment > 0.7 and not (mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())')
edit('sam_foreground.py','projected3D box plus image-derived yellow foreground and large low-saturation background points','projected3D editing box plus SAM multimask and exact replacement-text OpenCLIP ranking; no color-specific rules')
edit('provenance.py','(root / "gsle").rglob("*")','Path(__file__).parent.rglob("*")')
diff=[];hashes={}
for path in sorted(dest.glob('*.py')):
 old=(source/path.name).read_text();new=path.read_text()
 hashes[path.name]={'source':hashlib.sha256(old.encode()).hexdigest(),'run':hashlib.sha256(new.encode()).hexdigest()}
 if old!=new:diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='gsle/'+path.name,tofile='baseline_comparison/vendor/gsle/'+path.name))
(BASE/'records/gsle_case_adaptation.patch').write_text(''.join(diff))
(BASE/'records/gsle_snapshot.json').write_text(json.dumps(hashes,indent=2))
assert hashes['train_object.py']['source']==hashes['train_object.py']['run']
assert hashes['diffusion.py']['source']==hashes['diffusion.py']['run']
print('GSLE snapshot ready; train_object.py and diffusion.py byte-identical; original gsle untouched')
