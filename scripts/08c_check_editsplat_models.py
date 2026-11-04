"""Load official pretrained components without performing an EditSplat edit."""
import json
import os
from pathlib import Path
from PIL import Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
import ImageReward as RM
from lang_sam import LangSAM

root=Path(os.environ['ROOT'])
core=json.loads((root/'records/core_model_downloads.json').read_text())
reward_path=Path(json.loads((root/'records/imagereward_model_downloads.json').read_text())['THUDM/ImageReward']['path'])
pipeline=StableDiffusionInstructPix2PixPipeline.from_pretrained(core['timbrooks/instruct-pix2pix']['path'],torch_dtype=torch.bfloat16).to('cuda')
print('Official IP2P pipeline loaded',flush=True)
reward=RM.load(str(reward_path/'ImageReward.pt'),med_config=str(reward_path/'med_config.json'))
print('Official ImageReward loaded',flush=True)
segmentor=LangSAM()
image=Image.open(root/'data/edit_train/images/frame_000151.jpg').convert('RGB')
with torch.no_grad():
    masks,boxes,phrases,logits=segmentor.predict(image,'white industrial machine housing')
    score=float(reward.score('A factory with an industrial machine.',image))
record={'status':'pretrained_components_loaded_and_inference_checked','not_an_editing_run':True,'sam_masks':len(masks),'boxes':boxes.tolist(),'phrases':phrases,'reward_score_for_unedited_reference':score,'torch':torch.__version__}
(root/'records/editsplat.models.checked.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2),flush=True)
