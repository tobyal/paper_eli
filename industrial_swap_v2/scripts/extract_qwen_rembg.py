"""Use TRELLIS.2's existing pinned foreground extractor for reviewed SAM failures.

Run in trellis2_reused with the project HF_MODULES_CACHE and a free CUDA device.
No weights are downloaded and no generated image pixels are modified.
"""
import argparse
import importlib.util
from pathlib import Path
from PIL import Image
from industrial_swap_v2.core.context import read_json,write_json,sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True,type=Path)
    p.add_argument('--keys',nargs='+',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    source=root/'industrial_swap_v2/third_party/TRELLIS.2/trellis2/pipelines/rembg/BiRefNet.py'
    spec=importlib.util.spec_from_file_location('trellis_rembg_adapter',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    pin=read_json(root/'industrial_swap_v2/records/trellis_dependencies.json')['briaai/RMBG-2.0']
    items={i['key']:i for i in read_json(a.batch/'manifest.json')['items']}
    model=None
    for key in a.keys:
        item=items[key];out=a.batch/'rembg_review'/key;out.mkdir(parents=True,exist_ok=True)
        original=Image.open(item['input_path']).convert('RGB')
        if not (out/'object_rgba.png').exists():
            if model is None:model=module.BiRefNet(pin['snapshot_path']);model.cuda()
            rgba=model(original.copy());rgba.save(out/'object_rgba.png')
            rgba.getchannel('A').save(out/'object_mask.png')
            white=Image.new('RGBA',rgba.size,'white');white.alpha_composite(rgba)
            white.convert('RGB').save(out/'object_white.png')
        rgba=Image.open(out/'object_rgba.png')
        if rgba.convert('RGB').tobytes()!=original.tobytes():raise ValueError('Source RGB changed')
        write_json(out/'selection.json',{'method':'Official TRELLIS.2 BiRefNet adapter with pinned RMBG-2.0',
            'model':'briaai/RMBG-2.0','model_revision':pin['revision'],
            'implementation':str(source),'implementation_sha256':sha256(source),
            'source_image':item['input_path'],'source_sha256':sha256(item['input_path']),
            'sam_called':False,'manual_coordinates':False,'mask_pixels_source':'pretrained RMBG-2.0 soft alpha',
            'reason':'Initial SAM and prompt-refined masks had missing parts or retained unrelated rear equipment; all retained for audit',
            'review_status':'pending_visual_review','output_sha256':sha256(out/'object_rgba.png')})
        print(key,sha256(out/'object_rgba.png'),flush=True)


if __name__=='__main__':main()
