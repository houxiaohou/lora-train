import glob
import json
import os

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_ID = 'Salesforce/blip-image-captioning-large'

processor = BlipProcessor.from_pretrained(MODEL_ID, cache_dir='/workspace/model')
model = BlipForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    cache_dir='/workspace/model'
).to("cuda")


def do_caption(folder):
    """读取指定文件夹的图片，识别内容写入文件"""
    current_dir = os.path.dirname(__file__)
    full_dir = os.path.join(current_dir, 'dataset', folder)
    meta_file = os.path.join(full_dir, 'metadata.jsonl')
    image_files = glob.glob(os.path.join(full_dir, "*.jpg")) + glob.glob(os.path.join(full_dir, "*.jpeg")) + glob.glob(
        os.path.join(full_dir, "*.png"))
    with open(meta_file, 'w') as outfile:
        for image_file in image_files:
            image_full_name = os.path.basename(image_file)
            image_name = os.path.splitext(image_full_name)[0]
            raw_image = Image.open(image_file).convert('RGB')
            inputs = processor(raw_image, 'a photography of', return_tensors="pt").to("cuda", torch.float16)
            out = model.generate(**inputs, num_beams=4, max_length=256, min_length=24)
            caption = processor.decode(out[0], skip_special_tokens=True)
            caption = caption.replace('a photography of', 'TOK')
            text_file = os.path.join(full_dir, f'{image_name}.txt')
            print(caption)
            result = {'file_name': image_full_name, 'text': caption}
            json.dump(result, outfile)
            outfile.write('\n')
            with open(text_file, 'w') as f:
                f.write(caption)
                f.close()
    outfile.close()


do_caption('ju')
