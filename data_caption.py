import json

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

MODEL_ID = 'Salesforce/blip-image-captioning-large'

print('start model load...')

processor = BlipProcessor.from_pretrained(MODEL_ID, cache_dir='/workspace/model')
model = BlipForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    cache_dir='/workspace/model'
).to("cuda")

print('model loaded....')

results = []

with open('./dataset/dazhi/metadata.jsonl', 'w') as outfile:
    for i in range(1, 8):
        raw_image = Image.open(f'./dataset/dazhi/{i}.jpg').convert('RGB')
        inputs = processor(raw_image, 'a photography of', return_tensors="pt").to("cuda", torch.float16)
        out = model.generate(**inputs, num_beams=4, max_length=256, min_length=24)
        caption = processor.decode(out[0])
        result = {'file_name': f'{i}.jpg', 'text': caption}
        json.dump(result, outfile)
        outfile.write('\n')
    outfile.close()
