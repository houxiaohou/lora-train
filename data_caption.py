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

for i in range(1, 8):
    raw_image = Image.open(f'./dataset/dazhi/{i}.jpg').convert('RGB')
    inputs = processor(raw_image, 'a photography of', return_tensors="pt").to("cuda", torch.float16)
    out = model.generate(**inputs, num_beams=4, max_length=256, min_length=24)
    print(processor.decode(out[0], skip_special_tokens=True))
