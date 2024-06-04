import model_loader
import pipeline
from PIL import Image
from transformers import CLIPTokenizer
import torch
import matplotlib.pyplot as plt


DEVICE = "cpu"

ALLOW_CUDA = False
ALLOW_MPS = True

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif torch.backends.mps.is_available() and ALLOW_MPS:
    DEVICE = "mps"


tokenizer = CLIPTokenizer('../data/tokenizer_vocab.json', merges_file='../data/tokenizer_merges.txt')
model_file = '../data/v1-5-pruned-emaonly.ckpt'
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

prompt = "a cat on a table"
uncond = ""
do_cfg = True
cfg_scale = 8

# Image to Image
# input_image = Image.open("../data/cat.jpeg")
input_image = None
strength = 0.9

sampler = "ddpm"
num_inference_steps = 50
seed = 42

output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    models=models,
    seed=seed,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer
)

plt.imshow(output_image)
plt.show()
