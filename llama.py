import os
os.environ['HF_HOME'] = '/cmlscratch/zche/.cache/huggingface'
# Import stuff
import torch
import tqdm.auto as tqdm
import plotly.express as px

from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from jaxtyping import Float

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer

access_token = ""


LLAMA_2_7B_CHAT_PATH = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = LlamaTokenizer.from_pretrained(LLAMA_2_7B_CHAT_PATH,token=access_token)
hf_model = LlamaForCausalLM.from_pretrained(LLAMA_2_7B_CHAT_PATH, low_cpu_mem_usage=True,token=access_token)

model = HookedTransformer.from_pretrained(LLAMA_2_7B_CHAT_PATH, device="cpu", fold_ln=False, center_writing_weights=False, center_unembed=False)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
model.generate("The capital of Germany is", max_new_tokens=20, temperature=0)

prompts = [
    "The capital of Germany is",
    "2 * 42 = ", 
    "My favorite", 
    "aosetuhaosuh aostud aoestuaoentsudhasuh aos tasat naostutshaosuhtnaoe usaho uaotsnhuaosntuhaosntu haouaoshat u saotheu saonuh aoesntuhaosut aosu thaosu thaoustaho usaothusaothuao sutao sutaotduaoetudet uaosthuao uaostuaoeu aostouhsaonh aosnthuaoscnuhaoshkbaoesnit haosuhaoe uasotehusntaosn.p.uo ksoentudhao ustahoeuaso usant.hsa otuhaotsi aostuhs",
]

model.eval()
hf_model.eval()
prompt_ids = [tokenizer.encode(prompt, return_tensors="pt") for prompt in prompts]
tl_logits = [model(prompt_ids).detach().cpu() for prompt_ids in tqdm(prompt_ids)]

# hf logits are really slow as it's on CPU. If you have a big/multi-GPU machine, run `hf_model = hf_model.to("cuda")` to speed this up
logits = [hf_model(prompt_ids).logits.detach().cpu() for prompt_ids in tqdm(prompt_ids)]

for i in range(len(prompts)): 
    assert torch.allclose(logits[i], tl_logits[i], atol=1e-4, rtol=1e-2)