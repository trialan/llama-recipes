import sys
sys.path = sorted(sys.path) #Hack: avoids import of wrong datasets
if "" in sys.path:
    sys.path.remove("")
if "." in sys.path:
    sys.path.remove(".")
sys.path.append(".")

from llama_recipes.utils.config_utils import generate_peft_config
from llama_recipes.configs.training import train_config



"""
Reproduce train config from run command:

python -m llama_recipes.finetuning --dataset "biblechat_dataset" --custom_dataset.file "examples/biblechat_dataset.py" --use_peft --peft_method lora --quantization --model_name "meta-llama/Llama-2-7b-chat-hf" --output_dir "/home/paperspace/tr_models"

torchrun --nnodes 1 --nproc_per_node 1 examples/finetuning.py --dataset "biblechat_dataset" --custom_dataset.file "examples/biblechat_dataset.py" --use_peft --peft_method lora --quantization --model_name "meta-llama/Llama-2-7b-chat-hf" --output_dir "/home/paperspace/tr_models" --pure_bf16 --use_fast_kernels

torchrun --nnodes 1 --nproc_per_node 4  examples/finetuning.py --enable_fsdp --use_peft --peft_method lora --model_name /patht_of_model_folder/7B --pure_bf16 --output_dir Path/to/save/PEFT/model --use_fast_kernels

Inference command:

python examples/inference.py --model_name "meta-llama/Llama-2-7b-chat-hf" --peft_model "/home/paperspace/llama-recipes/src/llama_recipes/tr_models/checkpoint_4000/" --prompt_file "src/llama_recipes/prompt.txt"


"""
    training_config = train_config()
    lora_config = generate_peft_config(train_config, kwargs)

kwargs = {"use_peft": True,
          "peft_method": "lora",
          "quantization": True,
          "dataset": "biblechat_dataset",
          "model_name": "meta-llama/Llama-2-7b-chat-hf",
          "output_dir": "/home/paperspace/tr_models"}

from gloohack.modelling.llama import get_completion
from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
print("Loading model")
base_model = LlamaForCausalLM.from_pretrained(kwargs['model_name'])
peft_path = "/home/paperspace/llama-recipes/src/llama_recipes/tr_models/checkpoint_6000/"
peft_model = PeftModel.from_pretrained(base_model, peft_path, offload_folder="tmp") #offload to not run out of RAM
model = peft_model.to(0)
model = model.merge_and_unload() #converts back to not-peft-model
print("Loading tokenizer")
tokenizer = LlamaTokenizer.from_pretrained(kwargs['model_name'])
tokenizer.pad_token = tokenizer.eos_token
s = "Does Jesus like me?"
x = get_completion(s, model, tokenizer)
print(x)



