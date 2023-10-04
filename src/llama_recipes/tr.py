"""
Reproduce train config from run command:

python -m llama_recipes.finetuning --dataset "biblechat_dataset" --custom_dataset.file "examples/biblechat_dataset.py" --use_peft --peft_method lora --quantization --model_name "meta-llama/Llama-2-7b-chat-hf" --output_dir "/home/paperspace/tr_models"

"""

kwargs = {"use_peft": True,
          "peft_method": "lora",
          "quantization": True,
          "dataset": "biblechat_dataset",
          "model_name": "meta-llama/Llama-2-7b-chat-hf",
          "output_dir": "/home/paperspace/tr_models"}

from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
from gloohack.modelling.llama import get_completion



if __name__ == '__main__':
    print("Loading model")
    base_model = LlamaForCausalLM.from_pretrained(kwargs['model_name'])

    print("Loading PEFT model")
    peft_path = "/home/paperspace/llama-recipes/src/llama_recipes/tr_models/checkpoint_6000/"
    peft_model = PeftModel.from_pretrained(base_model, peft_path, offload_folder="tmp") #offload to not run out of RAM
    model = peft_model.to(0)
    model = model.merge_and_unload() #converts back to not-peft-model

    print("Loading tokenizer")
    tokenizer = LlamaTokenizer.from_pretrained(kwargs['model_name'])
    tokenizer.pad_token = tokenizer.eos_token

    print("Testing")
    s = "Does Jesus like me?"
    x = get_completion(s, model, tokenizer)
    print(x)


