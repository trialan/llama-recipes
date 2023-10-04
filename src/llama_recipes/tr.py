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


def load_ckpt(base, path):
    """ Offload to avoid RAM OOM, merge_and_unload for normal model fmt """
    peft_model = PeftModel.from_pretrained(base,
                                           path,
                                           offload_folder="tmp")
    model = peft_model.to(0)
    model = model.merge_and_unload()
    return model



if __name__ == '__main__':
    print("Loading model")
    base_model = LlamaForCausalLM.from_pretrained(kwargs['model_name'])

    print("Loading tokenizer")
    tokenizer = LlamaTokenizer.from_pretrained(kwargs['model_name'])
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading PEFT model")
    peft_paths = glob.glob("/home/paperspace/tr_models/*checkpoint*")
    paths = sorted(peft_paths)

    print("Testing")
    for p in paths:
        print(f"CKPT: {p}")
        model = load_ckpt(base_model, p)
        s = "Does Jesus like me?"
        x = get_completion(s, model, tokenizer)
        print(x)


