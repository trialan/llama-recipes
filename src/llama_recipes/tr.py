import sys
sys.path = sorted(sys.path) #Hack: avoids import of wrong datasets
if "" in sys.path:
    sys.path.remove("")
if "." in sys.path:
    sys.path.remove(".")
sys.path.append(".")

from llama_recipes.utils.config_utils import generate_peft_config
from llama_recipes.configs.training import train_config

from transformers import AutoModelForCausalLM


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


if __name__ == '__main__':
    training_config = train_config()
    lora_config = generate_peft_config(train_config, kwargs)
    1/0
    model = AutoModelForCausalLM.from_pretrained(
        "/home/paperspace/tr_models/checkpoint_4000",
        lora_config=lora_config)


python examples/inference.py --model_name "meta-llama/Llama-2-7b-chat-hf" --peft_model "/home/paperspace/tr_models/checkpoint_4000/" --prompt_file "src/llama_recipes/prompt.txt"
