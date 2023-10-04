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


if __name__ == '__main__':
    training_config = training_config()
    training_config.use_peft = True
    training_config.peft_method = "lora"
    training_config.quantization = True
    training_config.dataset = "biblechat_dataset"
    training_config.model_name = "meta-llama/Llama-2-7b-chat-hf"
    training_config.output_dir = "/home/paperspace/tr_models"

    lora_config = generate_peft_config(train_config)
    1/0
    model = AutoModelForCausalLM.from_pretrained(
        "path_to_your_checkpoint",
        lora_config=lora_config)
