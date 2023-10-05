import torch
import glob
from peft import PeftModel
from transformers import AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM

from gloohack.modelling.llama import get_completion


MODEL = "meta-llama/Llama-2-7b-chat-hf"


def load_ckpt(base, path):
    """ Offload to avoid RAM OOM, merge_and_unload for normal model fmt """
    peft_model = PeftModel.from_pretrained(base,
                                           path,
                                           offload_folder="tmp")
    model = peft_model.to(0)
    model = model.merge_and_unload()
    return model


def format_llm_for_inference(user_messages, model_answers):
    assert len(user_messages) - 1 == len(model_answers)
    formatted_strings = []
    for i, user_msg in enumerate(user_messages):
        # If there's a corresponding model answer, append it
        model_ans = model_answers[i] if i < len(model_answers) else ""
        formatted_strings.append(f"<s>[INST] {user_msg} [/INST] {model_ans} </s>")
    return "".join(formatted_strings)


if __name__ == '__main__':
    print("Loading model")
    base_model = LlamaForCausalLM.from_pretrained(MODEL)

    print("Loading tokenizer")
    tokenizer = LlamaTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading PEFT model")
    peft_paths = glob.glob("/home/paperspace/llama-recipes/src/llama_recipes/run_4x_ckpts/*checkpoint*")
    paths = sorted(peft_paths)

    1/0

    model = load_ckpt(base_model, paths[1])
    s = format_for_llm_inference(user_messages=["Does Jesus like me?"],
                                 model_answers=[])
    x = get_completion(s, model, tokenizer)
    print(x)
    del model
    torch.cuda.empty_cache()


