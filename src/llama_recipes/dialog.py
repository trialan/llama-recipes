from peft import PeftModel
import glob
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

from gloohack.modelling.llama import get_completion


MODEL = "meta-llama/Llama-2-7b-chat-hf"


class DialogChat:
    def __init__(self, model, tokenizer):
        self.conversation = ""
        self.model = model
        self.tokenizer = tokenizer

    def respond(self, user_message):
        self.update_conversation(user_message)
        model_output = self.get_model_output()
        self.conversation = model_output
        model_response = self.extract_answer_from(model_output)
        return model_response

    def get_model_output(self):
        out = get_completion(self.conversation,
                             self.model,
                             self.tokenizer)
        return out

    def update_conversation(self, user_message):
        self.conversation += f"[INST] {user_message} [/INST]"

    def extract_answer_from(self, output):
        answer = output.split("[/INST]")[-1]
        return answer


def load_ckpt(base, path):
    """ Offload to avoid RAM OOM, merge_and_unload for normal model fmt """
    peft_model = PeftModel.from_pretrained(base,
                                           path,
                                           offload_folder="tmp")
    model = peft_model.to(0)
    model = model.merge_and_unload()
    return model


def delete_model(model):
    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    print("Loading model")
    base_model = LlamaForCausalLM.from_pretrained(MODEL)

    print("Loading tokenizer")
    tokenizer = LlamaTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading PEFT model")
    peft_paths = glob.glob("/home/paperspace/tr_models/*checkpoint*")
    paths = sorted(sorted(peft_paths,
                          key=lambda x: int(x.split('checkpoint_')[-1])),
                          key=lambda y: int(y.split('epoch_')[1][:1]))

    model = load_ckpt(base_model, paths[1])

    chat = DialogChat(model, tokenizer)
    user_messages = ["Does Jesus like me?", "Can you explain more?", "Thanks"]
    for m in user_messages:
        print(f"Question: {m}")
        print(f"Answer: {chat.respond(m)}")



