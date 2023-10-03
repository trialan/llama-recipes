from llama_recipes.datasets import get_biblechat_dataset, get_samsum_dataset
from llama_recipes.configs.datasets import samsun_dataset, biblechat_dataset
from gloohack.modelling.config import get_tokenizer


if __name__ == '__main__':
    tokenizer = get_tokenizer()
    bc = get_biblechat_dataset(biblechat_dataset(), tokenizer, "test")
    ap = get_samsun_dataset(samsun_dataset(), tokenizer, "validation")


