from llama_recipes.datasets import get_biblechat_dataset, get_alpaca_dataset
from llama_recipes.configs.datasets import alpaca_dataset, biblechat_dataset
from gloohack.modelling.config import get_tokenizer

if __name__ == '__main__':
    tokenizer = get_tokenizer()
    bc = get_biblechat_dataset(biblechat_dataset(), tokenizer, "test")
    ap = get_alpaca_dataset(alpaca_dataset(), tokenizer, "val")


