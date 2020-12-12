from typing import Tuple

from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer

if __name__ == "__main__":
    model = BertModel.from_pretrained("bert-base-cased", output_attentions=True, output_values=True)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    inputs = tokenizer("Hello", return_tensors="pt")
    outputs = model(**inputs)