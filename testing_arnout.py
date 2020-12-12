from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer
import torch

if __name__ == "__main__":
    model = BertModel.from_pretrained("bert-base-cased", output_attentions=True, output_values=True)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    inputs = tokenizer("Hello", return_tensors="pt")
    outputs = model(**inputs)
    print(len(outputs["values"]))
    print(outputs["values"][0].shape)
    values = outputs["values"]
    values = torch.stack(values).squeeze()
    values = values.detach().numpy()
    print(values.shape)
