from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer
import torch

# # BERT
# if __name__ == "__main__":
#     model = BertModel.from_pretrained("bert-base-cased", output_attentions=True, output_values=True)
#     tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
#     inputs = tokenizer("Hello", return_tensors="pt")
#     outputs = model(**inputs)
#     print(len(outputs["values"]))
#     print(outputs["values"][0].shape)
#     values = outputs["values"]
#     values = torch.stack(values).squeeze()
#     values = values.detach().numpy()
#     print(values.shape)

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer

# # GPT-2
# if __name__ == "__main__":
#     model = GPT2Model.from_pretrained("gpt2", output_attentions=True, output_values=True)
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     inputs = tokenizer("Hello, I'm Arnout Hillen", return_tensors="pt")
#     outputs = model(**inputs)
#     print(len(outputs["values"]))
#     print(outputs["values"][0].shape)
#     values = outputs["values"]
#     values = torch.stack(values).squeeze()
#     values = values.detach().numpy()
#     print(values.shape)

from transformers.models.xlnet.modeling_xlnet import XLNetModel
from transformers.models.xlnet.tokenization_xlnet import XLNetTokenizer

# XLNet
if __name__ == "__main__":
    model = XLNetModel.from_pretrained("xlnet-base-cased", output_attentions=True, output_values=True)
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    inputs = tokenizer("Hello, I'm", return_tensors="pt")
    outputs = model(**inputs)
    print(len(outputs["values"]))
    print(outputs["values"][0].shape)
    values = outputs["values"]
    values = torch.stack(values).squeeze()
    values = values.detach().numpy()
    print(values.shape)


# from transformers.models.electra.modeling_electra import ElectraModel
# from transformers.models.electra.tokenization_electra import ElectraTokenizer
#
# # ELECTRA
# if __name__ == "__main__":
#     model = ElectraModel.from_pretrained("google/electra-base-discriminator", output_attentions=True, output_values=True)
#     tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")
#     inputs = tokenizer("Hello, I'm Arnout Hillen", return_tensors="pt")
#     outputs = model(**inputs)
#     print(len(outputs["values"]))
#     print(outputs["values"][0].shape)
#     values = outputs["values"]
#     values = torch.stack(values).squeeze()
#     values = values.detach().numpy()
#     print(values.shape)