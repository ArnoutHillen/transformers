Changes are made to [BERT](https://github.com/ArnoutHillen/transformers/blob/master/src/transformers/models/bert/modeling_bert.py), [GPT-2](https://github.com/ArnoutHillen/transformers/blob/master/src/transformers/models/gpt2/modeling_gpt2.py), [XLNet](https://github.com/ArnoutHillen/transformers/blob/master/src/transformers/models/xlnet/modeling_xlnet.py) and [ELECTRA](https://github.com/ArnoutHillen/transformers/blob/master/src/transformers/models/electra/modeling_electra.py).

## Steps to extract information from the models.

- Install via pip.
```bash
pip install git+https://github.com/ArnoutHillen/transformers
```

- Create a tokenizer and a model, and use the following flags to add additional information to the model output.
  - Attention weights: output_attentions=True
  - Query, key and value vectors: output_q_values=True, output_k_values=True and output_v_values=True
  - Linear transformation (W_o): output_dense=True
  - Multilayer perceptron activations (for BERT): output_mlp_activations=True
```python
info = {"output_attentions":True,"ouput_q_values":True,"output_k_values":True,"output_v_values":True,"output_dense":True,"output_mlp_activations":True} # output_mlp_activations, output_q_values and output_k_values currently only for BERT (attention weights were used directly for the other models, instead of the query and keys).
# BERT
tokenizer = BertTokenzier.from_pretrained("bert-base-cased")
model = BertModel.from_pretrained("bert-base-cased",**info)
# GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2",**info)
# XLNet
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
model = XLNetModel.from_pretrained("xlnet-base-cased",**info)
# ELECTRA
tokenizer = ElectraModel.from_pretrained("google/electra-base-discriminator")
model = ElectraTokenizer.from_pretrained("google/electra-base-discriminator",**info)
```

- Extract the information from the output
  - Attention weights: key is "attention" 
  - Query, key and value vectors: key is "q_activations", "k_activations", "v_activations"
  - Linear transformation: key is "dense"
```python
input = "An input sentence"
inputs = tokenizer(input, return_tensors="pt")
outputs = model(**inputs)
# Attention weights
attention = outputs["attention"]
# Query, key and value vectors.
q,k,v = outputs["q_activations"], outputs["k_activations"], outputs["v_activations"]
# Linear transformation matrix.
dense = outputs["dense"]
```
