Changes are made to the [BERT model](), [GPT-2](), [XLNet]() and [ELECTRA]().

## Steps to extract information from the models.

- Install via pip.
```bash
pip install git+https://github.com/ArnoutHillen/transformers
```

- Create a tokenizer and a model, and use the following flags to add additional information to the model output.
  - Attention weights: output_attentions=True
  - Query, key and value vectors: output_q_values=True, output_k_values=True and output_v_values=True
  - Linear transformation (W_o): output_dense=True
```python
// BERT
tokenizer = BertTokenzier.from_pretrained("bert-base-cased")
model = BertModel.from_pretrained("bert-base-cased")
// GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
// XLNet
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
model = XLNetModel.from_pretrained("xlnet-base-cased")
// ELECTRA
tokenizer = ElectraModel.from_pretrained("google/electra-base-discriminator")
model = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")
```

- Extract the information from the output
  - Attention weights: key is "attention" 
  - Query, key and value vectors: key is "q_activations", "k_activations", "v_activations"
  - Linear transformation: key is "dense"
```python
input = <input sentence>
inputs = tokenizer(input, return_tensors="pt")
outputs = model(**inputs)
// Attention weights
attention = outputs["attention"]
// Query, key and value vectors.
q,k,v = outputs["q_activations"], outputs["k_activations"], outputs["v_activations"]
// Linear transformation matrix.
dense = outputs["dense"]
```
