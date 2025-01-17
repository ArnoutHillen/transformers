---
language: ja
---
## Japanese ELECTRA-small

We provide a Japanese **ELECTRA-Small** model, as described in [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://openreview.net/pdf?id=r1xMH1BtvB).

Our pretraining process employs subword units derived from the [Japanese Wikipedia](https://dumps.wikimedia.org/jawiki/latest), using the [Byte-Pair Encoding](https://www.aclweb.org/anthology/P16-1162.pdf) method and building on an initial tokenization with [mecab-ipadic-NEologd](https://github.com/neologd/mecab-ipadic-neologd). For optimal performance, please take care to set your MeCab dictionary appropriately.

```
# ELECTRA-small generator usage

from transformers import BertJapaneseTokenizer, ElectraForMaskedLM

tokenizer = BertJapaneseTokenizer.from_pretrained('Cinnamon/electra-small-japanese-generator', mecab_kwargs={"mecab_option": "-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd"})

model = ElectraForMaskedLM.from_pretrained('Cinnamon/electra-small-japanese-generator')
```
