## Quick Tour : Under the hood

[π Huggingface Transformers Docs >> Under the hood : pretrained models](https://huggingface.co/transformers/quicktour.html#under-the-hood-pretrained-models)

μ΄μ  νμ΄νλΌμΈμ μ¬μ©ν  λ κ·Έ μμμ μ΄λ€ μΌμ΄ μΌμ΄λλμ§ μμλ³΄κ² μ΅λλ€.

μλ μ½λλ₯Ό λ³΄λ©΄, λͺ¨λΈκ³Ό ν ν¬λμ΄μ λ *from_pretrained* λ©μλλ₯Ό ν΅ν΄ λ§λ€μ΄μ§λλ€.

```python
# Pytorch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

```python
# Tensorflow
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tf_model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

### ν ν¬λμ΄μ  μ¬μ©νκΈ°

ν ν¬λμ΄μ λ νμ€νΈμ μ μ²λ¦¬λ₯Ό λ΄λΉν©λλ€. λ¨Όμ , μ£Όμ΄μ§ νμ€νΈλ₯Ό ν ν°(token) (λλ λ¨μ΄μ μΌλΆ, κ΅¬λμ  κΈ°νΈ λ±)μΌλ‘ λΆλ¦¬ν©λλ€. μ΄ κ³Όμ μ μ²λ¦¬ν  μ μλ λ€μν κ·μΉλ€μ΄ μμΌλ―λ‘([ν ν¬λμ΄μ  μμ½](https://huggingface.co/transformers/tokenizer_summary.html)μμ λ μμΈν μμλ³Ό μ μμ), λͺ¨λΈλͺμ μ¬μ©νμ¬ ν ν¬λμ΄μ λ₯Ό μΈμ€ν΄μ€νν΄μΌλ§ νλ¦¬νΈλ μΈ λͺ¨λΈκ³Ό λμΌν κ·μΉμ μ¬μ©ν  μ μμ΅λλ€.

λλ²μ§Έ λ¨κ³λ, ν ν°(token)μ μ«μ ννλ‘ λ³ννμ¬ νμ(tensor)λ₯Ό κ΅¬μΆνκ³  λͺ¨λΈμ μ μ©ν  μ μλλ‘ νλ κ²μλλ€. μ΄λ₯Ό μν΄, ν ν¬λμ΄μ μλ *from_pretrained* λ©μλλ‘ ν ν¬λμ΄μ λ₯Ό μΈμ€ν΄μ€νν  λ λ€μ΄λ‘λνλ *vocab*μ΄λΌλ κ²μ΄ μμ΅λλ€. λͺ¨λΈμ΄ μ¬μ νμ΅ λμμ λμ λμΌν vocabμ μ¬μ©ν΄μΌ νκΈ° λλ¬Έμλλ€.

μ£Όμ΄μ§ νμ€νΈμ μ΄ κ³Όμ λ€μ μ μ©νλ €λ©΄ ν ν¬λμ΄μ μ μλμ κ°μ΄ νμ€νΈλ₯Ό λ£μΌλ©΄ λ©λλ€.

```python
inputs = tokenizer("We are very happy to show you the π€ Transformers library.")
```

μ΄λ κ² νλ©΄, λμλλ¦¬ ννμ λ¬Έμμ΄μ΄ μ μ λ¦¬μ€νΈλ‘ λ³νλ©λλ€. μ΄ λ¦¬μ€νΈλ [ν ν° ID](https://huggingface.co/transformers/glossary.html#input-ids)(ids of the tokens)λ₯Ό ν¬ν¨νκ³  μκ³ , λͺ¨λΈμ νμν μΆκ° μΈμ λν κ°μ§κ³  μμ΅λλ€. μλ₯Ό λ€λ©΄, λͺ¨λΈμ΄ μνμ€λ₯Ό λ μ μ΄ν΄νκΈ° μν΄ μ¬μ©νλ [μ΄νμ λ§μ€ν¬](https://huggingface.co/transformers/glossary.html#attention-mask)(attention mask)λ ν¬ν¨νκ³  μμ΅λλ€.

```python
print(inputs)

"""
{'input_ids': [101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 100, 19081, 3075, 1012, 102],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
"""
```

ν ν¬λμ΄μ μ λ¬Έμ₯ λ¦¬μ€νΈλ₯Ό μ§μ  μ λ¬ν  μ μμ΅λλ€. λ°°μΉ(batch)λ‘ λͺ¨λΈμ μ λ¬νλ κ²μ΄ λͺ©νλΌλ©΄, λμΌν κΈΈμ΄λ‘ ν¨λ©νκ³  λͺ¨λΈμ΄ νμ©ν  μ μλ μ΅λ κΈΈμ΄λ‘ μλΌ νμλ₯Ό λ°ννλ κ²μ΄ μ’μ΅λλ€. ν ν¬λμ΄μ μ μ΄λ¬ν μ¬ν­λ€μ λͺ¨λ μ§μ ν  μ μμ΅λλ€. 

```python
# Pytorch
pt_batch = tokenizer(
    ["We are very happy to show you the π€ Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
```

```python
# Tensorflow
tf_batch = tokenizer(
    ["We are very happy to show you the π€ Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="tf"
)
```

λͺ¨λΈμ΄ μμΈ‘νλ μμΉμ(μ΄ κ°μ κ²½μ°μ μ€λ₯Έμͺ½) νλ¦¬νΈλ μ΄λλ ν¨λ© ν ν°μ μ΄μ©νμ¬ ν¨λ©μ΄ μλμΌλ‘ μ μ©λ©λλ€. μ΄νμ λ§μ€ν¬λ ν¨λ©μ κ³ λ €νμ¬ μ‘°μ λ©λλ€.

```python
# Pytorch
for key, value in pt_batch.items():
    print(f"{key}: {value.numpy().tolist()}"

"""
input_ids: [[101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 100, 19081, 3075, 1012, 102], [101, 2057, 3246, 2017, 2123, 1005, 1056, 5223, 2009, 1012, 102, 0, 0, 0]]
attention_mask: [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]
"""
```

```python
# Tensorflow
for key, value in tf_batch.items():
    print(f"{key}: {value.numpy().tolist()}")

"""
input_ids: [[101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 100, 19081, 3075, 1012, 102], [101, 2057, 3246, 2017, 2123, 1005, 1056, 5223, 2009, 1012, 102, 0, 0, 0]]
attention_mask: [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]]
"""
```

ν ν¬λμ΄μ μ λν΄ [μ΄κ³³](https://huggingface.co/transformers/preprocessing.html)μμ λ μμΈν μμλ³Ό μ μμ΅λλ€.

### λͺ¨λΈ μ¬μ©νκΈ°

μΈν λ°μ΄ν°κ° ν ν¬λμ΄μ λ₯Ό ν΅ν΄ μ μ²λ¦¬λλ©΄, λͺ¨λΈλ‘ μ§μ  λ³΄λΌ μ μμ΅λλ€. μμ μΈκΈν κ²μ²λΌ, λͺ¨λΈμ νμν λͺ¨λ  κ΄λ ¨ μ λ³΄κ° ν¬ν¨λ©λλ€. λ§μ½ νμνλ‘μ° λͺ¨λΈμ μ¬μ©νλ€λ©΄ λμλλ¦¬μ ν€λ₯Ό μ§μ  νμλ‘ μ λ¬ν  μ μκ³ , νμ΄ν μΉ λͺ¨λΈμ μ¬μ©νλ€λ©΄ '**'μ λν΄μ λμλλ¦¬λ₯Ό νμ΄ μ€μΌ ν©λλ€.

```python
# Pytorch
pt_outputs = pt_model(**pt_batch)
```

```python
# Tensorflow
tf_outputs = tf_model(tf_batch)
```

νκΉνμ΄μ€ νΈλμ€ν¬λ¨Έμμ λͺ¨λ  μμνμ λ€λ₯Έ λ©νλ°μ΄ν°μ ν¨κ» λͺ¨λΈμ μ΅μ’ νμ±ν μνκ° ν¬ν¨λ κ°μ²΄μλλ€. μ΄λ¬ν κ°μ²΄λ μ¬κΈ°μ λ μμΈν μ€λͺλμ΄ μμ΅λλ€. μΆλ ₯κ°μ μ΄ν΄λ³΄κ² μ΅λλ€.

```python
# Pytorch
print(pt_outputs)

"""
SequenceClassifierOutput(loss=None, logits=tensor([[-4.0833,  4.3364],
       [ 0.0818, -0.0418]], grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)
"""
```

```python
# Tensorflow
print(tf_outputs)
"""
TFSequenceClassifierOutput(loss=None, logits=<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[-4.0833 ,  4.3364  ],
       [ 0.0818, -0.0418]], dtype=float32)>, hidden_states=None, attentions=None)
"""
```

μΆλ ₯λ κ°μ μλ *logits* ν­λͺ©μ μ£Όλͺ©νμ­μμ€. μ΄ ν­λͺ©μ μ¬μ©νμ¬ λͺ¨λΈμ μ΅μ’ νμ±ν μνμ μ κ·Όν  μ μμ΅λλ€.

μ£Όμ
λͺ¨λ  νκΉνμ΄μ€ νΈλμ€ν¬λ¨Έ λͺ¨λΈ(νμ΄ν μΉ λλ νμνλ‘μ°)μ λ§μ§λ§ νμ±ν ν¨μκ° μ’μ’ μμ€(loss)κ³Ό λν΄μ§κΈ° λλ¬Έμ λ§μ§λ§ νμ±ν ν¨μ(μννΈλ§₯μ€ κ°μ)λ₯Ό μ μ©νκΈ° μ΄μ μ λͺ¨λΈ νμ±ν μνλ₯Ό λ¦¬ν΄ν©λλ€. 

μμΈ‘μ μν΄ μννΈλ§₯μ€ νμ±νλ₯Ό μ μ©ν΄ λ΄μλ€.

```python
# Pytorch
from torch import nn
pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
```

```python
# Tensorflow
import tensorflow as tf
tf_predictions = tf.nn.softmax(tf_outputs.logits, axis=-1)
```

μ΄μ  κ³Όμ μμ μ»μ΄μ§ μ«μλ€μ λ³Ό μ μμ΅λλ€.

```python
# Pytorch
print(pt_predictions)
"""
tensor([[2.2043e-04, 9.9978e-01],
        [5.3086e-01, 4.6914e-01]], grad_fn=<SoftmaxBackward>)
"""
```

```python
# Tensorflow
print(tf_predictions)
"""
tf.Tensor(
[[2.2043e-04 9.9978e-01]
 [5.3086e-01 4.6914e-01]], shape=(2, 2), dtype=float32)
"""
```

λͺ¨λΈμ μΈν λ°μ΄ν° μΈμ λΌλ²¨μ λ£λ κ²½μ°μλ, λͺ¨λΈ μΆλ ₯ κ°μ²΄μ λ€μκ³Ό κ°μ μμ€(loss) μμ±λ ν¬ν¨λ©λλ€.

```python
# Pytorch
import torch
pt_outputs = pt_model(**pt_batch, labels = torch.tensor([1, 0]))
print(pt_outputs)
"""
SequenceClassifierOutput(loss=tensor(0.3167, grad_fn=<NllLossBackward>), logits=tensor([[-4.0833,  4.3364],
        [ 0.0818, -0.0418]], grad_fn=<AddmmBackward>), hidden_states=None, attentions=None)
"""
```

```python
# Tensorflow
import tensorflow as tf
tf_outputs = tf_model(tf_batch, labels = tf.constant([1, 0]))
print(tf_outputs)
"""
TFSequenceClassifierOutput(loss=<tf.Tensor: shape=(2,), dtype=float32, numpy=array([2.2051e-04, 6.3326e-01], dtype=float32)>, logits=<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[-4.0833 ,  4.3364  ],
       [ 0.0818, -0.0418]], dtype=float32)>, hidden_states=None, attentions=None)
"""
```

λͺ¨λΈμ νμ€ [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)μ΄λ [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)λ‘ νΈλ μ΄λ λ£¨νμμ μ¬μ©ν  μ μμ΅λλ€. νκΉνμ΄μ€ νΈλμ€ν¬λ¨Έλ Trainer(νμνλ‘μ°μμλ TFTrainer) ν΄λμ€λ₯Ό μ κ³΅νμ¬ μ¬λ¬λΆμ΄ λͺ¨λΈμ νμ΅μν€λ κ²μ λμ΅λλ€(λΆμ° νΈλ μ΄λ, νΌν© μ λ°λ λ±κ³Ό κ°μ κ³Όμ μμλ μ£Όμν΄μΌ ν©λλ€). μμΈν λ΄μ©μ [νΈλ μ΄λ νν λ¦¬μΌ](https://huggingface.co/transformers/training.html)μ μ°Έμ‘°νμ­μμ€.

μ£Όμ
Pytorch λͺ¨λΈ μΆλ ₯μ IDEμ μμ±μ λν μλ μμ±μ κ°μ Έμ¬ μ μλ νΉμ λ°μ΄ν° ν΄λμ€μλλ€. λν νν λλ λμλλ¦¬μ²λΌ μλν©λλ€(μ μ, μ¬λΌμ΄μ€ λλ λ¬Έμμ΄λ‘ μΈλ±μ±ν  μ μμ). μ΄ κ²½μ° μ€μ λμ§ μμ μμ±(None κ°μ κ°μ§κ³  μλ)μ λ¬΄μλ©λλ€.

λͺ¨λΈμ νμΈνλμ΄ λλλ©΄, μλμ κ°μ λ°©λ²μΌλ‘ ν ν¬λμ΄μ μ ν¨κ» μ μ₯ν  μ μμ΅λλ€.

```python
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
```

κ·Έλ° λ€μ λͺ¨λΈ μ΄λ¦ λμ  λλ ν λ¦¬ μ΄λ¦μ μ λ¬νμ¬ from_pretrained() λ©μλλ₯Ό μ¬μ©νμ¬ μ΄ λͺ¨λΈμ λ€μ λ‘λν  μ μμ΅λλ€. νκΉνμ΄μ€ νΈλμ€ν¬λ¨Έμ μ λ©μ§ κΈ°λ₯ μ€ νλλ νμ΄ν μΉμ νμνλ‘μ° κ°μ μ½κ² μ νν  μ μλ€λ κ²μλλ€. μ΄μ κ³Ό κ°μ΄ μ μ₯λ λͺ¨λΈμ νμ΄ν μΉ λλ νμνλ‘μ°μμ λ€μ λ‘λν  μ μμ΅λλ€. μ μ₯λ νμ΄ν μΉ λͺ¨λΈμ νμνλ‘μ° λͺ¨λΈμ λ‘λνλ κ²½μ° from_pretrained()λ₯Ό λ€μκ³Ό κ°μ΄ μ¬μ©ν©λλ€.

```python
# Pytorch -> Tensorflow
from transformers import TFAutoModel
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = TFAutoModel.from_pretrained(save_directory, from_pt=True)
```

μ μ₯λ νμνλ‘μ° λͺ¨λΈμ νμ΄ν μΉ λͺ¨λΈμ λ‘λνλ κ²½μ° λ€μ μ½λλ₯Ό μ¬μ©ν΄μΌ ν©λλ€.

```python
# Tensorflow -> Pytorch
from transformers import AutoModel
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModel.from_pretrained(save_directory, from_tf=True)
```

λ§μ§λ§μΌλ‘, λͺ¨λΈμ λͺ¨λ  μλ μν(hidden state)μ λͺ¨λ  μ΄νμ κ°μ€μΉ(attention weight)λ₯Ό λ¦¬ν΄νλλ‘ μ€μ ν  μ μμ΅λλ€.

```python
# Pytorch
pt_outputs = pt_model(**pt_batch, output_hidden_states=True, output_attentions=True)
all_hidden_states  = pt_outputs.hidden_states
all_attentions = pt_outputs.attentions
```

```python
# Tensorflow
tf_outputs = tf_model(tf_batch, output_hidden_states=True, output_attentions=True)
all_hidden_states =  tf_outputs.hidden_states
all_attentions = tf_outputs.attentions
```

### μ½λμ μμΈμ€νκΈ°

*AutoModel* λ° *AutoTokenizer* ν΄λμ€λ μ¬μ  κ΅μ‘λ λͺ¨λΈλ‘ μλμΌλ‘ μ΄λν  μ μλ λ°λ‘κ°κΈ°μΌ λΏμλλ€. μ΄λ©΄μλ λΌμ΄λΈλ¬λ¦¬κ° μν€νμ²μ ν΄λμ€μ μ‘°ν©λΉ νλμ λͺ¨λΈ ν΄λμ€λ₯Ό κ°μ§κ³  μμΌλ―λ‘ νμμ λ°λΌ μ½λλ₯Ό μ½κ² μ‘μΈμ€νκ³  μ‘°μ ν  μ μμ΅λλ€.

μ΄μ  μμμμ, μ΄ λͺ¨λΈμ '*distilbert-base-cased-un-finetuned-sst-2-english*'λΌκ³  λΆλ Έλλ°, μ΄λ *[DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html)* κ΅¬μ‘°λ₯Ό μ¬μ©νλ€λ λ»μλλ€. *AutoModelForSequenceClassification*(λλ νμνλ‘μ°μμλ *TFAutoModelForSequenceClassification*)μ΄ μ¬μ©λμμΌλ―λ‘ μλμΌλ‘ μμ±λ λͺ¨λΈμ *DistilBertForSequenceClassification*μ΄ λ©λλ€. ν΄λΉ λͺ¨λΈμ μ€λͺμμμ ν΄λΉ λͺ¨λΈκ³Ό κ΄λ ¨λ λͺ¨λ  μΈλΆ μ λ³΄λ₯Ό νμΈνκ±°λ μμ€ μ½λλ₯Ό μ°Ύμλ³Ό μ μμ΅λλ€. λͺ¨λΈ λ° ν ν¬λμ΄μ λ₯Ό μ§μ  μΈμ€ν΄μ€νν  μ μλ λ°©λ²μ λ€μκ³Ό κ°μ΅λλ€.

```python
# Pytorch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = DistilBertForSequenceClassification.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
```

```python
# Tensorflow
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = TFDistilBertForSequenceClassification.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
```

### λͺ¨λΈ μ»€μ€ν°λ§μ΄μ§ νκΈ°

λͺ¨λΈ μμ²΄μ λΉλ λ°©λ²μ λ³κ²½νλ €λ©΄ μ¬μ©μ μ μ κ΅¬μ± ν΄λμ€λ₯Ό μ μν  μ μμ΅λλ€. κ° μν€νμ²μλ κ³ μ ν κ΄λ ¨ κ΅¬μ±(Configuration)μ΄ μ κ³΅λ©λλ€. μλ₯Ό λ€μ΄, [*DistilBertConfig*](https://huggingface.co/transformers/model_doc/distilbert.html#transformers.DistilBertConfig)λ₯Ό μ¬μ©νλ©΄ *DistilBERT*μ λν μλ μ°¨μ(hidden dimension), λλ‘­μμ λΉμ¨(dropout rate) λ±μ λ§€κ°λ³μ(parameter)λ₯Ό μ§μ ν  μ μμ΅λλ€. μλ μ°¨μμ ν¬κΈ°λ₯Ό λ³κ²½νλ κ²κ³Ό κ°μ΄ μ€μν μμ  μμμ νλ©΄ μ¬μ  νλ ¨λ λͺ¨λΈμ λ μ΄μ μ¬μ©ν  μ μκ³  μ²μλΆν° νμ΅μμΌμΌ ν©λλ€. κ·Έλ° λ€μ Configμμ μ§μ  λͺ¨λΈμ μΈμ€ν΄μ€νν©λλ€.

μλμμλ from_pretrained() λ©μλλ₯Ό μ¬μ©νμ¬ ν ν¬λμ΄μ μ μ¬μ  μ μλ μ΄νλ₯Ό λ‘λν©λλ€. κ·Έλ¬λ ν ν¬λμ΄μ μ λ¬λ¦¬ μ°λ¦¬λ μ²μλΆν° λͺ¨λΈμ μ΄κΈ°ννκ³ μ ν©λλ€. λ°λΌμ from_pretrained() λ°©λ²μ μ¬μ©νλ λμ  Configμμ λͺ¨λΈμ μΈμ€ν΄μ€νν©λλ€.

```python
# Pytorch
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
config = DistilBertConfig(n_heads=8, dim=512, hidden_dim=4*512)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification(config)
```

```python
# Tensorflow
from transformers import DistilBertConfig, DistilBertTokenizer, TFDistilBertForSequenceClassification
config = DistilBertConfig(n_heads=8, dim=512, hidden_dim=4*512)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertForSequenceClassification(config)
```

λͺ¨λΈ ν€λλ§ λ³κ²½νλ κ²½μ°(λΌλ²¨ μμ κ°μ)μλ μ¬μ  νλ ¨λ λͺ¨λΈμ μ¬μ©ν  μ μμ΅λλ€. μλ₯Ό λ€μ΄, μ¬μ  νλ ¨λ λͺ¨λΈμ μ¬μ©νμ¬ 10κ°μ μλ‘ λ€λ₯Έ λΌλ²¨μ λν λΆλ₯κΈ°(Classifier)λ₯Ό μ μν΄ λ³΄κ² μ΅λλ€. λΌλ²¨ μλ₯Ό λ³κ²½νκΈ° μν΄ λͺ¨λ  κΈ°λ³Έκ°μ μ¬μ©νμ¬ μ Configλ₯Ό μμ±νλ λμ μ Configκ° from_pretrained() λ©μλμ μΈμλ₯Ό μ λ¬νλ©΄ κΈ°λ³Έ Configκ° μ μ ν μλ°μ΄νΈλ©λλ€.

```python
# Pytorch
from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertForSequenceClassification
model_name = "distilbert-base-uncased"
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=10)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
```

```python
# Tensorflow
from transformers import DistilBertConfig, DistilBertTokenizer, TFDistilBertForSequenceClassification
model_name = "distilbert-base-uncased"
model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=10)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
```
