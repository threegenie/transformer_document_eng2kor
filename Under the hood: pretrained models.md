## Quick Tour : Under the hood

[🔗 Huggingface Transformers Docs >> Under the hood : pretrained models](https://huggingface.co/transformers/quicktour.html#under-the-hood-pretrained-models)

이제 파이프라인을 사용할 때 그 안에서 어떤 일이 일어나는지 알아보겠습니다.

아래 코드를 보면, 모델과 토크나이저는 *from_pretrained* 메서드를 통해 만들어집니다.

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

### 토크나이저 사용하기

토크나이저는 텍스트의 전처리를 담당합니다. 먼저, 주어진 텍스트를 토큰(token) (또는 단어의 일부, 구두점 기호 등)으로 분리합니다. 이 과정을 처리할 수 있는 다양한 규칙들이 있으므로([토크나이저 요약](https://huggingface.co/transformers/tokenizer_summary.html)에서 더 자세히 알아볼 수 있음), 모델명을 사용하여 토크나이저를 인스턴스화해야만 프리트레인 모델과 동일한 규칙을 사용할 수 있습니다.

두번째 단계는, 토큰(token)을 숫자 형태로 변환하여 텐서(tensor)를 구축하고 모델에 적용할 수 있도록 하는 것입니다. 이를 위해, 토크나이저에는 *from_pretrained* 메서드로 토크나이저를 인스턴스화할 때 다운로드하는 *vocab*이라는 것이 있습니다. 모델이 사전학습 되었을 때와 동일한 vocab을 사용해야 하기 때문입니다.

주어진 텍스트에 이 과정들을 적용하려면 토크나이저에 아래와 같이 텍스트를 넣으면 됩니다.

```python
inputs = tokenizer("We are very happy to show you the 🤗 Transformers library.")
```

이렇게 하면, 딕셔너리 형태의 문자열이 정수 리스트로 변환됩니다. 이 리스트는 [토큰 ID](https://huggingface.co/transformers/glossary.html#input-ids)(ids of the tokens)를 포함하고 있고, 모델에 필요한 추가 인수 또한 가지고 있습니다. 예를 들면, 모델이 시퀀스를 더 잘 이해하기 위해 사용하는 [어텐션 마스크](https://huggingface.co/transformers/glossary.html#attention-mask)(attention mask)도 포함하고 있습니다.

```python
print(inputs)

"""
{'input_ids': [101, 2057, 2024, 2200, 3407, 2000, 2265, 2017, 1996, 100, 19081, 3075, 1012, 102],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
"""
```

토크나이저에 문장 리스트를 직접 전달할 수 있습니다. 배치(batch)로 모델에 전달하는 것이 목표라면, 동일한 길이로 패딩하고 모델이 허용할 수 있는 최대 길이로 잘라 텐서를 반환하는 것이 좋습니다. 토크나이저에 이러한 사항들을 모두 지정할 수 있습니다. 

```python
# Pytorch
pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
```

```python
# Tensorflow
tf_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="tf"
)
```

모델이 예측하는 위치에(이 같은 경우엔 오른쪽) 프리트레이닝된 패딩 토큰을 이용하여 패딩이 자동으로 적용됩니다. 어텐션 마스크도 패딩을 고려하여 조정됩니다.

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

토크나이저에 대해 [이곳](https://huggingface.co/transformers/preprocessing.html)에서 더 자세히 알아볼 수 있습니다.

