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

### 모델 사용하기

인풋 데이터가 토크나이저를 통해 전처리되면, 모델로 직접 보낼 수 있습니다. 앞서 언급한 것처럼, 모델에 필요한 모든 관련 정보가 포함됩니다. 만약 텐서플로우 모델을 사용한다면 딕셔너리의 키를 직접 텐서로 전달할 수 있고, 파이토치 모델을 사용한다면 '**'을 더해서 딕셔너리를 풀어 줘야 합니다.

```python
# Pytorch
pt_outputs = pt_model(**pt_batch)
```

```python
# Tensorflow
tf_outputs = tf_model(tf_batch)
```

허깅페이스 트랜스포머에서 모든 아웃풋은 다른 메타데이터와 함께 모델의 최종 활성화 상태가 포함된 개체입니다. 이러한 개체는 여기에 더 자세히 설명되어 있습니다. 출력값을 살펴보겠습니다.

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

출력된 값에 있는 *logits* 항목에 주목하십시오. 이 항목을 사용하여 모델의 최종 활성화 상태에 접근할 수 있습니다.

주의
모든 허깅페이스 트랜스포머 모델(파이토치 또는 텐서플로우)은 마지막 활성화 함수가 종종 손실(loss)과 더해지기 때문에 마지막 활성화 함수(소프트맥스 같은)를 적용하기 이전의 모델 활성화 상태를 리턴합니다. 

예측을 위해 소프트맥스 활성화를 적용해 봅시다.

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

이전 과정에서 얻어진 숫자들을 볼 수 있습니다.

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

모델에 인풋 데이터 외에 라벨을 넣는 경우에는, 모델 출력 개체에 다음과 같은 손실(loss) 속성도 포함됩니다.

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

모델은 표준 [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)이나 [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)로 트레이닝 루프에서 사용할 수 있습니다. 허깅페이스 트랜스포머는 Trainer(텐서플로우에서는 TFTrainer) 클래스를 제공하여 여러분이 모델을 학습시키는 것을 돕습니다(분산 트레이닝, 혼합 정밀도 등과 같은 과정에서는 주의해야 합니다). 자세한 내용은 [트레이닝 튜토리얼](https://huggingface.co/transformers/training.html)을 참조하십시오.

주의
Pytorch 모델 출력은 IDE의 속성에 대한 자동 완성을 가져올 수 있는 특수 데이터 클래스입니다. 또한 튜플 또는 딕셔너리처럼 작동합니다(정수, 슬라이스 또는 문자열로 인덱싱할 수 있음). 이 경우 설정되지 않은 속성(None 값을 가지고 있는)은 무시됩니다.

모델의 파인튜닝이 끝나면, 아래와 같은 방법으로 토크나이저와 함께 저장할 수 있습니다.

```python
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
```

그런 다음 모델 이름 대신 디렉토리 이름을 전달하여 from_pretrained() 메서드를 사용하여 이 모델을 다시 로드할 수 있습니다. 허깅페이스 트랜스포머의 의 멋진 기능 중 하나는 파이토치와 텐서플로우 간에 쉽게 전환할 수 있다는 것입니다. 이전과 같이 저장된 모델은 파이토치 또는 텐서플로우에서 다시 로드할 수 있습니다. 저장된 파이토치 모델을 텐서플로우 모델에 로드하는 경우 from_pretrained()를 다음과 같이 사용합니다.

```python
# Pytorch -> Tensorflow
from transformers import TFAutoModel
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = TFAutoModel.from_pretrained(save_directory, from_pt=True)
```

저장된 텐서플로우 모델을 파이토치 모델에 로드하는 경우 다음 코드를 사용해야 합니다.

```python
# Tensorflow -> Pytorch
from transformers import AutoModel
tokenizer = AutoTokenizer.from_pretrained(save_directory)
model = AutoModel.from_pretrained(save_directory, from_tf=True)
```

마지막으로, 모델의 모든 은닉 상태(hidden state)와 모든 어텐션 가중치(attention weight)를 리턴하도록 설정할 수 있습니다.

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
