[🔗 Docs >> Using Transformers >> Summary of the tasks](https://huggingface.co/transformers/task_summary.html)

이 페이지에는 라이브러리 사용 시 가장 많이 적용되는 사례가 소개되어 있습니다. 허깅페이스 트랜스포머의 모델들은 다양한 구성과 사용 사례를 지원합니다. 가장 간단한 것은 질문 답변(question answering), 시퀀스 분류(sequence classification), 개체명 인식(named entity recognition) 등과 같은 작업에 대한 사례들입니다.

이러한 예제에서는 오토모델(auto-models)을 활용합니다. 오토모델은 주어진 체크포인트에 따라 모델을 인스턴스화하고 올바른 모델 아키텍처를 자동으로 선택하는 클래스입니다. 자세한 내용은 [AutoModel](https://huggingface.co/transformers/model_doc/auto.html#transformers.AutoModel) 문서를 참조하십시오. 문서를 참조하여 코드를 더 구체적으로 수정하고, 특정 사용 사례에 맞게 자유롭게 조정할 수 있습니다.

모델이 잘 실행되려면 해당 태스크에 해당하는 체크포인트에서 로드되어야 합니다. 이러한 체크포인트는 일반적으로 대규모 데이터 집합을 사용하여 프리트레인되고 특정 태스크에 대해 파인튜닝 됩니다. 이는 아래와 같습니다.

- 모든 모델이 모든 태스크에 대해 파인튜닝된 것은 아닙니다. 특정 태스크에서 모델을 파인튜닝하려면 [예제 디렉토리](https://github.com/huggingface/transformers/tree/master/examples)의 *run_$TASK.py*스크립트를 활용할 수 있습니다.
- 파인튜닝된 모델은 특정 데이터셋을 사용하여 파인튜닝되었습니다. 이 데이터셋은 사용 예제 및 도메인과 관련이 있을 수 있지만, 그렇지 않을 수도 있습니다. 앞서 언급했듯이 [예제](https://github.com/huggingface/transformers/tree/master/examples) 스크립트를 활용하여 모델을 파인튜닝하거나 모델 학습에 사용할 스크립트를 직접 작성할 수 있습니다.

추론 태스크를 위해 라이브러리에서 몇 가지 메커니즘을 사용할 수 있습니다.

- 파이프라인 : 사용하기 매우 쉬운 방식으로, 두 줄의 코드로 사용이 가능합니다.
- 직접 모델 사용하기 : 추상화가 덜 되지만, 토크나이저(파이토치/텐서플로우)에 직접 액세스할 수 있다는 점에서 유연성과 성능이 향상됩니다.

여기에 두 가지 접근 방식이 모두 제시되어 있습니다.

> 💛 주의
> 
> 여기에 제시된 모든 태스크에서는 특정 태스크에 맞게 파인튜닝된 프리트레인 체크포인트를 활용합니다. 특정 작업에서 파인튜닝 되지 않은 체크포인트를 로드하면 태스크에 사용되는 추가 헤드가 아닌 기본 트랜스포머 레이어만 로드되어 해당 헤드의 가중치가 무작위로 초기화됩니다.
이렇게 하면 랜덤으로 출력이 생성됩니다.
>

### 시퀀스 분류(Sequence Classification)

시퀀스 분류는 주어진 클래스 수에 따라 시퀀스를 분류하는 태스크입니다. 시퀀스 분류의 예시로는 이 태스크를 기반으로 하는 GLUE 데이터셋이 있습니다. GLUE 시퀀스 분류 태스크에서 모델을 파인튜닝 하려면 [*run_glue.py*](https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py), *[run_tf_glue.py](https://github.com/huggingface/transformers/blob/master/examples/tensorflow/text-classification/run_tf_glue.py)*, *[run_tf_classification.py](https://github.com/huggingface/transformers/blob/master/examples/tensorflow/text-classification/run_tf_text_classification.py)* 또는 *[run_xnli.py](https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_xnli.py)* 스크립트를 활용할 수 있습니다.

다음은 파이프라인을 사용하여 시퀀스가 긍정인지 부정인지를 식별하여 감성분석을 수행하는 예입니다. GLUE 태스크인 sst2에서 파인튜닝된 모델을 활용합니다.

이렇게 하면 다음과 같이 스코어와 함께 라벨(POSITIVE-긍정 or NEGATIVE-부정)이 반환됩니다.

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result = classifier("I hate you")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

result = classifier("I love you")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
```

다음은 모델을 사용하여 두 시퀀스가 서로 같은 의미의 다른 문장인지의 여부(paraphrase or not)를 결정하는 시퀀스 분류의 예입니다. 프로세스는 다음과 같습니다.

1. 체크포인트 이름에서 토크나이저 및 모델을 인스턴스화합니다. 모델은 BERT 모델로서 식별되며 체크포인트에 저장된 가중치로 로드됩니다.
2. 올바른 모델별 구분 기호, 토큰 유형 ID 및 어텐션 마스크(토크나이저에 의해 자동으로 작성됨)를 사용하여 두 문장의 시퀀스를 작성합니다.
3. 모델을 통해 이 시퀀스를 전달하고 사용 가능한 두 클래스 중 하나인 0(no paraphrase)과 1(paraphrase) 중 하나로 분류합니다.
4. 클래스 분류에 대한 확률을 계산하기 위해 결과에 소프트맥스 함수를 적용하여 계산합니다.
5. 결과를 프린트합니다.

```python
# Pytorch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

classes = ["not paraphrase", "is paraphrase"]

sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

# The tokenizer will automatically add any model specific separators (i.e. <CLS> and <SEP>) and tokens to
# the sequence, as well as compute the attention masks.
paraphrase = tokenizer(sequence_0, sequence_2, return_tensors="pt")
not_paraphrase = tokenizer(sequence_0, sequence_1, return_tensors="pt")

paraphrase_classification_logits = model(**paraphrase).logits
not_paraphrase_classification_logits = model(**not_paraphrase).logits

paraphrase_results = torch.softmax(paraphrase_classification_logits, dim=1).tolist()[0]
not_paraphrase_results = torch.softmax(not_paraphrase_classification_logits, dim=1).tolist()[0]

# Should be paraphrase
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(paraphrase_results[i] * 100))}%")
"""
not paraphrase: 10%
is paraphrase: 90%
"""

# Should not be paraphrase
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(not_paraphrase_results[i] * 100))}%")
"""
not paraphrase: 94%
is paraphrase: 6%
"""
```

```python
# Tensorflow
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

classes = ["not paraphrase", "is paraphrase"]

sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

# The tokenizer will automatically add any model specific separators (i.e. <CLS> and <SEP>) and tokens to
# the sequence, as well as compute the attention masks.
paraphrase = tokenizer(sequence_0, sequence_2, return_tensors="tf")
not_paraphrase = tokenizer(sequence_0, sequence_1, return_tensors="tf")

paraphrase_classification_logits = model(paraphrase).logits
not_paraphrase_classification_logits = model(not_paraphrase).logits

paraphrase_results = tf.nn.softmax(paraphrase_classification_logits, axis=1).numpy()[0]
not_paraphrase_results = tf.nn.softmax(not_paraphrase_classification_logits, axis=1).numpy()[0]

# Should be paraphrase
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(paraphrase_results[i] * 100))}%")
"""
not paraphrase: 10%
is paraphrase: 90%
"""

# Should not be paraphrase
for i in range(len(classes)):
    print(f"{classes[i]}: {int(round(not_paraphrase_results[i] * 100))}%")
"""
not paraphrase: 94%
is paraphrase: 6%
"""
```

### 추출 질의응답(Extractive Question Answering)

추출 질의응답은 주어진 질문 텍스트에서 답을 추출하는 작업입니다. 질문 답변 데이터셋의 예로는 해당 작업을 기반으로 하는 SQuAD 데이터셋이 있습니다. SQuAD 작업에서 모델을 파인튜닝하려면 *[run_qa.py](https://github.com/huggingface/transformers/tree/master/examples/pytorch/question-answering/run_qa.py)* 및 *[run_tf_squad.py](https://github.com/huggingface/transformers/tree/master/examples/tensorflow/question-answering/run_tf_squad.py)* 스크립트를 활용할 수 있습니다.

다음은 파이프라인을 사용하여 주어진 질문 텍스트에서 답변을 추출하는 질의응답을 수행하는 예입니다. SQuAD 데이터셋을 통해 파인튜닝된 모델을 활용합니다.

```python
from transformers import pipeline

question_answerer = pipeline("question-answering")

context = r"""
Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.
"""
```

이렇게 하면 텍스트에서 추출된 **답변**과 **신뢰 점수(confidence score)**가 텍스트에서 추출된 답변의 **위치**인 '시작' 및 '종료' 값과 함께 반환됩니다.

```python
result = question_answerer(question="What is extractive question answering?", context=context)
print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")

result = question_answerer(question="What is a good example of a question answering dataset?", context=context)
print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
```

모델 및 토크나이저를 사용하여 질문에 대답하는 예입니다. 프로세스는 다음과 같습니다.

1. 체크포인트 이름에서 토크나이저 및 모델을 인스턴스화합니다. 모델은 BERT 모델로 식별되며 체크포인트에 저장된 가중치로 로드됩니다.
2. 텍스트와 몇 가지 질문을 정의합니다.
3. 질문을 반복하고 올바른 모델별 식별자 토큰 타입 ID 및 어텐션 마스크를 사용하여 텍스트와 현재 질문의 시퀀스를 작성합니다.
4. 이 시퀀스를 모델에 전달합니다. 그러면 시작 위치와 끝 위치 모두에 대해 전체 시퀀스 토큰(질문과 텍스트)에 걸쳐 다양한 점수가 출력됩니다.
5. 토큰에 대한 확률을 얻기 위해 결과값에 소프트맥스 함수를 취합니다.
6. 식별된 시작 및 끝 위치에서 토큰을 가져와 문자열로 변환합니다.
7. 결과를 프린트합니다.

```python
# Pytorch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

text = r"""
🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
TensorFlow 2.0 and PyTorch.
"""

questions = [
    "How many pretrained models are available in 🤗 Transformers?",
    "What does 🤗 Transformers provide?",
    "🤗 Transformers provides interoperability between which frameworks?",
]

for question in questions:
    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    # Get the most likely beginning of answer with the argmax of the score
    answer_start = torch.argmax(answer_start_scores)
    # Get the most likely end of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    print(f"Question: {question}")
    print(f"Answer: {answer}")

"""
Question: How many pretrained models are available in 🤗 Transformers?
Answer: over 32 +
Question: What does 🤗 Transformers provide?
Answer: general - purpose architectures
Question: 🤗 Transformers provides interoperability between which frameworks?
Answer: tensorflow 2. 0 and pytorch
"""
```

```python
# Tensorflow
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = TFAutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

text = r"""
🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
TensorFlow 2.0 and PyTorch.
"""

questions = [
    "How many pretrained models are available in 🤗 Transformers?",
    "What does 🤗 Transformers provide?",
    "🤗 Transformers provides interoperability between which frameworks?",
]

for question in questions:
    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="tf")
    input_ids = inputs["input_ids"].numpy()[0]
    outputs = model(inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    # Get the most likely beginning of answer with the argmax of the score
    answer_start = tf.argmax(answer_start_scores, axis=1).numpy()[0]
    # Get the most likely end of answer with the argmax of the score
    answer_end = tf.argmax(answer_end_scores, axis=1).numpy()[0] + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    print(f"Question: {question}")
    print(f"Answer: {answer}")

"""
Question: How many pretrained models are available in 🤗 Transformers?
Answer: over 32 +
Question: What does 🤗 Transformers provide?
Answer: general - purpose architectures
Question: 🤗 Transformers provides interoperability between which frameworks?
Answer: tensorflow 2. 0 and pytorch
"""
```

### 언어 모델링(Language Modeling)

언어 모델링은 모델을 코퍼스에 맞추는 작업이며, 특정 도메인에 특화시킬 수 있습니다. 모든 트랜스포머 기반 모델은 언어 모델링을 변형(예: 마스크된 언어 모델링을 사용한 BERT, 일상 언어 모델링을 사용한 GPT-2)하여 훈련됩니다.

언어 모델링은 프리트레이닝 이외에도 모델 배포를 각 도메인에 맞게 특화시키기 위해 유용하게 사용될 수 있습니다. 예를 들어, 대용량 코퍼스를 통해 훈련된 언어 모델을 사용한 다음 뉴스 데이터셋 또는 과학 논문 데이터셋(예 : [LysandreJik/arxiv-nlp](https://huggingface.co/lysandre/arxiv-nlp))으로 파인튜닝하는 것입니다.

**마스크된 언어 모델링(Masked Language Modeling)**

마스크된 언어 모델링은 마스킹 토큰을 사용하여 순서대로 토큰을 마스킹하고 모델이 해당 마스크를 적절한 토큰으로 채우도록 요청하는 작업입니다. 따라서 모델이 오른쪽 컨텍스트(마스크 오른쪽의 토큰)와 왼쪽 컨텍스트(마스크 왼쪽의 토큰)를 모두 살펴볼 수 있습니다. 이러한 훈련은 SQuAD(질의응답, [Lewis, Lui, Goyal et al](https://arxiv.org/abs/1910.13461), 파트 4.2)와 같은 양방향 컨텍스트를 필요로 하는 다운스트림 작업에 대한 강력한 기초 모델을 만듭니다. 마스킹된 언어 모델링 작업에서 모델을 파인튜닝하려면 *[run_mlm.py](https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py)* 스크립트를 활용할 수 있습니다.

다음은 파이프라인을 사용하여 시퀀스에서 마스크를 교체하는 예입니다.

```python
from transformers import pipeline

unmasker = pipeline("fill-mask")
```
그러면 마스크가 채워진 시퀀스, 스코어 및 토큰ID가 토크나이저를 통해 출력됩니다.

```python
from pprint import pprint
pprint(unmasker(f"HuggingFace is creating a {unmasker.tokenizer.mask_token} that the community uses to solve NLP tasks."))
[{'score': 0.1793,
  'sequence': 'HuggingFace is creating a tool that the community uses to solve '
              'NLP tasks.',
  'token': 3944,
  'token_str': ' tool'},
 {'score': 0.1135,
  'sequence': 'HuggingFace is creating a framework that the community uses to '
              'solve NLP tasks.',
  'token': 7208,
  'token_str': ' framework'},
 {'score': 0.0524,
  'sequence': 'HuggingFace is creating a library that the community uses to '
              'solve NLP tasks.',
  'token': 5560,
  'token_str': ' library'},
 {'score': 0.0349,
  'sequence': 'HuggingFace is creating a database that the community uses to '
              'solve NLP tasks.',
  'token': 8503,
  'token_str': ' database'},
 {'score': 0.0286,
  'sequence': 'HuggingFace is creating a prototype that the community uses to '
              'solve NLP tasks.',
  'token': 17715,
  'token_str': ' prototype'}]
```
다음은 모델 및 토크나이저를 사용하여 마스킹된 언어 모델링을 수행하는 예입니다. 프로세스는 다음과 같습니다.

1. 체크포인트 이름에서 토크라이저 및 모델을 인스턴스화합니다. 여기서는 DistilBERT 모델을 사용할 것이고, 가중치가 체크포인트에 저장됩니다.
2. 단어 대신 tokenizer.mask_token을 배치하여 마스킹된 토큰으로 시퀀스를 정의합니다.
3. 해당 시퀀스를 ID 목록으로 인코딩하고 해당 목록에서 마스킹된 토큰의 위치를 찾습니다.
4. 마스킹된 토큰의 인덱스에서 예측값을 검색합니다. 이 텐서는 어휘와 크기가 같고, 값은 각 토큰에 귀속되는 점수입니다. 이 모델은 그런 맥락에서 가능성이 높다고 생각되는 토큰에 더 높은 점수를 부여합니다.
5. PyTorch topk 또는 TensorFlow top_k 메서드를 사용하여 상위 5개의 토큰을 검색합니다.
6. 마스킹된 토큰을 토큰으로 바꾸고 결과를 프린트합니다.

```python
# Pytorch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-cased")

sequence = "Distilled models are smaller than the models they mimic. Using them instead of the large " \
    f"versions would help {tokenizer.mask_token} our carbon footprint."

inputs = tokenizer(sequence, return_tensors="pt")
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

token_logits = model(**inputs).logits
mask_token_logits = token_logits[0, mask_token_index, :]

top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
"""
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help reduce our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help increase our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help decrease our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help offset our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help improve our carbon footprint.
"""
```

```python
# Tensorflow
from transformers import TFAutoModelForMaskedLM, AutoTokenizer
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
model = TFAutoModelForMaskedLM.from_pretrained("distilbert-base-cased")

sequence = "Distilled models are smaller than the models they mimic. Using them instead of the large " \
    f"versions would help {tokenizer.mask_token} our carbon footprint."

inputs = tokenizer(sequence, return_tensors="tf")
mask_token_index = tf.where(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1]

token_logits = model(**inputs).logits
mask_token_logits = token_logits[0, mask_token_index, :]

top_5_tokens = tf.math.top_k(mask_token_logits, 5).indices.numpy()

for token in top_5_tokens:
    print(sequence.replace(tokenizer.mask_token, tokenizer.decode([token])))
"""
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help reduce our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help increase our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help decrease our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help offset our carbon footprint.
Distilled models are smaller than the models they mimic. Using them instead of the large versions would help improve our carbon footprint.
"""
```

모델에서 예측한 상위 5개의 토큰들으로 이루어진 5개의 시퀀스가 프린트됩니다.

### 인과 언어 모델링(Causal Language Modeling)

인과 언어 모델링은 토큰 순서에 따라 다음 토큰을 예측하는 작업입니다. 이 과정에서는 모델이 왼쪽 컨텍스트(마스크 왼쪽에 있는 토큰)에만 집중하게 됩니다. 이러한 학습 과정은 문장 생성 작업과 특히 연관이 있습니다. 인과 언어 모델링 작업에서 모델을 파인튜닝하려면 *run_clm.py* 스크립트를 활용할 수 있습니다.

일반적으로 다음 토큰은 모델이 입력 시퀀스에서 생성하는 마지막 히든 레이어의 *logit*에서 샘플링되어 예측됩니다.

다음은 토크나이저와 모델을 사용하고 *top_k_top_p_filtering()* 메소드를 활용하여 인풋 토큰 시퀀스에 따라 다음 토큰을 샘플링하는 예입니다.

```python
# Pytorch

from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering
import torch
from torch import nn

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

sequence = f"Hugging Face is based in DUMBO, New York City, and"

inputs = tokenizer(sequence, return_tensors="pt")
input_ids = inputs["input_ids"]

# get logits of last hidden state
next_token_logits = model(**inputs).logits[:, -1, :]

# filter
filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)

# sample
probs = nn.functional.softmax(filtered_next_token_logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)

generated = torch.cat([input_ids, next_token], dim=-1)

resulting_string = tokenizer.decode(generated.tolist()[0])
print(resulting_string)
"""
Hugging Face is based in DUMBO, New York City, and ...
"""
```

```python
# Tensorflow

from transformers import TFAutoModelForCausalLM, AutoTokenizer, tf_top_k_top_p_filtering
import tensorflow as tf

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = TFAutoModelForCausalLM.from_pretrained("gpt2")

sequence = f"Hugging Face is based in DUMBO, New York City, and"

inputs = tokenizer(sequence, return_tensors="tf")
input_ids = inputs["input_ids"]

# get logits of last hidden state
next_token_logits = model(**inputs).logits[:, -1, :]

# filter
filtered_next_token_logits = tf_top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)

# sample
next_token = tf.random.categorical(filtered_next_token_logits, dtype=tf.int32, num_samples=1)

generated = tf.concat([input_ids, next_token], axis=1)

resulting_string = tokenizer.decode(generated.numpy().tolist()[0])
print(resulting_string)

"""
Hugging Face is based in DUMBO, New York City, and ...
"""
```

이렇게 하면 원래의 순서에 따라 일관성 있는 다음 토큰이 출력됩니다. 이 토큰은 우리의 경우 단어 또는 특징입니다.

다음 섹션에서는 한 번에 하나의 토큰이 아니라 지정된 길이로 여러 토큰을 생성하는 데 *generate()*를 사용하는 방법을 보여 줍니다.

### 텍스트 생성(Text Generation)

텍스트 생성(개방형 텍스트 생성이라고도 함)의 목표는 주어진 Context와 일관되게 이어지는 텍스트를 만드는 것입니다. 다음 예는 파이프라인에서 GPT-2를 사용하여 텍스트를 생성하는 방법을 보여줍니다. 기본적으로 모든 모델은 파이프라인에서 사용할 때 각 Config에서 설정한 대로 Top-K 샘플링을 적용합니다(예시 : [gpt-2 config](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json) 참조).

```python
from transformers import pipeline

text_generator = pipeline("text-generation")
print(text_generator("As far as I am concerned, I will", max_length=50, do_sample=False))

"""
[{'generated_text': 'As far as I am concerned, I will be the first to admit that I am not a fan of the idea of a
"free market." I think that the idea of a free market is a bit of a stretch. I think that the idea'}]
"""
```

여기서 모델은 "*As far as I am concerned, I will*"라는 Context에서 총 최대 길이 50개의 토큰을 가진 임의의 텍스트를 생성합니다. 백그라운드에서 파이프라인 객체는  *generate()* 메서드를 호출하여 텍스트를 생성합니다. *max_length* 및 *do_sample* 인수와 같이 이 메서드의 기본 인수는 파이프라인에서 재정의할 수 있습니다.

다음은 *XLNet* 및 해당 토크나이저를 사용한 텍스트 생성 예제이며, *generate()* 메서드를 포함하고 있습니다.

```python
# Pytorch

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("xlnet-base-cased")
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

# Padding text helps XLNet with short prompts - proposed by Aman Rusia in https://github.com/rusiaaman/XLNet-gen#methodology
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

prompt = "Today the weather is really nice and I am planning on "
inputs = tokenizer(PADDING_TEXT + prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

prompt_length = len(tokenizer.decode(inputs[0]))
outputs = model.generate(inputs, max_length=250, do_sample=True, top_p=0.95, top_k=60)
generated = prompt + tokenizer.decode(outputs[0])[prompt_length+1:]

print(generated)

"""
Today the weather is really nice and I am planning ...
"""
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("xlnet-base-cased")
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

# Padding text helps XLNet with short prompts - proposed by Aman Rusia in https://github.com/rusiaaman/XLNet-gen#methodology
PADDING_TEXT = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""

prompt = "Today the weather is really nice and I am planning on "
inputs = tokenizer(PADDING_TEXT + prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

prompt_length = len(tokenizer.decode(inputs[0]))
outputs = model.generate(inputs, max_length=250, do_sample=True, top_p=0.95, top_k=60)
generated = prompt + tokenizer.decode(outputs[0])[prompt_length+1:]

print(generated)

"""
Today the weather is really nice and I am planning ...
"""
```

텍스트 생성은 현재 PyTorch의 *GPT-2, OpenAi-GPT, CTRL, XLNet, Transpo-XL* 및 Reformer와 Tensorflow의 대부분의 모델에서도 가능합니다. 위의 예에서 볼 수 있듯이, *XLNet* 및 *Transpo-XL*이 제대로 작동하려면 패딩이 필요한 경우가 많습니다. *GPT-2*는 인과 언어 모델링 목적으로 수백만 개의 웹 페이지를 통해 학습되었기 때문에 일반적으로 개방형 텍스트 생성에 적합합니다.

텍스트 생성을 위해 다양한 디코딩 전략을 적용하는 방법에 대한 자세한 내용은 [텍스트 생성 블로그 게시물](https://huggingface.co/blog/how-to-generate)을 참조하십시오.

### 개체명 인식(Named Entity Recognition)

개체명 인식(NER)은 개인, 기관 또는 장소의 이름 등으로 식별 가능한 클래스에 따라 토큰을 분류하는 작업입니다. 개체명 인식 데이터셋의 예로는 CoNLL-2003 데이터셋이 있습니다. NER 작업에서 모델을 파인튜닝하려는 경우 [run_ner.py](https://github.com/huggingface/transformers/blob/master/examples/pytorch/token-classification/run_ner.py) 스크립트를 활용할 수 있습니다.

다음은 파이프라인을 사용하여 개체명 인식으로 토큰을 9개 클래스 중 하나에 속하도록 예측하는 예시입니다(BIO 표현).

> **O**, 개체명이 아닌 부분
**B-MIS**, 기타 엔티티가 시작되는 부분
**I-MIS**, 기타 엔티티 
**B-PER**, 사람의 이름이 시작되는 부분
**I-PER**, 사람의 이름
**B-ORG**, 기관명이 시작되는 부분
**I-ORG**, 기관명
**B-LOC**, 장소명이 시작되는 부분
**I-LOC**, 장소명
> 

CoNLL-2003의 파인튜닝 모델을 사용하였으며, dbmdz의 @stefan-it에 의해 파인튜닝 되었습니다.

```python
from transformers import pipeline

ner_pipe = pipeline("ner")

sequence = """Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO,
therefore very close to the Manhattan Bridge which is visible from the window."""
```

이렇게 하면 위에서 정의한 9개 클래스의 엔티티 중 하나로 식별된 모든 단어 목록이 출력됩니다. 예상되는 결과는 다음과 같습니다.

```python
for entity in ner_pipe(sequence):
    print(entity)
"""
{'entity': 'I-ORG', 'score': 0.9996, 'index': 1, 'word': 'Hu', 'start': 0, 'end': 2}
{'entity': 'I-ORG', 'score': 0.9910, 'index': 2, 'word': '##gging', 'start': 2, 'end': 7}
{'entity': 'I-ORG', 'score': 0.9982, 'index': 3, 'word': 'Face', 'start': 8, 'end': 12}
{'entity': 'I-ORG', 'score': 0.9995, 'index': 4, 'word': 'Inc', 'start': 13, 'end': 16}
{'entity': 'I-LOC', 'score': 0.9994, 'index': 11, 'word': 'New', 'start': 40, 'end': 43}
{'entity': 'I-LOC', 'score': 0.9993, 'index': 12, 'word': 'York', 'start': 44, 'end': 48}
{'entity': 'I-LOC', 'score': 0.9994, 'index': 13, 'word': 'City', 'start': 49, 'end': 53}
{'entity': 'I-LOC', 'score': 0.9863, 'index': 19, 'word': 'D', 'start': 79, 'end': 80}
{'entity': 'I-LOC', 'score': 0.9514, 'index': 20, 'word': '##UM', 'start': 80, 'end': 82}
{'entity': 'I-LOC', 'score': 0.9337, 'index': 21, 'word': '##BO', 'start': 82, 'end': 84}
{'entity': 'I-LOC', 'score': 0.9762, 'index': 28, 'word': 'Manhattan', 'start': 114, 'end': 123}
{'entity': 'I-LOC', 'score': 0.9915, 'index': 29, 'word': 'Bridge', 'start': 124, 'end': 130}
"""
```

어떻게 "Huggingface" 시퀀스의 토큰이 기관명으로 식별되고 "New York City", "DUMBO" 및 "Manhattan Bridge"가 장소명으로 식별되는지에 주의해서 보십시오.

다음은 모델 및 토크나이저를 사용하여 개체명 인식을 수행하는 예시입니다. 프로세스는 다음과 같습니다.

1. 체크포인트에서 토크나이저 및 모델을 인스턴스화합니다. BERT 모델을 사용하고, 체크포인트에 저장된 가중치를 로드합니다.
2. 각 시퀀스의 엔티티를 정의합니다. 예를 들어 "Hugging Face"를 기관명으로, "New York City"를 장소명으로 정의할 수 있습니다.
3. 단어를 토큰으로 분할하여 예측에 매핑할 수 있도록 합니다. 우리는 먼저 시퀀스를 완전히 인코딩하고 디코딩하여 특별한 토큰이 포함된 문자열을 남겨두도록 합니다.
4. 해당 시퀀스를 ID로 인코딩합니다(특수 토큰이 자동으로 추가됨).
5. 입력 토큰을 모델에 전달하고, 첫 번째 출력을 가져와서 예측을 수행합니다. 이 결과를 각 토큰에 대해 매칭 가능한 9개 클래스와 대조합니다. 각 토큰에 대해 가장 가능성이 높은 클래스를 검색하기 위해 argmax 함수를 사용합니다.
6. 각각의 토큰을 예측 결과와 묶어 프린트합니다.

```python
# Pytorch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, " \
           "therefore very close to the Manhattan Bridge."

inputs = tokenizer(sequence, return_tensors="pt")
tokens = inputs.tokens()

outputs = model(**inputs).logits
predictions = torch.argmax(outputs, dim=2)
```

```python
# Tensorflow
from transformers import TFAutoModelForTokenClassification, AutoTokenizer
import tensorflow as tf

model = TFAutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, " \
           "therefore very close to the Manhattan Bridge."

inputs = tokenizer(sequence, return_tensors="tf")
tokens = inputs.tokens()

outputs = model(**inputs)[0]
predictions = tf.argmax(outputs, axis=2)
```

해당 예측 결과로 매핑된 각 토큰 목록을 출력합니다. 파이프라인과 달리 모든 토큰에 예측 결과가 나오게 되는데, 엔티티가 없는 토큰인 클래스 0의 경우를 제거하지 않았기 때문입니다.

위의 예시에서 예측 결과는 정수로 표현됩니다. 아래 그림과 같이 정수 형태의 클래스 번호를 클래스 이름으로 바꾸기 위해 model.config.id2label 속성을 사용할 수 있습니다.

```python
for token, prediction in zip(tokens, predictions[0].numpy()):
    print((token, model.config.id2label[prediction]))

"""
('[CLS]', 'O')
('Hu', 'I-ORG')
('##gging', 'I-ORG')
('Face', 'I-ORG')
('Inc', 'I-ORG')
('.', 'O')
('is', 'O')
('a', 'O')
('company', 'O')
('based', 'O')
('in', 'O')
('New', 'I-LOC')
('York', 'I-LOC')
('City', 'I-LOC')
('.', 'O')
('Its', 'O')
('headquarters', 'O')
('are', 'O')
('in', 'O')
('D', 'I-LOC')
('##UM', 'I-LOC')
('##BO', 'I-LOC')
(',', 'O')
('therefore', 'O')
('very', 'O')
('close', 'O')
('to', 'O')
('the', 'O')
('Manhattan', 'I-LOC')
('Bridge', 'I-LOC')
('.', 'O')
('[SEP]', 'O')
"""
```
