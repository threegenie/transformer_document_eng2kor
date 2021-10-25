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
여기에 제시된 모든 태스크에서는 특정 태스크에 맞게 파인튜닝된 프리트레인 체크포인트를 활용합니다. 특정 작업에서 파인튜닝 되지 않은 체크포인트를 로드하면 태스크에 사용되는 추가 헤드가 아닌 기본 트랜스포머 레이어만 로드되어 해당 헤드의 가중치가 무작위로 초기화됩니다.
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

다음은 모델을 사용하여 두 시퀀스가 서로 같은 의미의 다른 문장인지의 여부를 결정하는 시퀀스 분류의 예입니다. 프로세스는 다음과 같습니다.

1. 체크포인트 이름에서 토크나이저 및 모델을 인스턴스화합니다. 모델은 BERT 모델로서 식별되며 체크포인트에 저장된 가중치로 로드됩니다.
2. 올바른 모델별 구분 기호, 토큰 유형 ID 및 어텐션 마스크(토크나이저에 의해 자동으로 작성됨)를 사용하여 두 문장의 시퀀스를 작성합니다.
3. 모델을 통해 이 시퀀스를 전달하고 사용 가능한 두 클래스 중 하나인 0(파라프레이스가 아님)과 1(파라프레이스임) 중 하나로 분류합니다.
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
