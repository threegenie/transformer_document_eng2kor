## Quick tour : Getting started on a task with a pipeline

[🔗 Huggingface Transformers Docs >>Quick tour](https://huggingface.co/transformers/quicktour.html#)

Huggingface🤗 트랜스포머 라이브러리의 특징에 대해 간단히 알아보겠습니다. 이 라이브러리는 텍스트 감성 분석과 같은 자연어 이해(NLU) 태스크와, 새로운 텍스트를 만들어내거나 다른 언어로 번역하는 것과 같은 자연어 생성(NLG) 태스크를 위해 사전 훈련된 모델을 다운로드합니다.

먼저 파이프라인 API를 쉽게 활용하여 사전 검증된 모델을 신속하게 사용하는 방법에 대해 알아보겠습니다. 또한 라이브러리가 이러한 모델에 대한 액세스 권한을 어떻게 제공하는지와 데이터를 사전 처리하는 데 효과적인 방법에 대해 알아보겠습니다.

알아두면 좋은 점 

모든 문서의 코드는 우측의 스위치를 왼쪽으로 바꾸면 Pytorch로, 반대로 바꾸면 Tensorflow로 볼 수 있습니다. 만약 그렇게 설정되어 있지 않다면, 코드를 수정하지 않아도 두 가지 언어에서 모두 작동합니다. 

### 파이프라인으로 작업 시작하기

[🔗 Getting started on a task with a pipeline](https://huggingface.co/transformers/quicktour.html#getting-started-on-a-task-with-a-pipeline)

[📺 The pipeline function](https://youtu.be/tiZFewofSLM)

주어진 테스크에서 사전학습모델(Pre-trained Model)을 사용하는 가장 쉬운 방법은 pipeline() 함수를 사용하는 것 입니다.

트랜스포머는 아래와 같은 작업들을 제공합니다. 

- 감성 분석(Sentiment Analysis): 텍스트의 긍정 or 부정 판별
- 영문 텍스트 생성(Text Generation) : 프롬프트를 제공하고, 모델이 뒷 문장을 생성함
- 개체명 인식(Name Entity Recognition, NER): 입력 문장에서 각 단어에 나타내는 엔티티(사용자, 장소 등)로 라벨을 지정함
- 질의응답(Question Answering): 모델에 문맥(Context)과 질문을 제공하고 문맥에서 정답 추출
- 빈칸 채우기(Filling Masked Text): 마스크된 단어가 포함된 텍스트([MASK]로 대체됨)를 주면 빈 칸을 채움
- 요약(Summarization): 긴 텍스트의 요약본을 생성
- 번역(Translation): 텍스트를 다른 언어로 번역
- 특성 추출(Feature Extraction): 텍스트를 텐서 형태로 반환

감성분석이 어떻게 이루어지는지 알아보겠습니다. (기타 작업들은 [task summary](https://huggingface.co/transformers/task_summary.html)에서 다룹니다)

```python
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
```

이 코드를 처음 입력하면 사전학습모델과 해당 토크나이저가 다운로드 및 캐시됩니다. 이후에 두 가지 모두에 대해 알아보겠지만, 토크나이저의 역할은 모델에 대한 텍스트를 전처리하고 예측 작업을 수행하는 것입니다. 파이프라인은 이 모든 것을 그룹화하고 예측 결과를 후처리하여 사용자가 읽을 수 있도록 변환합니다. 

예를 들면 이하와 같습니다. 

```python
classifier('We are very happy to show you the 🤗 Transformers library.')

# [{'label': 'POSITIVE', 'score': 0.9998}]
```

흥미롭지 않나요? 이러한 문장들을 넣으면 모델을 통해 전처리되고, 딕셔너리 형태의 리스트를 반환합니다.

```python
results = classifier(["We are very happy to show you the 🤗 Transformers library.",
           "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

# label: POSITIVE, with score: 0.9998
# label: NEGATIVE, with score: 0.5309
```

대용량 데이터셋과 함께 이 라이브러리를 사용하려면 [iterating over a pipeline](https://huggingface.co/transformers/main_classes/pipelines.html)을 참조하세요.

여러분은 위의 예시에서 두 번째 문장이 부정적으로 분류되었다는 것을 알 수 있지만(긍정 또는 부정으로 분류되어야 합니다), 스코어는 0.5에 가까운 중립적인 점수입니다.

이 파이프라인에 기본적으로 다운로드되는 모델은 distilbert-base-uncaseed-finetuned-sst-2-english입니다. [모델 페이지](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)에서 더 자세한 정보를 얻을 수 있습니다. 이 모델은 DistilBERT 구조를 사용하며, 감성 분석 작업을 위해 SST-2라는 데이터셋을 통해 미세 조정(fine-tuning)되었습니다.

만약 다른 모델을 사용하길 원한다면(예를 들어 프랑스어 데이터), 연구소에서 대량의 데이터를 통해 사전학습된 모델과 커뮤니티 모델(특정 데이터셋을 통해 미세조정된 버전의 모델)들을 수집하는 모델 허브에서 다른 모델을 검색할 수 있습니다. 'French'나 'text-classification' 태그를 적용하면 'nlptown/bert-base-multilingual-uncased-sentiment'모델을 사용해 보라는 결과를 얻을 수 있습니다. 

어떻게 다른 모델을 적용할지 알아봅시다.

pipeline() 함수에 모델명을 바로 넘겨줄 수 있습니다.

```python
classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")
```

이 분류기는 이제 영어, 프랑스어뿐만 아니라 네덜란드어, 독일어, 이탈리아어, 스페인어로 된 텍스트도 처리할 수 있습니다! 또한 사전학습된 모델을 저장한 로컬 폴더로 이름을 바꿀 수도 있습니다(이하 참조). 모델 개체 및 연관된 토큰나이저를 전달할 수도 있습니다.

이를 위해 두 개의 클래스가 필요합니다. 

첫 번째는 AutoTokenizer입니다. 이것은 선택한 모델과 연결된 토크나이저를 다운로드하고 인스턴스화하는 데 사용됩니다. 

두 번째는 AutoModelForSequenceClassification(or TensorFlow -  TFAutoModelForSequenceClassification)으로, 모델 자체를 다운로드하는 데 사용됩니다. 라이브러리를 다른 작업에 사용하는 경우 모델의 클래스가 변경됩니다. [Task summary](https://huggingface.co/transformers/task_summary.html) 튜토리얼에 어떤 클래스가 어떤 작업에 사용되는지 정리되어 있습니다.

```python
# Pytorch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Tensorflow
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
```

이제 이전에 찾은 모델과 토크나이저를 다운로드하려면 from_pretricted() 메서드를 사용하면 됩니다(모델 허브에서 model_name을 다른 모델로 자유롭게 바꿀 수 있음).

```python
# Pytorch
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
```

```python
# Tensorflow
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

# 이 모델은 파이토치에 있는 모델이기 때문에, 텐서플로에서 이용하려면 'from_pt'라고 지정해줘야 합니다. 
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
```

당신이 가지고 있는 데이터와 비슷한 데이터로 사전학습된 모델을 찾을 수 없는 경우엔, 당신의 데이터에 사전학습된 모델을 적용하여 파인튜닝을 해야 합니다. 이를 위한 [예제 스크립트](https://huggingface.co/transformers/examples.html)를 제공합니다. 파인튜닝을 완료한 후엔, [이 튜토리얼](https://huggingface.co/transformers/model_sharing.html)을 통해 커뮤니티 허브에 모델을 공유해 주시면 감사하겠습니다.
