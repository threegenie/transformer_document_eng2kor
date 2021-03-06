## Quick tour : Getting started on a task with a pipeline

[๐ Huggingface Transformers Docs >>Quick tour](https://huggingface.co/transformers/quicktour.html#)

Huggingface๐ค ํธ๋์คํฌ๋จธ ๋ผ์ด๋ธ๋ฌ๋ฆฌ์ ํน์ง์ ๋ํด ๊ฐ๋จํ ์์๋ณด๊ฒ ์ต๋๋ค. ์ด ๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ ํ์คํธ ๊ฐ์ฑ ๋ถ์๊ณผ ๊ฐ์ ์์ฐ์ด ์ดํด(NLU) ํ์คํฌ์, ์๋ก์ด ํ์คํธ๋ฅผ ๋ง๋ค์ด๋ด๊ฑฐ๋ ๋ค๋ฅธ ์ธ์ด๋ก ๋ฒ์ญํ๋ ๊ฒ๊ณผ ๊ฐ์ ์์ฐ์ด ์์ฑ(NLG) ํ์คํฌ๋ฅผ ์ํด ์ฌ์  ํ๋ จ๋ ๋ชจ๋ธ์ ๋ค์ด๋ก๋ํฉ๋๋ค.

๋จผ์  ํ์ดํ๋ผ์ธ API๋ฅผ ์ฝ๊ฒ ํ์ฉํ์ฌ ์ฌ์  ๊ฒ์ฆ๋ ๋ชจ๋ธ์ ์ ์ํ๊ฒ ์ฌ์ฉํ๋ ๋ฐฉ๋ฒ์ ๋ํด ์์๋ณด๊ฒ ์ต๋๋ค. ๋ํ ๋ผ์ด๋ธ๋ฌ๋ฆฌ๊ฐ ์ด๋ฌํ ๋ชจ๋ธ์ ๋ํ ์ก์ธ์ค ๊ถํ์ ์ด๋ป๊ฒ ์ ๊ณตํ๋์ง์ ๋ฐ์ดํฐ๋ฅผ ์ฌ์  ์ฒ๋ฆฌํ๋ ๋ฐ ํจ๊ณผ์ ์ธ ๋ฐฉ๋ฒ์ ๋ํด ์์๋ณด๊ฒ ์ต๋๋ค.

์์๋๋ฉด ์ข์ ์  

๋ชจ๋  ๋ฌธ์์ ์ฝ๋๋ ์ฐ์ธก์ ์ค์์น๋ฅผ ์ผ์ชฝ์ผ๋ก ๋ฐ๊พธ๋ฉด Pytorch๋ก, ๋ฐ๋๋ก ๋ฐ๊พธ๋ฉด Tensorflow๋ก ๋ณผ ์ ์์ต๋๋ค. ๋ง์ฝ ๊ทธ๋ ๊ฒ ์ค์ ๋์ด ์์ง ์๋ค๋ฉด, ์ฝ๋๋ฅผ ์์ ํ์ง ์์๋ ๋ ๊ฐ์ง ์ธ์ด์์ ๋ชจ๋ ์๋ํฉ๋๋ค. 

### ํ์ดํ๋ผ์ธ์ผ๋ก ์์ ์์ํ๊ธฐ

[๐ Getting started on a task with a pipeline](https://huggingface.co/transformers/quicktour.html#getting-started-on-a-task-with-a-pipeline)

[๐บ The pipeline function](https://youtu.be/tiZFewofSLM)

์ฃผ์ด์ง ํ์คํฌ์์ ์ฌ์ ํ์ต๋ชจ๋ธ(Pre-trained Model)์ ์ฌ์ฉํ๋ ๊ฐ์ฅ ์ฌ์ด ๋ฐฉ๋ฒ์ pipeline() ํจ์๋ฅผ ์ฌ์ฉํ๋ ๊ฒ ์๋๋ค.

ํธ๋์คํฌ๋จธ๋ ์๋์ ๊ฐ์ ์์๋ค์ ์ ๊ณตํฉ๋๋ค. 

- ๊ฐ์ฑ ๋ถ์(Sentiment Analysis): ํ์คํธ์ ๊ธ์  or ๋ถ์  ํ๋ณ
- ์๋ฌธ ํ์คํธ ์์ฑ(Text Generation) : ํ๋กฌํํธ๋ฅผ ์ ๊ณตํ๊ณ , ๋ชจ๋ธ์ด ๋ท ๋ฌธ์ฅ์ ์์ฑํจ
- ๊ฐ์ฒด๋ช ์ธ์(Name Entity Recognition, NER): ์๋ ฅ ๋ฌธ์ฅ์์ ๊ฐ ๋จ์ด์ ๋ํ๋ด๋ ์ํฐํฐ(์ฌ์ฉ์, ์ฅ์ ๋ฑ)๋ก ๋ผ๋ฒจ์ ์ง์ ํจ
- ์ง์์๋ต(Question Answering): ๋ชจ๋ธ์ ๋ฌธ๋งฅ(Context)๊ณผ ์ง๋ฌธ์ ์ ๊ณตํ๊ณ  ๋ฌธ๋งฅ์์ ์ ๋ต ์ถ์ถ
- ๋น์นธ ์ฑ์ฐ๊ธฐ(Filling Masked Text): ๋ง์คํฌ๋ ๋จ์ด๊ฐ ํฌํจ๋ ํ์คํธ([MASK]๋ก ๋์ฒด๋จ)๋ฅผ ์ฃผ๋ฉด ๋น ์นธ์ ์ฑ์
- ์์ฝ(Summarization): ๊ธด ํ์คํธ์ ์์ฝ๋ณธ์ ์์ฑ
- ๋ฒ์ญ(Translation): ํ์คํธ๋ฅผ ๋ค๋ฅธ ์ธ์ด๋ก ๋ฒ์ญ
- ํน์ฑ ์ถ์ถ(Feature Extraction): ํ์คํธ๋ฅผ ํ์ ํํ๋ก ๋ฐํ

๊ฐ์ฑ๋ถ์์ด ์ด๋ป๊ฒ ์ด๋ฃจ์ด์ง๋์ง ์์๋ณด๊ฒ ์ต๋๋ค. (๊ธฐํ ์์๋ค์ [task summary](https://huggingface.co/transformers/task_summary.html)์์ ๋ค๋ฃน๋๋ค)

```python
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
```

์ด ์ฝ๋๋ฅผ ์ฒ์ ์๋ ฅํ๋ฉด ์ฌ์ ํ์ต๋ชจ๋ธ๊ณผ ํด๋น ํ ํฌ๋์ด์ ๊ฐ ๋ค์ด๋ก๋ ๋ฐ ์บ์๋ฉ๋๋ค. ์ดํ์ ๋ ๊ฐ์ง ๋ชจ๋์ ๋ํด ์์๋ณด๊ฒ ์ง๋ง, ํ ํฌ๋์ด์ ์ ์ญํ ์ ๋ชจ๋ธ์ ๋ํ ํ์คํธ๋ฅผ ์ ์ฒ๋ฆฌํ๊ณ  ์์ธก ์์์ ์ํํ๋ ๊ฒ์๋๋ค. ํ์ดํ๋ผ์ธ์ ์ด ๋ชจ๋  ๊ฒ์ ๊ทธ๋ฃนํํ๊ณ  ์์ธก ๊ฒฐ๊ณผ๋ฅผ ํ์ฒ๋ฆฌํ์ฌ ์ฌ์ฉ์๊ฐ ์ฝ์ ์ ์๋๋ก ๋ณํํฉ๋๋ค. 

์๋ฅผ ๋ค๋ฉด ์ดํ์ ๊ฐ์ต๋๋ค. 

```python
classifier('We are very happy to show you the ๐ค Transformers library.')

# [{'label': 'POSITIVE', 'score': 0.9998}]
```

ํฅ๋ฏธ๋กญ์ง ์๋์? ์ด๋ฌํ ๋ฌธ์ฅ๋ค์ ๋ฃ์ผ๋ฉด ๋ชจ๋ธ์ ํตํด ์ ์ฒ๋ฆฌ๋๊ณ , ๋์๋๋ฆฌ ํํ์ ๋ฆฌ์คํธ๋ฅผ ๋ฐํํฉ๋๋ค.

```python
results = classifier(["We are very happy to show you the ๐ค Transformers library.",
           "We hope you don't hate it."])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

# label: POSITIVE, with score: 0.9998
# label: NEGATIVE, with score: 0.5309
```

๋์ฉ๋ ๋ฐ์ดํฐ์๊ณผ ํจ๊ป ์ด ๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ฅผ ์ฌ์ฉํ๋ ค๋ฉด [iterating over a pipeline](https://huggingface.co/transformers/main_classes/pipelines.html)์ ์ฐธ์กฐํ์ธ์.

์ฌ๋ฌ๋ถ์ ์์ ์์์์ ๋ ๋ฒ์งธ ๋ฌธ์ฅ์ด ๋ถ์ ์ ์ผ๋ก ๋ถ๋ฅ๋์๋ค๋ ๊ฒ์ ์ ์ ์์ง๋ง(๊ธ์  ๋๋ ๋ถ์ ์ผ๋ก ๋ถ๋ฅ๋์ด์ผ ํฉ๋๋ค), ์ค์ฝ์ด๋ 0.5์ ๊ฐ๊น์ด ์ค๋ฆฝ์ ์ธ ์ ์์๋๋ค.

์ด ํ์ดํ๋ผ์ธ์ ๊ธฐ๋ณธ์ ์ผ๋ก ๋ค์ด๋ก๋๋๋ ๋ชจ๋ธ์ distilbert-base-uncaseed-finetuned-sst-2-english์๋๋ค. [๋ชจ๋ธ ํ์ด์ง](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)์์ ๋ ์์ธํ ์ ๋ณด๋ฅผ ์ป์ ์ ์์ต๋๋ค. ์ด ๋ชจ๋ธ์ DistilBERT ๊ตฌ์กฐ๋ฅผ ์ฌ์ฉํ๋ฉฐ, ๊ฐ์ฑ ๋ถ์ ์์์ ์ํด SST-2๋ผ๋ ๋ฐ์ดํฐ์์ ํตํด ๋ฏธ์ธ ์กฐ์ (fine-tuning)๋์์ต๋๋ค.

๋ง์ฝ ๋ค๋ฅธ ๋ชจ๋ธ์ ์ฌ์ฉํ๊ธธ ์ํ๋ค๋ฉด(์๋ฅผ ๋ค์ด ํ๋์ค์ด ๋ฐ์ดํฐ), ์ฐ๊ตฌ์์์ ๋๋์ ๋ฐ์ดํฐ๋ฅผ ํตํด ์ฌ์ ํ์ต๋ ๋ชจ๋ธ๊ณผ ์ปค๋ฎค๋ํฐ ๋ชจ๋ธ(ํน์  ๋ฐ์ดํฐ์์ ํตํด ๋ฏธ์ธ์กฐ์ ๋ ๋ฒ์ ์ ๋ชจ๋ธ)๋ค์ ์์งํ๋ ๋ชจ๋ธ ํ๋ธ์์ ๋ค๋ฅธ ๋ชจ๋ธ์ ๊ฒ์ํ  ์ ์์ต๋๋ค. 'French'๋ 'text-classification' ํ๊ทธ๋ฅผ ์ ์ฉํ๋ฉด 'nlptown/bert-base-multilingual-uncased-sentiment'๋ชจ๋ธ์ ์ฌ์ฉํด ๋ณด๋ผ๋ ๊ฒฐ๊ณผ๋ฅผ ์ป์ ์ ์์ต๋๋ค. 

์ด๋ป๊ฒ ๋ค๋ฅธ ๋ชจ๋ธ์ ์ ์ฉํ ์ง ์์๋ด์๋ค.

pipeline() ํจ์์ ๋ชจ๋ธ๋ช์ ๋ฐ๋ก ๋๊ฒจ์ค ์ ์์ต๋๋ค.

```python
classifier = pipeline('sentiment-analysis', model="nlptown/bert-base-multilingual-uncased-sentiment")
```

์ด ๋ถ๋ฅ๊ธฐ๋ ์ด์  ์์ด, ํ๋์ค์ด๋ฟ๋ง ์๋๋ผ ๋ค๋๋๋์ด, ๋์ผ์ด, ์ดํ๋ฆฌ์์ด, ์คํ์ธ์ด๋ก ๋ ํ์คํธ๋ ์ฒ๋ฆฌํ  ์ ์์ต๋๋ค! ๋ํ ์ฌ์ ํ์ต๋ ๋ชจ๋ธ์ ์ ์ฅํ ๋ก์ปฌ ํด๋๋ก ์ด๋ฆ์ ๋ฐ๊ฟ ์๋ ์์ต๋๋ค(์ดํ ์ฐธ์กฐ). ๋ชจ๋ธ ๊ฐ์ฒด ๋ฐ ์ฐ๊ด๋ ํ ํฐ๋์ด์ ๋ฅผ ์ ๋ฌํ  ์๋ ์์ต๋๋ค.

์ด๋ฅผ ์ํด ๋ ๊ฐ์ ํด๋์ค๊ฐ ํ์ํฉ๋๋ค. 

์ฒซ ๋ฒ์งธ๋ AutoTokenizer์๋๋ค. ์ด๊ฒ์ ์ ํํ ๋ชจ๋ธ๊ณผ ์ฐ๊ฒฐ๋ ํ ํฌ๋์ด์ ๋ฅผ ๋ค์ด๋ก๋ํ๊ณ  ์ธ์คํด์คํํ๋ ๋ฐ ์ฌ์ฉ๋ฉ๋๋ค. 

๋ ๋ฒ์งธ๋ AutoModelForSequenceClassification(or TensorFlow -  TFAutoModelForSequenceClassification)์ผ๋ก, ๋ชจ๋ธ ์์ฒด๋ฅผ ๋ค์ด๋ก๋ํ๋ ๋ฐ ์ฌ์ฉ๋ฉ๋๋ค. ๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ฅผ ๋ค๋ฅธ ์์์ ์ฌ์ฉํ๋ ๊ฒฝ์ฐ ๋ชจ๋ธ์ ํด๋์ค๊ฐ ๋ณ๊ฒฝ๋ฉ๋๋ค. [Task summary](https://huggingface.co/transformers/task_summary.html) ํํ ๋ฆฌ์ผ์ ์ด๋ค ํด๋์ค๊ฐ ์ด๋ค ์์์ ์ฌ์ฉ๋๋์ง ์ ๋ฆฌ๋์ด ์์ต๋๋ค.

```python
# Pytorch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Tensorflow
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
```

์ด์  ์ด์ ์ ์ฐพ์ ๋ชจ๋ธ๊ณผ ํ ํฌ๋์ด์ ๋ฅผ ๋ค์ด๋ก๋ํ๋ ค๋ฉด from_pretricted() ๋ฉ์๋๋ฅผ ์ฌ์ฉํ๋ฉด ๋ฉ๋๋ค(๋ชจ๋ธ ํ๋ธ์์ model_name์ ๋ค๋ฅธ ๋ชจ๋ธ๋ก ์์ ๋กญ๊ฒ ๋ฐ๊ฟ ์ ์์).

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

# ์ด ๋ชจ๋ธ์ ํ์ดํ ์น์ ์๋ ๋ชจ๋ธ์ด๊ธฐ ๋๋ฌธ์, ํ์ํ๋ก์์ ์ด์ฉํ๋ ค๋ฉด 'from_pt'๋ผ๊ณ  ์ง์ ํด์ค์ผ ํฉ๋๋ค. 
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
```

๋น์ ์ด ๊ฐ์ง๊ณ  ์๋ ๋ฐ์ดํฐ์ ๋น์ทํ ๋ฐ์ดํฐ๋ก ์ฌ์ ํ์ต๋ ๋ชจ๋ธ์ ์ฐพ์ ์ ์๋ ๊ฒฝ์ฐ์, ๋น์ ์ ๋ฐ์ดํฐ์ ์ฌ์ ํ์ต๋ ๋ชจ๋ธ์ ์ ์ฉํ์ฌ ํ์ธํ๋์ ํด์ผ ํฉ๋๋ค. ์ด๋ฅผ ์ํ [์์  ์คํฌ๋ฆฝํธ](https://huggingface.co/transformers/examples.html)๋ฅผ ์ ๊ณตํฉ๋๋ค. ํ์ธํ๋์ ์๋ฃํ ํ์, [์ด ํํ ๋ฆฌ์ผ](https://huggingface.co/transformers/model_sharing.html)์ ํตํด ์ปค๋ฎค๋ํฐ ํ๋ธ์ ๋ชจ๋ธ์ ๊ณต์ ํด ์ฃผ์๋ฉด ๊ฐ์ฌํ๊ฒ ์ต๋๋ค.
