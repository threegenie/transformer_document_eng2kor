[π Docs >> Using Transformers >> Summary of the tasks](https://huggingface.co/transformers/task_summary.html)

μ΄ νμ΄μ§μλ λΌμ΄λΈλ¬λ¦¬ μ¬μ© μ κ°μ₯ λ§μ΄ μ μ©λλ μ¬λ‘κ° μκ°λμ΄ μμ΅λλ€. νκΉνμ΄μ€ νΈλμ€ν¬λ¨Έμ λͺ¨λΈλ€μ λ€μν κ΅¬μ±κ³Ό μ¬μ© μ¬λ‘λ₯Ό μ§μν©λλ€. κ°μ₯ κ°λ¨ν κ²μ μ§λ¬Έ λ΅λ³(question answering), μνμ€ λΆλ₯(sequence classification), κ°μ²΄λͺ μΈμ(named entity recognition) λ±κ³Ό κ°μ μμμ λν μ¬λ‘λ€μλλ€.

μ΄λ¬ν μμ μμλ μ€ν λͺ¨λΈ(auto-models)μ νμ©ν©λλ€. μ€ν λͺ¨λΈμ μ£Όμ΄μ§ μ²΄ν¬ν¬μΈνΈμ λ°λΌ λͺ¨λΈμ μΈμ€ν΄μ€ννκ³  μ¬λ°λ₯Έ λͺ¨λΈ μν€νμ²λ₯Ό μλμΌλ‘ μ ννλ ν΄λμ€μλλ€. μμΈν λ΄μ©μ [AutoModel](https://huggingface.co/transformers/model_doc/auto.html#transformers.AutoModel) λ¬Έμλ₯Ό μ°Έμ‘°νμ­μμ€. λ¬Έμλ₯Ό μ°Έμ‘°νμ¬ μ½λλ₯Ό λ κ΅¬μ²΄μ μΌλ‘ μμ νκ³ , νΉμ  μ¬μ© μ¬λ‘μ λ§κ² μμ λ‘­κ² μ‘°μ ν  μ μμ΅λλ€.

λͺ¨λΈμ΄ μ μ€νλλ €λ©΄ ν΄λΉ νμ€ν¬μ ν΄λΉνλ μ²΄ν¬ν¬μΈνΈμμ λ‘λλμ΄μΌ ν©λλ€. μ΄λ¬ν μ²΄ν¬ν¬μΈνΈλ μΌλ°μ μΌλ‘ λκ·λͺ¨ λ°μ΄ν° μ§ν©μ μ¬μ©νμ¬ νλ¦¬νΈλ μΈλκ³  νΉμ  νμ€ν¬μ λν΄ νμΈνλ λ©λλ€. μ΄λ μλμ κ°μ΅λλ€.

- λͺ¨λ  λͺ¨λΈμ΄ λͺ¨λ  νμ€ν¬μ λν΄ νμΈνλλ κ²μ μλλλ€. νΉμ  νμ€ν¬μμ λͺ¨λΈμ νμΈνλνλ €λ©΄ [μμ  λλ ν λ¦¬](https://github.com/huggingface/transformers/tree/master/examples)μ *run_$TASK.py*μ€ν¬λ¦½νΈλ₯Ό νμ©ν  μ μμ΅λλ€.
- νμΈνλλ λͺ¨λΈμ νΉμ  λ°μ΄ν°μμ μ¬μ©νμ¬ νμΈνλλμμ΅λλ€. μ΄ λ°μ΄ν°μμ μ¬μ© μμ  λ° λλ©μΈκ³Ό κ΄λ ¨μ΄ μμ μ μμ§λ§, κ·Έλ μ§ μμ μλ μμ΅λλ€. μμ μΈκΈνλ―μ΄ [μμ ](https://github.com/huggingface/transformers/tree/master/examples) μ€ν¬λ¦½νΈλ₯Ό νμ©νμ¬ λͺ¨λΈμ νμΈνλνκ±°λ λͺ¨λΈ νμ΅μ μ¬μ©ν  μ€ν¬λ¦½νΈλ₯Ό μ§μ  μμ±ν  μ μμ΅λλ€.

μΆλ‘  νμ€ν¬λ₯Ό μν΄ λΌμ΄λΈλ¬λ¦¬μμ λͺ κ°μ§ λ©μ»€λμ¦μ μ¬μ©ν  μ μμ΅λλ€.

- νμ΄νλΌμΈ : μ¬μ©νκΈ° λ§€μ° μ¬μ΄ λ°©μμΌλ‘, λ μ€μ μ½λλ‘ μ¬μ©μ΄ κ°λ₯ν©λλ€.
- μ§μ  λͺ¨λΈ μ¬μ©νκΈ° : μΆμνκ° λ λμ§λ§, ν ν¬λμ΄μ (νμ΄ν μΉ/νμνλ‘μ°)μ μ§μ  μ‘μΈμ€ν  μ μλ€λ μ μμ μ μ°μ±κ³Ό μ±λ₯μ΄ ν₯μλ©λλ€.

μ¬κΈ°μ λ κ°μ§ μ κ·Ό λ°©μμ΄ λͺ¨λ μ μλμ΄ μμ΅λλ€.

> π μ£Όμ
> 
> μ¬κΈ°μ μ μλ λͺ¨λ  νμ€ν¬μμλ νΉμ  νμ€ν¬μ λ§κ² νμΈνλλ νλ¦¬νΈλ μΈ μ²΄ν¬ν¬μΈνΈλ₯Ό νμ©ν©λλ€. νΉμ  μμμμ νμΈνλ λμ§ μμ μ²΄ν¬ν¬μΈνΈλ₯Ό λ‘λνλ©΄ νμ€ν¬μ μ¬μ©λλ μΆκ° ν€λκ° μλ κΈ°λ³Έ νΈλμ€ν¬λ¨Έ λ μ΄μ΄λ§ λ‘λλμ΄ ν΄λΉ ν€λμ κ°μ€μΉκ° λ¬΄μμλ‘ μ΄κΈ°νλ©λλ€.
μ΄λ κ² νλ©΄ λλ€μΌλ‘ μΆλ ₯μ΄ μμ±λ©λλ€.
>

### μνμ€ λΆλ₯(Sequence Classification)

μνμ€ λΆλ₯λ μ£Όμ΄μ§ ν΄λμ€ μμ λ°λΌ μνμ€λ₯Ό λΆλ₯νλ νμ€ν¬μλλ€. μνμ€ λΆλ₯μ μμλ‘λ μ΄ νμ€ν¬λ₯Ό κΈ°λ°μΌλ‘ νλ GLUE λ°μ΄ν°μμ΄ μμ΅λλ€. GLUE μνμ€ λΆλ₯ νμ€ν¬μμ λͺ¨λΈμ νμΈνλ νλ €λ©΄ [*run_glue.py*](https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py), *[run_tf_glue.py](https://github.com/huggingface/transformers/blob/master/examples/tensorflow/text-classification/run_tf_glue.py)*, *[run_tf_classification.py](https://github.com/huggingface/transformers/blob/master/examples/tensorflow/text-classification/run_tf_text_classification.py)* λλ *[run_xnli.py](https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_xnli.py)* μ€ν¬λ¦½νΈλ₯Ό νμ©ν  μ μμ΅λλ€.

λ€μμ νμ΄νλΌμΈμ μ¬μ©νμ¬ μνμ€κ° κΈμ μΈμ§ λΆμ μΈμ§λ₯Ό μλ³νμ¬ κ°μ±λΆμμ μννλ μμλλ€. GLUE νμ€ν¬μΈ sst2μμ νμΈνλλ λͺ¨λΈμ νμ©ν©λλ€.

μ΄λ κ² νλ©΄ λ€μκ³Ό κ°μ΄ μ€μ½μ΄μ ν¨κ» λΌλ²¨(POSITIVE-κΈμ  or NEGATIVE-λΆμ )μ΄ λ°νλ©λλ€.

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result = classifier("I hate you")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

result = classifier("I love you")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
```

λ€μμ λͺ¨λΈμ μ¬μ©νμ¬ λ μνμ€κ° μλ‘ κ°μ μλ―Έμ λ€λ₯Έ λ¬Έμ₯μΈμ§μ μ¬λΆ(paraphrase or not)λ₯Ό κ²°μ νλ μνμ€ λΆλ₯μ μμλλ€. νλ‘μΈμ€λ λ€μκ³Ό κ°μ΅λλ€.

1. μ²΄ν¬ν¬μΈνΈ μ΄λ¦μμ ν ν¬λμ΄μ  λ° λͺ¨λΈμ μΈμ€ν΄μ€νν©λλ€. λͺ¨λΈμ BERT λͺ¨λΈλ‘μ μλ³λλ©° μ²΄ν¬ν¬μΈνΈμ μ μ₯λ κ°μ€μΉλ‘ λ‘λλ©λλ€.
2. μ¬λ°λ₯Έ λͺ¨λΈλ³ κ΅¬λΆ κΈ°νΈ, ν ν° μ ν ID λ° μ΄νμ λ§μ€ν¬(ν ν¬λμ΄μ μ μν΄ μλμΌλ‘ μμ±λ¨)λ₯Ό μ¬μ©νμ¬ λ λ¬Έμ₯μ μνμ€λ₯Ό μμ±ν©λλ€.
3. λͺ¨λΈμ ν΅ν΄ μ΄ μνμ€λ₯Ό μ λ¬νκ³  μ¬μ© κ°λ₯ν λ ν΄λμ€ μ€ νλμΈ 0(no paraphrase)κ³Ό 1(paraphrase) μ€ νλλ‘ λΆλ₯ν©λλ€.
4. ν΄λμ€ λΆλ₯μ λν νλ₯ μ κ³μ°νκΈ° μν΄ κ²°κ³Όμ μννΈλ§₯μ€ ν¨μλ₯Ό μ μ©νμ¬ κ³μ°ν©λλ€.
5. κ²°κ³Όλ₯Ό νλ¦°νΈν©λλ€.

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

### μΆμΆ μ§μμλ΅(Extractive Question Answering)

μΆμΆ μ§μμλ΅μ μ£Όμ΄μ§ μ§λ¬Έ νμ€νΈμμ λ΅μ μΆμΆνλ μμμλλ€. μ§λ¬Έ λ΅λ³ λ°μ΄ν°μμ μλ‘λ ν΄λΉ μμμ κΈ°λ°μΌλ‘ νλ SQuAD λ°μ΄ν°μμ΄ μμ΅λλ€. SQuAD μμμμ λͺ¨λΈμ νμΈνλνλ €λ©΄ *[run_qa.py](https://github.com/huggingface/transformers/tree/master/examples/pytorch/question-answering/run_qa.py)* λ° *[run_tf_squad.py](https://github.com/huggingface/transformers/tree/master/examples/tensorflow/question-answering/run_tf_squad.py)* μ€ν¬λ¦½νΈλ₯Ό νμ©ν  μ μμ΅λλ€.

λ€μμ νμ΄νλΌμΈμ μ¬μ©νμ¬ μ£Όμ΄μ§ μ§λ¬Έ νμ€νΈμμ λ΅λ³μ μΆμΆνλ μ§μμλ΅μ μννλ μμλλ€. SQuAD λ°μ΄ν°μμ ν΅ν΄ νμΈνλλ λͺ¨λΈμ νμ©ν©λλ€.

```python
from transformers import pipeline

question_answerer = pipeline("question-answering")

context = r"""
Extractive Question Answering is the task of extracting an answer from a text given a question. An example of a
question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune
a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.
"""
```

μ΄λ κ² νλ©΄ νμ€νΈμμ μΆμΆλ **λ΅λ³**κ³Ό **μ λ’° μ μ(confidence score)**κ° νμ€νΈμμ μΆμΆλ λ΅λ³μ **μμΉ**μΈ 'μμ' λ° 'μ’λ£' κ°κ³Ό ν¨κ» λ°νλ©λλ€.

```python
result = question_answerer(question="What is extractive question answering?", context=context)
print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")

result = question_answerer(question="What is a good example of a question answering dataset?", context=context)
print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
```

λͺ¨λΈ λ° ν ν¬λμ΄μ λ₯Ό μ¬μ©νμ¬ μ§λ¬Έμ λλ΅νλ μμλλ€. νλ‘μΈμ€λ λ€μκ³Ό κ°μ΅λλ€.

1. μ²΄ν¬ν¬μΈνΈ μ΄λ¦μμ ν ν¬λμ΄μ  λ° λͺ¨λΈμ μΈμ€ν΄μ€νν©λλ€. λͺ¨λΈμ BERT λͺ¨λΈλ‘ μλ³λλ©° μ²΄ν¬ν¬μΈνΈμ μ μ₯λ κ°μ€μΉλ‘ λ‘λλ©λλ€.
2. νμ€νΈμ λͺ κ°μ§ μ§λ¬Έμ μ μν©λλ€.
3. μ§λ¬Έμ λ°λ³΅νκ³  μ¬λ°λ₯Έ λͺ¨λΈλ³ μλ³μ ν ν° νμ ID λ° μ΄νμ λ§μ€ν¬λ₯Ό μ¬μ©νμ¬ νμ€νΈμ νμ¬ μ§λ¬Έμ μνμ€λ₯Ό μμ±ν©λλ€.
4. μ΄ μνμ€λ₯Ό λͺ¨λΈμ μ λ¬ν©λλ€. κ·Έλ¬λ©΄ μμ μμΉμ λ μμΉ λͺ¨λμ λν΄ μ μ²΄ μνμ€ ν ν°(μ§λ¬Έκ³Ό νμ€νΈ)μ κ±Έμ³ λ€μν μ μκ° μΆλ ₯λ©λλ€.
5. ν ν°μ λν νλ₯ μ μ»κΈ° μν΄ κ²°κ³Όκ°μ μννΈλ§₯μ€ ν¨μλ₯Ό μ·¨ν©λλ€.
6. μλ³λ μμ λ° λ μμΉμμ ν ν°μ κ°μ Έμ λ¬Έμμ΄λ‘ λ³νν©λλ€.
7. κ²°κ³Όλ₯Ό νλ¦°νΈν©λλ€.

```python
# Pytorch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

text = r"""
π€ Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNetβ¦) for Natural Language Understanding (NLU) and Natural
Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
TensorFlow 2.0 and PyTorch.
"""

questions = [
    "How many pretrained models are available in π€ Transformers?",
    "What does π€ Transformers provide?",
    "π€ Transformers provides interoperability between which frameworks?",
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
Question: How many pretrained models are available in π€ Transformers?
Answer: over 32 +
Question: What does π€ Transformers provide?
Answer: general - purpose architectures
Question: π€ Transformers provides interoperability between which frameworks?
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
π€ Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNetβ¦) for Natural Language Understanding (NLU) and Natural
Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
TensorFlow 2.0 and PyTorch.
"""

questions = [
    "How many pretrained models are available in π€ Transformers?",
    "What does π€ Transformers provide?",
    "π€ Transformers provides interoperability between which frameworks?",
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
Question: How many pretrained models are available in π€ Transformers?
Answer: over 32 +
Question: What does π€ Transformers provide?
Answer: general - purpose architectures
Question: π€ Transformers provides interoperability between which frameworks?
Answer: tensorflow 2. 0 and pytorch
"""
```

### μΈμ΄ λͺ¨λΈλ§(Language Modeling)

μΈμ΄ λͺ¨λΈλ§μ λͺ¨λΈμ μ½νΌμ€μ λ§μΆλ μμμ΄λ©°, νΉμ  λλ©μΈμ νΉνμν¬ μ μμ΅λλ€. λͺ¨λ  νΈλμ€ν¬λ¨Έ κΈ°λ° λͺ¨λΈμ μΈμ΄ λͺ¨λΈλ§μ λ³ν(μ: λ§μ€ν¬λ μΈμ΄ λͺ¨λΈλ§μ μ¬μ©ν BERT, μΌμ μΈμ΄ λͺ¨λΈλ§μ μ¬μ©ν GPT-2)νμ¬ νλ ¨λ©λλ€.

μΈμ΄ λͺ¨λΈλ§μ νλ¦¬νΈλ μ΄λ μ΄μΈμλ λͺ¨λΈ λ°°ν¬λ₯Ό κ° λλ©μΈμ λ§κ² νΉνμν€κΈ° μν΄ μ μ©νκ² μ¬μ©λ  μ μμ΅λλ€. μλ₯Ό λ€μ΄, λμ©λ μ½νΌμ€λ₯Ό ν΅ν΄ νλ ¨λ μΈμ΄ λͺ¨λΈμ μ¬μ©ν λ€μ λ΄μ€ λ°μ΄ν°μ λλ κ³Όν λΌλ¬Έ λ°μ΄ν°μ(μ : [LysandreJik/arxiv-nlp](https://huggingface.co/lysandre/arxiv-nlp))μΌλ‘ νμΈνλνλ κ²μλλ€.

**λ§μ€ν¬λ μΈμ΄ λͺ¨λΈλ§(Masked Language Modeling)**

λ§μ€ν¬λ μΈμ΄ λͺ¨λΈλ§μ λ§μ€νΉ ν ν°μ μ¬μ©νμ¬ μμλλ‘ ν ν°μ λ§μ€νΉνκ³  λͺ¨λΈμ΄ ν΄λΉ λ§μ€ν¬λ₯Ό μ μ ν ν ν°μΌλ‘ μ±μ°λλ‘ μμ²­νλ μμμλλ€. λ°λΌμ λͺ¨λΈμ΄ μ€λ₯Έμͺ½ μ»¨νμ€νΈ(λ§μ€ν¬ μ€λ₯Έμͺ½μ ν ν°)μ μΌμͺ½ μ»¨νμ€νΈ(λ§μ€ν¬ μΌμͺ½μ ν ν°)λ₯Ό λͺ¨λ μ΄ν΄λ³Ό μ μμ΅λλ€. μ΄λ¬ν νλ ¨μ SQuAD(μ§μμλ΅, [Lewis, Lui, Goyal et al](https://arxiv.org/abs/1910.13461), ννΈ 4.2)μ κ°μ μλ°©ν₯ μ»¨νμ€νΈλ₯Ό νμλ‘ νλ λ€μ΄μ€νΈλ¦Ό μμμ λν κ°λ ₯ν κΈ°μ΄ λͺ¨λΈμ λ§λ­λλ€. λ§μ€νΉλ μΈμ΄ λͺ¨λΈλ§ μμμμ λͺ¨λΈμ νμΈνλνλ €λ©΄ *[run_mlm.py](https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_mlm.py)* μ€ν¬λ¦½νΈλ₯Ό νμ©ν  μ μμ΅λλ€.

λ€μμ νμ΄νλΌμΈμ μ¬μ©νμ¬ μνμ€μμ λ§μ€ν¬λ₯Ό κ΅μ²΄νλ μμλλ€.

```python
from transformers import pipeline

unmasker = pipeline("fill-mask")
```
κ·Έλ¬λ©΄ λ§μ€ν¬κ° μ±μμ§ μνμ€, μ€μ½μ΄ λ° ν ν°IDκ° ν ν¬λμ΄μ λ₯Ό ν΅ν΄ μΆλ ₯λ©λλ€.

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
λ€μμ λͺ¨λΈ λ° ν ν¬λμ΄μ λ₯Ό μ¬μ©νμ¬ λ§μ€νΉλ μΈμ΄ λͺ¨λΈλ§μ μννλ μμλλ€. νλ‘μΈμ€λ λ€μκ³Ό κ°μ΅λλ€.

1. μ²΄ν¬ν¬μΈνΈ μ΄λ¦μμ ν ν¬λΌμ΄μ  λ° λͺ¨λΈμ μΈμ€ν΄μ€νν©λλ€. μ¬κΈ°μλ DistilBERT λͺ¨λΈμ μ¬μ©ν  κ²μ΄κ³ , κ°μ€μΉκ° μ²΄ν¬ν¬μΈνΈμ μ μ₯λ©λλ€.
2. λ¨μ΄ λμ  tokenizer.mask_tokenμ λ°°μΉνμ¬ λ§μ€νΉλ ν ν°μΌλ‘ μνμ€λ₯Ό μ μν©λλ€.
3. ν΄λΉ μνμ€λ₯Ό ID λͺ©λ‘μΌλ‘ μΈμ½λ©νκ³  ν΄λΉ λͺ©λ‘μμ λ§μ€νΉλ ν ν°μ μμΉλ₯Ό μ°Ύμ΅λλ€.
4. λ§μ€νΉλ ν ν°μ μΈλ±μ€μμ μμΈ‘κ°μ κ²μν©λλ€. μ΄ νμλ μ΄νμ ν¬κΈ°κ° κ°κ³ , κ°μ κ° ν ν°μ κ·μλλ μ μμλλ€. μ΄ λͺ¨λΈμ κ·Έλ° λ§₯λ½μμ κ°λ₯μ±μ΄ λλ€κ³  μκ°λλ ν ν°μ λ λμ μ μλ₯Ό λΆμ¬ν©λλ€.
5. PyTorch topk λλ TensorFlow top_k λ©μλλ₯Ό μ¬μ©νμ¬ μμ 5κ°μ ν ν°μ κ²μν©λλ€.
6. λ§μ€νΉλ ν ν°μ ν ν°μΌλ‘ λ°κΎΈκ³  κ²°κ³Όλ₯Ό νλ¦°νΈν©λλ€.

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

λͺ¨λΈμμ μμΈ‘ν μμ 5κ°μ ν ν°λ€μΌλ‘ μ΄λ£¨μ΄μ§ 5κ°μ μνμ€κ° νλ¦°νΈλ©λλ€.

### μΈκ³Ό μΈμ΄ λͺ¨λΈλ§(Causal Language Modeling)

μΈκ³Ό μΈμ΄ λͺ¨λΈλ§μ ν ν° μμμ λ°λΌ λ€μ ν ν°μ μμΈ‘νλ μμμλλ€. μ΄ κ³Όμ μμλ λͺ¨λΈμ΄ μΌμͺ½ μ»¨νμ€νΈ(λ§μ€ν¬ μΌμͺ½μ μλ ν ν°)μλ§ μ§μ€νκ² λ©λλ€. μ΄λ¬ν νμ΅ κ³Όμ μ λ¬Έμ₯ μμ± μμκ³Ό νΉν μ°κ΄μ΄ μμ΅λλ€. μΈκ³Ό μΈμ΄ λͺ¨λΈλ§ μμμμ λͺ¨λΈμ νμΈνλνλ €λ©΄ *run_clm.py* μ€ν¬λ¦½νΈλ₯Ό νμ©ν  μ μμ΅λλ€.

μΌλ°μ μΌλ‘ λ€μ ν ν°μ λͺ¨λΈμ΄ μλ ₯ μνμ€μμ μμ±νλ λ§μ§λ§ νλ  λ μ΄μ΄μ *logit*μμ μνλ§λμ΄ μμΈ‘λ©λλ€.

λ€μμ ν ν¬λμ΄μ μ λͺ¨λΈμ μ¬μ©νκ³  *top_k_top_p_filtering()* λ©μλλ₯Ό νμ©νμ¬ μΈν ν ν° μνμ€μ λ°λΌ λ€μ ν ν°μ μνλ§νλ μμλλ€.

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

μ΄λ κ² νλ©΄ μλμ μμμ λ°λΌ μΌκ΄μ± μλ λ€μ ν ν°μ΄ μΆλ ₯λ©λλ€. μ΄ ν ν°μ μ°λ¦¬μ κ²½μ° λ¨μ΄ λλ νΉμ§μλλ€.

λ€μ μΉμμμλ ν λ²μ νλμ ν ν°μ΄ μλλΌ μ§μ λ κΈΈμ΄λ‘ μ¬λ¬ ν ν°μ μμ±νλ λ° *generate()*λ₯Ό μ¬μ©νλ λ°©λ²μ λ³΄μ¬ μ€λλ€.

### νμ€νΈ μμ±(Text Generation)

νμ€νΈ μμ±(κ°λ°©ν νμ€νΈ μμ±μ΄λΌκ³ λ ν¨)μ λͺ©νλ μ£Όμ΄μ§ Contextμ μΌκ΄λκ² μ΄μ΄μ§λ νμ€νΈλ₯Ό λ§λλ κ²μλλ€. λ€μ μλ νμ΄νλΌμΈμμ GPT-2λ₯Ό μ¬μ©νμ¬ νμ€νΈλ₯Ό μμ±νλ λ°©λ²μ λ³΄μ¬μ€λλ€. κΈ°λ³Έμ μΌλ‘ λͺ¨λ  λͺ¨λΈμ νμ΄νλΌμΈμμ μ¬μ©ν  λ κ° Configμμ μ€μ ν λλ‘ Top-K μνλ§μ μ μ©ν©λλ€(μμ : [gpt-2 config](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json) μ°Έμ‘°).

```python
from transformers import pipeline

text_generator = pipeline("text-generation")
print(text_generator("As far as I am concerned, I will", max_length=50, do_sample=False))

"""
[{'generated_text': 'As far as I am concerned, I will be the first to admit that I am not a fan of the idea of a
"free market." I think that the idea of a free market is a bit of a stretch. I think that the idea'}]
"""
```

μ¬κΈ°μ λͺ¨λΈμ "*As far as I am concerned, I will*"λΌλ Contextμμ μ΄ μ΅λ κΈΈμ΄ 50κ°μ ν ν°μ κ°μ§ μμμ νμ€νΈλ₯Ό μμ±ν©λλ€. λ°±κ·ΈλΌμ΄λμμ νμ΄νλΌμΈ κ°μ²΄λ  *generate()* λ©μλλ₯Ό νΈμΆνμ¬ νμ€νΈλ₯Ό μμ±ν©λλ€. *max_length* λ° *do_sample* μΈμμ κ°μ΄ μ΄ λ©μλμ κΈ°λ³Έ μΈμλ νμ΄νλΌμΈμμ μ¬μ μν  μ μμ΅λλ€.

λ€μμ *XLNet* λ° ν΄λΉ ν ν¬λμ΄μ λ₯Ό μ¬μ©ν νμ€νΈ μμ± μμ μ΄λ©°, *generate()* λ©μλλ₯Ό ν¬ν¨νκ³  μμ΅λλ€.

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

νμ€νΈ μμ±μ νμ¬ PyTorchμ *GPT-2, OpenAi-GPT, CTRL, XLNet, Transpo-XL* λ° Reformerμ Tensorflowμ λλΆλΆμ λͺ¨λΈμμλ κ°λ₯ν©λλ€. μμ μμμ λ³Ό μ μλ―μ΄, *XLNet* λ° *Transpo-XL*μ΄ μ λλ‘ μλνλ €λ©΄ ν¨λ©μ΄ νμν κ²½μ°κ° λ§μ΅λλ€. *GPT-2*λ μΈκ³Ό μΈμ΄ λͺ¨λΈλ§ λͺ©μ μΌλ‘ μλ°±λ§ κ°μ μΉ νμ΄μ§λ₯Ό ν΅ν΄ νμ΅λμκΈ° λλ¬Έμ μΌλ°μ μΌλ‘ κ°λ°©ν νμ€νΈ μμ±μ μ ν©ν©λλ€.

νμ€νΈ μμ±μ μν΄ λ€μν λμ½λ© μ λ΅μ μ μ©νλ λ°©λ²μ λν μμΈν λ΄μ©μ [νμ€νΈ μμ± λΈλ‘κ·Έ κ²μλ¬Ό](https://huggingface.co/blog/how-to-generate)μ μ°Έμ‘°νμ­μμ€.

### κ°μ²΄λͺ μΈμ(Named Entity Recognition)

κ°μ²΄λͺ μΈμ(NER)μ κ°μΈ, κΈ°κ΄ λλ μ₯μμ μ΄λ¦ λ±μΌλ‘ μλ³ κ°λ₯ν ν΄λμ€μ λ°λΌ ν ν°μ λΆλ₯νλ μμμλλ€. κ°μ²΄λͺ μΈμ λ°μ΄ν°μμ μλ‘λ CoNLL-2003 λ°μ΄ν°μμ΄ μμ΅λλ€. NER μμμμ λͺ¨λΈμ νμΈνλνλ €λ κ²½μ° [run_ner.py](https://github.com/huggingface/transformers/blob/master/examples/pytorch/token-classification/run_ner.py) μ€ν¬λ¦½νΈλ₯Ό νμ©ν  μ μμ΅λλ€.

λ€μμ νμ΄νλΌμΈμ μ¬μ©νμ¬ κ°μ²΄λͺ μΈμμΌλ‘ ν ν°μ 9κ° ν΄λμ€ μ€ νλμ μνλλ‘ μμΈ‘νλ μμμλλ€(BIO νν).


>**O**, κ°μ²΄λͺμ΄ μλ λΆλΆ

>**B-MIS**, κΈ°ν μν°ν°κ° μμλλ λΆλΆ

>**I-MIS**, κΈ°ν μν°ν° 

>**B-PER**, μ¬λμ μ΄λ¦μ΄ μμλλ λΆλΆ

> **I-PER**, μ¬λμ μ΄λ¦

> **B-ORG**, κΈ°κ΄λͺμ΄ μμλλ λΆλΆ

> **I-ORG**, κΈ°κ΄λͺ

> **B-LOC**, μ₯μλͺμ΄ μμλλ λΆλΆ

> **I-LOC**, μ₯μλͺ



CoNLL-2003μ νμΈνλ λͺ¨λΈμ μ¬μ©νμμΌλ©°, dbmdzμ @stefan-itμ μν΄ νμΈνλ λμμ΅λλ€.

```python
from transformers import pipeline

ner_pipe = pipeline("ner")

sequence = """Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO,
therefore very close to the Manhattan Bridge which is visible from the window."""
```

μ΄λ κ² νλ©΄ μμμ μ μν 9κ° ν΄λμ€μ μν°ν° μ€ νλλ‘ μλ³λ λͺ¨λ  λ¨μ΄ λͺ©λ‘μ΄ μΆλ ₯λ©λλ€. μμλλ κ²°κ³Όλ λ€μκ³Ό κ°μ΅λλ€.

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

μ΄λ»κ² "Huggingface" μνμ€μ ν ν°μ΄ κΈ°κ΄λͺμΌλ‘ μλ³λκ³  "New York City", "DUMBO" λ° "Manhattan Bridge"κ° μ₯μλͺμΌλ‘ μλ³λλμ§μ μ£Όμν΄μ λ³΄μ­μμ€.

λ€μμ λͺ¨λΈ λ° ν ν¬λμ΄μ λ₯Ό μ¬μ©νμ¬ κ°μ²΄λͺ μΈμμ μννλ μμμλλ€. νλ‘μΈμ€λ λ€μκ³Ό κ°μ΅λλ€.

1. μ²΄ν¬ν¬μΈνΈμμ ν ν¬λμ΄μ  λ° λͺ¨λΈμ μΈμ€ν΄μ€νν©λλ€. BERT λͺ¨λΈμ μ¬μ©νκ³ , μ²΄ν¬ν¬μΈνΈμ μ μ₯λ κ°μ€μΉλ₯Ό λ‘λν©λλ€.
2. κ° μνμ€μ μν°ν°λ₯Ό μ μν©λλ€. μλ₯Ό λ€μ΄ "Hugging Face"λ₯Ό κΈ°κ΄λͺμΌλ‘, "New York City"λ₯Ό μ₯μλͺμΌλ‘ μ μν  μ μμ΅λλ€.
3. λ¨μ΄λ₯Ό ν ν°μΌλ‘ λΆν νμ¬ μμΈ‘μ λ§€νν  μ μλλ‘ ν©λλ€. μ°λ¦¬λ λ¨Όμ  μνμ€λ₯Ό μμ ν μΈμ½λ©νκ³  λμ½λ©νμ¬ νΉλ³ν ν ν°μ΄ ν¬ν¨λ λ¬Έμμ΄μ λ¨κ²¨λλλ‘ ν©λλ€.
4. ν΄λΉ μνμ€λ₯Ό IDλ‘ μΈμ½λ©ν©λλ€(νΉμ ν ν°μ΄ μλμΌλ‘ μΆκ°λ¨).
5. μλ ₯ ν ν°μ λͺ¨λΈμ μ λ¬νκ³ , μ²« λ²μ§Έ μΆλ ₯μ κ°μ Έμμ μμΈ‘μ μνν©λλ€. μ΄ κ²°κ³Όλ₯Ό κ° ν ν°μ λν΄ λ§€μΉ­ κ°λ₯ν 9κ° ν΄λμ€μ λμ‘°ν©λλ€. κ° ν ν°μ λν΄ κ°μ₯ κ°λ₯μ±μ΄ λμ ν΄λμ€λ₯Ό κ²μνκΈ° μν΄ argmax ν¨μλ₯Ό μ¬μ©ν©λλ€.
6. κ°κ°μ ν ν°μ μμΈ‘ κ²°κ³Όμ λ¬Άμ΄ νλ¦°νΈν©λλ€.

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

ν΄λΉ μμΈ‘ κ²°κ³Όλ‘ λ§€νλ κ° ν ν° λͺ©λ‘μ μΆλ ₯ν©λλ€. νμ΄νλΌμΈκ³Ό λ¬λ¦¬ λͺ¨λ  ν ν°μ μμΈ‘ κ²°κ³Όκ° λμ€κ² λλλ°, μν°ν°κ° μλ ν ν°μΈ ν΄λμ€ 0μ κ²½μ°λ₯Ό μ κ±°νμ§ μμκΈ° λλ¬Έμλλ€.

μμ μμμμ μμΈ‘ κ²°κ³Όλ μ μλ‘ ννλ©λλ€. μλ κ·Έλ¦Όκ³Ό κ°μ΄ μ μ ννμ ν΄λμ€ λ²νΈλ₯Ό ν΄λμ€ μ΄λ¦μΌλ‘ λ°κΎΈκΈ° μν΄ model.config.id2label μμ±μ μ¬μ©ν  μ μμ΅λλ€.

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

### μμ½(Summarization)

μμ½μ λ¬Έμλ κΈ°μ¬λ₯Ό λ μ§§μ νμ€νΈλ‘ μ€μ΄λ μμμλλ€. μμ½ μμμμ λͺ¨λΈμ νμΈνλνλ €λ©΄ [run_summarization.py](https://github.com/huggingface/transformers/tree/master/examples/pytorch/summarization/run_summarization.py)λ₯Ό νμ©ν  μ μμ΅λλ€.

μμ½ λ°μ΄ν°μ μλ‘λ CNN / Daily Mail λ°μ΄ν°μμ΄ μμ΅λλ€. μ΄ λ°μ΄ν°μμ κΈ΄ λ΄μ€ κΈ°μ¬λ‘ κ΅¬μ±λμ΄ μμΌλ©° μμ½ μμμ μν΄ λ§λ€μ΄μ‘μ΅λλ€. μμ½ μμμμ λͺ¨λΈμ νμΈνλνλ €λ©΄, [μ΄ λ¬Έμ](https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/README.md)μμ λ€μν μ κ·Ό λ°©μμ λ°°μΈ μ μμ΅λλ€.

λ€μμ νμ΄νλΌμΈμ μ¬μ©νμ¬ μμ½μ μννλ μμλλ€. CNN/Daily Mail λ°μ΄ν°μμΌλ‘ νμΈνλλ Bart λͺ¨λΈμ νμ©ν©λλ€.

```python
from transformers import pipeline

summarizer = pipeline("summarization")

ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
2010 marriage license application, according to court documents.
Prosecutors said the marriages were part of an immigration scam.
On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
"""
```

μμ½ νμ΄νλΌμΈμ *PreTrainedModel.generate()* λ©μλμ μμ‘΄νλ―λ‘ μλμ κ°μ΄ νμ΄νλΌμΈμμ *max_length* λ° *min_length*μ λν *PreTrainedModel.generate()*μ κΈ°λ³Έ μΈμλ₯Ό μ§μ  μ¬μ μν  μ μμ΅λλ€. μ΄λ κ² νλ©΄ λ€μκ³Ό κ°μ μμ½ κ²°κ³Όκ° μΆλ ₯λ©λλ€.

```python
print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))
"""
[{'summary_text': ' Liana Barrientos, 39, is charged with two counts of "offering a false instrument for filing in
the first degree" In total, she has been married 10 times, with nine of her marriages occurring between 1999 and
2002 . At one time, she was married to eight men at once, prosecutors say .'}]
"""
```

λ€μμ λͺ¨λΈκ³Ό ν ν¬λμ΄μ λ₯Ό μ¬μ©νμ¬ μμ½μ μννλ μμμλλ€. νλ‘μΈμ€λ λ€μκ³Ό κ°μ΅λλ€.

1. μ²΄ν¬ν¬μΈνΈμμ ν ν¬λμ΄μ  λ° λͺ¨λΈμ μΈμ€ν΄μ€νν©λλ€.  μΌλ°μ μΌλ‘ Bart λλ T5μ κ°μ μΈμ½λ-λμ½λ λͺ¨λΈμ μ¬μ©νμ¬ μνν©λλ€.
2. μμ½ν΄μΌ ν  λ¬Έμλ₯Ό μ μν©λλ€.
3. T5μ νΉμν μ λμ¬μΈ "summarize: "λ₯Ό μΆκ°ν©λλ€.
4. μμ½λ¬Έ μμ±μ μν΄ *PreTrainedModel.generate()* λ©μλλ₯Ό μ¬μ©ν©λλ€.

μ΄ μμμμλ Googleμ T5 λͺ¨λΈμ μ¬μ©ν©λλ€. λ€μ€ μμ νΌν© λ°μ΄ν°μ(CNN/Daily Mail ν¬ν¨)μμλ§ νλ¦¬νΈλ μΈμ νμμλ λΆκ΅¬νκ³  λ§€μ° μ’μ κ²°κ³Όλ₯Ό μ»μ μ μμ΅λλ€.

```python
# Pytorch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

# T5 uses a max_length of 512 so we cut the article to 512 tokens.
inputs = tokenizer("summarize: " + ARTICLE, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(
    inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True
)

print(tokenizer.decode(outputs[0]))
"""
<pad> prosecutors say the marriages were part of an immigration scam. if convicted, barrientos faces two criminal
counts of "offering a false instrument for filing in the first degree" she has been married 10 times, nine of them
between 1999 and 2002.</s>
"""
```

### λ²μ­(Translation)

λ²μ­μ ν μΈμ΄μμ λ€λ₯Έ μΈμ΄λ‘ νμ€νΈλ₯Ό λ°κΎΈλ μμμλλ€. λ²μ­ μμμμ λͺ¨λΈμ νμΈνλ νλ €λ©΄ [run_translation.py](https://github.com/huggingface/transformers/tree/master/examples/pytorch/translation/run_translation.py) μ€ν¬λ¦½νΈλ₯Ό νμ©ν  μ μμ΅λλ€.

λ²μ­ λ°μ΄ν°μμ μλ‘λ WMT English to German λ°μ΄ν°μμ΄ μλλ°, μ΄ λ°μ΄ν°μμλ μμ΄λ‘ λ λ¬Έμ₯μ΄ μλ ₯ λ°μ΄ν°λ‘, λμΌμ΄λ‘ λ λ¬Έμ₯μ΄ νκ² λ°μ΄ν°λ‘ ν¬ν¨λμ΄ μμ΅λλ€. λ²μ­ μμμμ λͺ¨λΈμ νμΈνλνλ €λ κ²½μ°μ λν΄ [μ΄ λ¬Έμ](https://github.com/huggingface/transformers/blob/master/examples/pytorch/translation/README.md)μμλ λ€μν μ κ·Ό λ°©μμ μ€λͺν©λλ€.

λ€μμ νμ΄νλΌμΈμ μ¬μ©νμ¬ λ²μ­μ μννλ μμλλ€. λ€μ€ μμ νΌν© λ°μ΄ν° μΈνΈ(WMT ν¬ν¨)μμ νλ¦¬νΈλ μΈλ T5 λͺ¨λΈμ νμ©νμ¬ λ²μ­ κ²°κ³Όλ₯Ό μ κ³΅ν©λλ€.

```python
from transformers import pipeline

translator = pipeline("translation_en_to_de")
print(translator("Hugging Face is a technology company based in New York and Paris", max_length=40))
"""
[{'translation_text': 'Hugging Face ist ein Technologieunternehmen mit Sitz in New York und Paris.'}]
"""
```

λ³μ­ νμ΄νλΌμΈμ *PreTrainedModel.generate()* λ©μλμ μμ‘΄νλ―λ‘ μμ κ°μ΄ νμ΄νλΌμΈμμ *max_length*μ λν *PreTrainedModel.generate()*μ κΈ°λ³Έ μΈμλ₯Ό μ§μ  μ¬μ μν  μ μμ΅λλ€.

λ€μμ λͺ¨λΈκ³Ό ν ν¬λμ΄μ λ₯Ό μ¬μ©νμ¬ λ²μ­μ μννλ μμμλλ€. νλ‘μΈμ€λ λ€μκ³Ό κ°μ΅λλ€.

1. μ²΄ν¬ν¬μΈνΈμμ ν ν¬λμ΄μ  λ° λͺ¨λΈμ μΈμ€ν΄μ€νν©λλ€.  μΌλ°μ μΌλ‘ Bart λλ T5μ κ°μ μΈμ½λ-λμ½λ λͺ¨λΈμ μ¬μ©νμ¬ μνν©λλ€.
2. λ²μ­ν΄μΌ ν  λ¬Έμλ₯Ό μ μν©λλ€.
3. T5μ νΉμν μ λμ¬μΈ "translate English to German:βμ μΆκ°ν©λλ€.
4. λ²μ­λ¬Έ μμ±μ μν΄ *PreTrainedModel.generate()* λ©μλλ₯Ό μ¬μ©ν©λλ€.

```python
# Pytorch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

inputs = tokenizer(
    "translate English to German: Hugging Face is a technology company based in New York and Paris",
    return_tensors="pt"
)
outputs = model.generate(inputs["input_ids"], max_length=40, num_beams=4, early_stopping=True)

print(tokenizer.decode(outputs[0]))
"""
<pad> Hugging Face ist ein Technologieunternehmen mit Sitz in New York und Paris.</s>
"""
```

```python
# Tensorflow
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

model = TFAutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

inputs = tokenizer(
    "translate English to German: Hugging Face is a technology company based in New York and Paris",
    return_tensors="tf"
)
outputs = model.generate(inputs["input_ids"], max_length=40, num_beams=4, early_stopping=True)

print(tokenizer.decode(outputs[0]))
"""
<pad> Hugging Face ist ein Technologieunternehmen mit Sitz in New York und Paris.
"""
```

μμ μμμ κ°μ΄ λ²μ­λ¬Έμ΄ μΆλ ₯λ©λλ€.
