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
