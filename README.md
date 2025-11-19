# Machine-Learning
<a name="readme-top"></a> 

### AI (Artificial Intelligence)  
<img width="1030" height="600" alt="intro" src="https://github.com/user-attachments/assets/d99fd50e-fb58-46d2-9add-30dfe2da630c" />

#### Rule-based AI
- 특정 상황을 이해하는 전문가가 직접 입력값(문제)과 특징을 전달(규칙)하여 출력값(정답)을 내보내는 알고리즘이다.
- 광범위한 데이터 수집, 정리 또는 교육이 필요하지 않으므로 전문가의 기존 지식을 기반으로 비지니스 규칙을 정의하여 구현 복잡성을 줄일 수 있다.
- 의사 결정 프로세스가 명시적인 "if-then" 사전 정의 규칙에 의존하므로 높은 투명성을 제공한다.
- 본질적으로 정적이며 유연성이 없기 때문에, 사전 정의된 규칙을 수동으로 조정하여 변화나 진화하는 조건에만 적응할 수 있다.
- 사전 정의된 규칙이 명확한 지침을 제공하지 않는 모호하거나 불확실한 상황에 직면할 때 어려움을 겪을 수 있다.
- 미리 정의된 기준에 의존하면, 전문가 개인의 편견이 들어갈 수 밖에 없고, 이로 인해 미묘한 행동을 설명하지 못할 수 있으며 잠재적으로 불공평하거나 부정확한 평가로 이어질 수 있다.

#### Machine Learning AI
- 데이터를 기반으로 규칙성(패턴)을 학습하여 결과를 추론하는 알고리즘이다.
- 현실 세계의 패턴을 분석하여 개발자가 직접 코드를 작성하는 것이 어려웠으나 머신러닝을 이용해서 해결할 수 있다.
- <sub>※</sub> 데이터 마이닝, 음성 인식(언어 구분), 영상 인식(이미지 판별), 자연어 처리(번역, 문맥 찾기)에서 머신러닝이 적용된다.  
<sub>※ 데이터 마이닝이란, 대규모 데이터 안에서 체계적이고 자동적으로 통계적 규칙이나 짜임을 분석하여 가치있는 정보를 빼내는 과정이다.</sub>
- 데이터의 패턴을 학습하여 이를 통해 예측 등을 수행할 수 있다.  

1. 지도 학습 (Supervised Learning)  
   
> 입력값(문제)과 출력값(정답)을 전달하면, 알아서 특징을 직접 추출하는 방식이다.
> 다른 입력값(문제)과 동일한 출력값(정답)을 전달하면 새로운 특징을 알아서 추출한다.

> 문제(Feature)와 답(Target, Label)을 주는 것이다.
> - 분류 (코로나 양성/음성, 사기 번호/일반 번호 등 단일 값 예측)
> - 회귀 (1년 뒤의 매출액, 내일 주식 가격 등 연속 값 예측)
  
2. 비지도 학습 (Unsupervised Learning)
   
> 입력값(문제)만 전달하고 정답 없이 특징만 추출하는 학습이다.
> 추출한 특징에 대해 AI가 출력값(정답)을 부여하여 입력값(문제)은 출력값(정답)이라는 것을 알아낸다.

> 문제(Feature)를 주고 답은 주지 않는 것이다.
> - 군집화 (클러스터링, 비슷한 문제를 분석하여 편을 나누어 각 편으로 모으는 것)
> - 차원 축소 (문제의 개수를 압축(축소)하여 함축된 의미를 찾아내는 것)

3. 강화 학습 (Reinforcement Learning)  

> https://kr.mathworks.com/discovery/reinforcement-learning.html

#### Machine Learning의 단점
- 데이터에 의존적이다 (Garbage In, Garbage Out), 데이터가 안좋으면 결과도 안좋을 수 밖에 없다.
- 학습 데이터로 잘 만들어진 로직을 가진 모델일지라도 실제 데이터 적용 시 정확한 결과가 나오지 않을 수 있다.
- 머신러닝 학습을 통해 로직이 생기면, 나온 결과가 어떻게 이렇게 나왔는 지에 대한 분석이 쉽지 않다(블랙박스).
- 데이터를 넣으면 원하는 것처럼 좋은 결과를 얻기란 쉽지 않다.

#### R vs Python

- R  
  
> 개발 언어에 익숙하지 않지만 통계 분석에 능한 현업 사용자일 경우
> 오랜 기간동안 쌓인 다양하고 많은 통계 패키지

- Python  

> 직관적인 문법, 객체지향 함수형 프로그래밍언어, 다양한 라이브러리
> 다양한 영역 (운영체제, 서버, 네트워크 등)으로 연계 및 상용화하기 좋음

### 머신러닝과 딥러닝은 R보다는 "파이썬"을 사용하자!

### 분류 (Classifier)
- 대표적인 지도학습 방법 중 하나이며, 다양한 문제와 정답을 학습한 뒤 별도의 테스트에서 정답을 예측한다.
- 주어진 문제와 정답을 먼저 학습한 뒤 새로운 문제에 대한 정답을 예측하는 방식이다.
- 이진 분류 (Binary Classification)의 경우 정답은 0(음성, Negative)과 1(양성, Positive)과 같이 True, False값을 가진다.
- 다중 분류 (Muticlass Classification)는 정답이 가질 수 있는 값은 3개 이상이다(예: 0, 1, 2, 3).

#### 피처 (Feature)
- 데이터 세트의 일반 컬럼이며, 2차원 이상의 다차원 데이터까지 통들어 피처라고 한다.
- 타겟을 제외한 나머지 속성을 의미한다.

#### 레이블(Label), 클래스(Class), 타겟(Target), 결정(Decision)
- 지도 학습 시, 데이터의 학습을 위해 주어지는 정답을 의미한다.
- 지도 학습 중, 분류의 경우 이를 레이블 또는 클래스라고도 부른다.

<img width="600px" alt="feature_target" src="https://github.com/user-attachments/assets/1a3e8795-e7e9-477b-b5f2-1e3d6c5af1ec" />

# 분류 예측 프로세스

<img width="1920" height="1080" alt="classifier_flow" src="https://github.com/user-attachments/assets/ad9379fa-2ba5-4275-bdb8-d1e939dba84a" />

#### 데이터 세트 분리  

**train_test_split(feature, target, test_size, random_state)**

- 학습 데이터 세트와 테스트 데이터 세트를 분리해준다.
- feature: 전체 데이터 세트 중 feature
- target: 전체 데이터 세트 중 target
- test_size: 테스트 세트의 비율 (0 ~ 1)
- random_state: 매번 동일한 결과를 원할 때, 원하는 seed(기준점)를 작성한다.

#### 모델 학습
**fit(train_feature, train_target)**
- 모델을 학습시킬 때 사용한다.
- train_feature: 훈련 데이터 세트 중 feature
- train_target: 훈련 데이터 세트 중 target

#### 평가
**accuracy_score(y_test, predict(X_test))**
- 모델이 얼마나 잘 예측했는지를 "정확도"라는 평가 지표로 평가할 때 사용한다.
- y_test: 실제 정답
- predict(X_test): 예측한 정답

### 결정 트리 (Decision Tree)
- 매우 쉽고 유연하게 적용될 수 있는 알고리즘으로서 데이터의 스케일링, 정규화 등의 데이터 전처리의 의존도가 매우 적다.
- 학습을 통해 데이터에 있는 규칙을 자동으로 찾아내서 Tree기반의 분류 규칙을 만든다.
- 각 특성이 개별적으로 처리되어 데이터를 분할하는데 데이터 스케일의 영향을 받지 않으므로 결정트리에서는 정규화나 표준화같은 전처리 과정이 필요없다.
- 영향을 가장 많이 미치는 feature를 찾아낼 수도 있다.
- 예측 성능을 계속해서 향상시키면 복잡한 규칙 구조를 가지기 때문에 <sub>※</sub>과적합(Overfitting)이 발생해서 예측 성능이 저하될 수도 있다.
- 가장 상위 노드를 "루트 노드"라고 하며, 나머지 분기점을 "서브 노드", 결정된 분류값 노드를 "리프 노드"라고 한다.

<img width="641" height="281" alt="decision_tree" src="https://github.com/user-attachments/assets/89363918-eb26-4a89-9a9c-6bc7f009caed" />

- 복잡도를 감소시키는 것이 주목적이며, 정보의 복잡도를 불순도(Impurity)라고 한다.
- 이를 수치화한 값으로 지니 계수(Gini coeficient)가 있다.
- 클래스가 섞이지 않고 분류가 잘 되었다면, 불순도 낮다.
- 클래스가 많이 섞여 있고 분류가 잘 안되었다면, 불순도 높다.
- 통계적 분산 정도를 정량화하여 표현한 값이고, 0과 1사이의 값을 가진다.
- 지니 계수가 낮을 수록 분류가 잘 된 것이다.

---
<sub>※ 과적합이란, 학습 데이터를 과하게 학습시켜서 실제 데이터에서는 오차가 오히려 증가하는 현상이다. </sub>  

<img width="429" height="292" alt="overfitting" src="https://github.com/user-attachments/assets/e61bab7f-4d5f-47da-bf73-5aebd48ab350" />

#### Graphviz
- 결정트리 모델을 시각화할 수 있다.
- 설치 후 CMD창은 끄고 새롭게 켜기
- https://graphviz.org/download/  
  graphviz-9.0.0 (64-bit) EXE installer [sha256]
- https://drive.google.com/file/d/1oCXidIjNAvUT2UcNFEdhRfFhnZ96iHrp/view?usp=sharing
- pip install graphviz

### 베이즈 추론, 베이즈 정리, 베이즈 추정 (Bayesian Inference)
- 역확률(inverse probability) 문제를 해결하기 위한 방법으로서, 조건부 확률(P(B|A)))을 알고 있을 때, 정반대인 조건부 확률(P(A|B))을 구하는 방법이다.
- 추론 대상의 사전 확률과 추가적인 정보를 기반으로 해당 대상의 "사후 확률"을 추론하는 통계적 방법이다.
- 어떤 사건이 서로 "배반"하는(독립하는) 원인 둘에 의해 일어난다고 하면, 실제 사건이 일어났을 때 이 사건이 두 원인 중 하나일 확률을 구하는 방식이다.
- 어떤 상황에서 N개의 원인이 있을 때, 실제 사건이 발생하면 N개 중 한 가지 원인일 확률을 구하는 방법이다.
- 기존 사건들의 확률을 알 수 없을 때, 전혀 사용할 수 없는 방식이다.
- 하지만, 그 간 데이터가 쌓이면서, 기존 사건들의 확률을 대략적으로 뽑아낼 수 있게 되었다.
- 이로 인해, 사회적 통계나 주식에서 베이즈 정리 활용이 필수로 꼽히고 있다.  

> ##### 예시
질병 A의 양성판정 정확도가 80%인 검사기가 있다. 검사를 시행해서 양성이 나왔다면, 이 사람이 80%의 확률로 병에 걸렸다고 이야기할 수 없다. 왜냐하면 검사기가 알려주는 확률과 양성일 경우 질병을 앓고 있을 확률은 조건부 확률의 의미에서 정반대이기 때문이다.  
<table style="width:50%; margin-left: 50px">
    <tr>
        <th>전제</th>
        <th>관심 사건</th>
        <th>확률</th>
    </tr>
    <tr>
        <th>병을 앓고 있다</th>
        <th>양성이다</th>
        <th>80%</th>
    </tr>
    <tr>
        <th>양성이다</th>
        <th>병을 앓고 있다</th>
        <th>알수 없음</th>
    </tr>
</table>  

> 이런 식의 확률을 구해야 하는 문제를 역확률 문제라고 하고 이를 베이즈 추론을 활용하여 구할 수 있다.  
단, 검사 대상인 질병의 유병률(사전 확률, 기존 사건들의 확률)을 알고 있어야 한다.  
전세계 인구 중 10%의 사람들이 질병 A를 앓는다고 가정한다.

<div style="width: 60%; display:flex; margin-top: -20px; margin-left:30px">
    <div>
      <img width="300" alt="bayesian_inference01" src="https://github.com/user-attachments/assets/3481f1b0-7fed-4556-b0c0-2c5acc416b61" />
    </div>
    <div style="margin-top: 28px; margin-left: 20px">
      <img width="309" alt="bayesian_inference02" src="https://github.com/user-attachments/assets/a407facb-416b-472e-8f39-b75232a42647" />
    </div>
</div>  

<div style="width: 60%; display:flex; margin-left:30px">
    <div>
      <img width="475" alt="bayesian_inference03" src="https://github.com/user-attachments/assets/1dbd0879-7188-4dfe-a603-0f1272c77cb8" />
    </div>
    <div style="margin-top: 28px; margin-left: 20px">
      <img width="384" alt="bayesian_inference04" src="https://github.com/user-attachments/assets/8d2c7593-4d9a-4ecf-9fef-0e5c2571286f" />
    </div>
</div>  

> 🚩결과: 약 30.8%
<img width="284" height="42" alt="bayesian_inference05" src="https://github.com/user-attachments/assets/9c174064-7bbc-4240-979d-8c5046f7a548" />

### 나이브 베이즈 분류 (Naive Bayes Classifier)
- 텍스트 분류를 위해 전통적으로 사용되는 분류기로서, 분류에 있어서 준수한 성능을 보인다.
- 베이즈 정리에 기반한 통계적 분류 기법으로서, 정확성도 높고 대용량 데이터에 대한 속도도 빠르다.
- 반드시 모든 feature가 서로 독립적이여야 한다. 즉, 서로 영향을 미치지 않는 feature들로 구성되어야 한다.
- 감정 분석, 스팸 메일 필터링, 텍스트 분류, 추천 시스템 등 여러 서비스에서 활용되는 분류 기법이다.
- 빠르고 정확하고 간단한 분류 방법이지만, 실제 데이터에서  
  모든 feature가 독립적인 경우는 드물기 때문에 실생활에 적용하기 어려운 점이 있다.

<img width="468" height="308" alt="naive_bayes_classifier" src="https://github.com/user-attachments/assets/d3a1577e-a6d8-4e42-a276-bd3774fa3a75" />

#### 사전 확률 : 우리가 수집한 데이터 세트의 타겟 데이터의 비중(value_counts())

- 타켓 데이터의 비중을 확인하자

- 피쳐끼리의 상관관계가 있으면 안된다. (코 릴레이션)
	
- 역확률

- 문자열은 수집 x 숫자만 가능: 레이블 인코딩(종류 2가지), 워낫 인코딩(종류 여러가지)

- 빈도 수를 숫자로 나타내 전처리를 하자 (카운트 벡터라이저) 문장의 띄워쓰기 기준 각각의 단어 인덱스, 중복은 카운트

- 중복된 값 제거 : spam_df.drop_duplicates()

#### CountVectorizer
- 문장에 있는 단어들에 인덱스를 붙여서 각 단어의 빈도수를 세어주는 기술

<p align="right">(<a href="#readme-top">back to top</a>)</p>
