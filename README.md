

![미니프로젝트_2조](https://github.com/AhnHz/SQL-study/assets/132975657/9809bbf3-464a-496f-a25b-5ac0af97027c)

<br/>

## :family: 팀원
- 안현주
- 이경주
- 이성철
- 장수현
- 홍량택
- 정승균

<br/>

## :calendar: 프로젝트 기간
2023.06.12 ~ 2023.06.23

<br/>

## :page_with_curl: 프로젝트 개요
- 원본 데이터 : [MBTI Personality Types 500 Dataset](https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset)
- 자연어 처리(NLP)  +  딥러닝
- Bert model 과 torch 를 이용함.
- 텍스트를 통해 작성자 MBTI 예측할 수 있다.
- 다른 게시물 작성자의 MBTI 분포를 확인할 수 있다.
- 전처리한 데이터와 학습시킨 모델 : [Project Data(BERT ModelData, CSV Data)](https://drive.google.com/drive/folders/1RpiNPAqSp0TLooxiZDREexODc94jlhLF?usp=drive_link)

<br/>

## :running: 프로젝트 진행 과정
### 1일차 ( 4시간 )

- **조장 선정**
- **머신러닝에 필요한 데이터 개념 정의**
- **데이터 찾기**
    
    [ 데이터 후보 ] 
    
     - 폐음 데이터 세트(질병예측)
     - 신용 카드 기본값(채무 불이행 사례를 통한 채무불이행 예측 )
     -  주식 예측
     - 은행 고객 이탈(이탈 방지로 인한 충성도 프로그램, 유지 캠페인 개발
     - 사용 어휘 및 리뷰 길이에 따른 평점 재정의
     - 도시별 인구 감소 비율에 따른 사라질 도시 예측
     - 애완동물 행동 분석 데이터를 이용
     - 글자 인식
     - 강아지 표정 분석
     - 객체 인식 (사진 및 그림)

<br/>

### 2일차 ( 4시간 )

- **후보 데이터 추리기**
    
    [ 플랜 A ]
     - 사용 어휘 및 리뷰 길이에 따른 평점 재정의
     - OCR 글자 인식
     - 객체 인식 ( 사진 및 그림)
    
    [ 플랜 B ]
     - 신용 카드 기본값 ( 채무 불이행 예측)
     - 주식 예측

<br/>

- **데이터 선정  : 사용 어휘 및 리뷰 길이데 따른 평점 재정의 (리뷰데이터)**
- **진행 방향 구체화**
    1. 세부 주제 뽑기
    - 리뷰 길이와 어휘에 따른 평점 재정의
    - 게시글과 무관한 리뷰 필터링 ( 다른 상품 리뷰 OR 관련 없는 댓글)
    - 문서 분류 ( MBTI별 트위터 사용 태그 분석 )
    - 뉴스 리뷰 또는 도서 리뷰를 활용한 주기별 트랜드 분석
    - 트위터 등의 SNS를 통한 시대별 어휘 변화 분석
    - 쇼셜미디어 게시물 기반 감정 분석 ( 우울증 )
    - 키워드 추출로 댓글 분석 
    2. 세부 주제 추리기
    - 문서 분류 ( MBTI별 트위터 사용 태그 분석)
    - 뉴스 리뷰 또는 도서 리뷰를 활용한 주기별 트랜드 분석
    - 소셜미디어 게시물 기반 감정 분석 (우울증)
    3. 세부 주제 선정 : MBTI 예측
    4.  사용 언어 선택 - 영어 or 한글 
    5. 참고 사이트 
    [[Brightics Studio] # 팀 분석 프로젝트 - 01 기획 및 구상, 텍스트 데이터로 MBTI 예측하기 - 데이터 사이언스 사용 설명서 (tistory.com)](https://dsbook.tistory.com/362)

<br/>

- **최종적으로 선택사항 선정**  
선택1. 리뷰 및 게시글 등 한국어 데이터 셋 찾기  
선택2. 영어 데이터 셋으로 진행하기  
선택3. 다른 주제로 변경

<br/>

### 3일차 ( 4시간 )

- **추가 의견 제시**
    - 청와대 국민 청원 데이터 셋 사용하기
 
<br/>

- **최종 선택**
    
    영어 데이터 셋을 활용한 MBTI

<br/>

- **최종 프로젝트 주제 : *영어 데이터 셋을 활용한 MBTI별 텍스트 분석 및 분류***
- **활용 데이터 셋 분석 및 해결 방안 찾기**
    
    [MBTI Personality Types 500 Dataset | Kaggle](https://www.kaggle.com/datasets/zeyadkhalid/mbti-personality-types-500-dataset)

   - 초기 데이터    
![Untitled](https://github.com/AhnHz/Algorithm-study/assets/132975657/c729b016-817f-489f-93ca-84467ba84dfd)  


    - 데이터 상세정보
![Untitled 1](https://github.com/AhnHz/Algorithm-study/assets/132975657/50c8bccb-2a08-44a2-a05d-71326e96548f)  


    - 초기 데이터 시각화
![mbti_count](https://github.com/AhnHz/Algorithm-study/assets/132975657/97cc6eb3-b494-4ae4-a635-fc7e4091e306)  

    
    <aside>
    ⚠️ 문제점 : 각 mbti 마다 데이터의 수의 불균형
    
    해결방안 : mbti의 알파벳 속성별로 데이터 분할<br/>
    
    </aside>
    
    - MBTI 구분자 추가
![sc1](https://github.com/AhnHz/Algorithm-study/assets/132975657/1c6687d3-d949-4291-b8b7-0f17a20c02eb)
    
    :white_check_mark: **여전히 불균형 문제가 있으나 비율의 폭이 감소하여 적은 데이터에 대한 학습 데이터 증가**

<br/>

- **진행 계획 및 담당 지정**
    - MBTI 16가지 분류 전처리 : 전 인원
    <br/>
    
    - MBTI 16가지 종류 나누어 담당   
    I : 수현  
    E : 경주  
    S : 성철  
    N : 현주  
    T : 량택  
    F : 승균  
    P : 현주  
    J : 성철
    <br/>
   
    - 타입 별 뽑아내는 작업 실행_당일
    알파벳 별로 분류 후 타입의 그룹별 데이터 분할
    - BERT, 텍스트 전처리
    - 자연어 처리 , 단어 임베딩

<br/>

- **자연어 처리 참고 사이트** 
[[BERT] BERT에 대해 쉽게 알아보기1 - BERT는 무엇인가, 동작 구조 :: 삶은 확률의 구름 (tistory.com)](https://ebbnflow.tistory.com/151)
- 불균형 현상 해결방안 참고 사이트
[MBTI 500 - 84% 정확도 | 캐글 (kaggle.com)](https://www.kaggle.com/code/clebermarques/mbti-500-84-accuracy/notebook)

<br/>

- **이후 진행 계획**
    1. 전처리 필요여부 확인하기 
    - accented character 제거
    - 축약어 원래 형태로 변환
    - 모든 단어 원영화 ( lemma )
    - 불용어 제거 **nltk 참고
    2. WordCloud를 통한 시각화
    3. s에 맞춰 다운 샘플링 (약 9천개)
    ** 게시글 중복 제거, 데이터 선정 기준 정하기
    4. 임베딩 진행 방향 논의

<br/>

### 4일차 ( 8시간 )

1. **요약/축약어 풀기**
참고 사이트 : [영어 - 채팅, SNS 줄임말(약어, 용어) 모음 : 네이버 블로그 (naver.com)](https://m.blog.naver.com/park-hye/221492311971)
2. **불용어 선정 :**
참고 사이트 : [02-04 불용어(Stopword) - 딥 러닝을 이용한 자연어 처리 입문 (wikidocs.net)](https://wikidocs.net/22530)
3. **불용어 선정 기준 :**
원소별 빈도수가 높은 20위 선정하여 ie, ns, tf, jp 끼리 묶어 동일하게 나온 단어, 영어 불용어
4. **불용어 처리 코드**
  ```python    
  from nltk.corpus import stopwords
  from nltk.tokenize import word_tokenize
  
  def stopword_process(text, wordslst : list):
      
      tokens = word_tokenize(text)
      text = text.lower()
  
      stop_words = stopwords.words('english')
      stop_words.extend(["mbti", "infj", "intj", "infp", "intp", "enfj", "entj", "enfp", "entp",
                          "isfj", "istj", "isfp", "istp", "esfj", "estj", "esfp", "estp", "personality", "type"])
      stop_words.extend(['like', 'think', 'people', 'get', 'thing', 'make', 'know', 'feel', 'one', 'go'])
      stop_words.extend(wordslst)
  
      tokens = [word for word in tokens if word not in stop_words]
  
      changepost = ' '.join(s for s in tokens)    
      
      return changepost
  
  stop_word_df = pd.DataFrame({'posts' : [], 'type' : [], 'type2' : []})
  
  def duplication_words_remove(MBTI, wordslst):
      global stop_word_df
      temp_totaldf = totaldf[totaldf['type2'].str.contains(MBTI[0]) | totaldf['type2'].str.contains(MBTI[1])]
  
      temp_totaldf['posts'] = temp_totaldf['posts'].apply(stopword_process, args = (wordslst, ))
  
      stop_word_df = pd.concat([stop_word_df, temp_totaldf])
  
  
  MBTIlst = [['I', 'E'], 
             ['S', 'N'],
             ['T', 'F'],
             ['J', 'P']]
  
  remove_words = [['feel', 'time', 'say', 'good', 'would', 'really', 'want', 'way', 'see'],
                   ['know', 'time', 'really', 'good', 'feel', 'want', 'see', 'way', 'also'],
                   ['know', 'make', 'really', 'time', 'say', 'good', 'would', 'want', 'way', 'see'],
                  ['say', 'time', 'good', 'would', 'really', 'want', 'way', 'see', 'also']]
  
  
  for i in range(4):
      duplication_words_remove(MBTIlst[i], remove_words[i])
  ```
    
5. **이후 진행 계획** 
- 축약어 정리 및 처리 논의
- 모델 학습  
- 참고사이트 : [[Pytorch][BERT] 버트 소스코드 이해_④ BertModel (tistory.com)](https://hyen4110.tistory.com/90)

<br/>

### 5일차 ( 8시간 )

- **BERT 임베딩**
    
    - **BERT 란?**
    
    <aside>
    💡 BERT(Bidirectional Encoder Representations from Transformers)는 다양한 자연어 처리 분야에서 가장 좋은 성능을 내면서 여러 가지 일들을 수행하는데 사용되고 있습니다.
    BERT란 문맥을 양방향으로 이해해서 숫자형태로 바꿔주는 딥러닝 모델이다.
    
    </aside>
    
    
    - **BERT 특징**
    
    > [ 전이 학습 모델 ] 
    사전 학습된 대용량의 레이블링 되지 않는 데이터를 이용하여 언어 모델을 학습하고 이를 토대로 특정 작업( 문서분로, 질의 응답, 번역 등)을 위한 신경망을 추가하는 학습 방법
    > 
    
    > [ 사전 학습 모델 ]
    ****대용량의 데이터를 직접 학습시키기 위해서는 매우 많은 지원과 시간이 필요하지만 BERT모델은 기본적으로 대량의 단어 임베딩 등에 대해 사전 학습이 되어있는 모델을 제공하기 때문에 상대적으로 적은 지원만으로도 충분히 자연어 처리의 여러 일을 수행할 수 있다.
    > 
    - **BERT 구조**
    
    ![Untitled 6](https://github.com/AhnHz/Algorithm-study/assets/132975657/3220247f-f978-4b5d-96ab-9c8f87382e80)
    
   **그림과 같이 세 가지 임베딩 값의 합으로 구성되는 것이 특징**
    
    - **Token Embeddings**
    
    토큰 임베딩은 Word piece임베딩 방식을 사용한다.
    Word Piece임베딩은 자주 등장하면서 가장 긴 길이의 sub-word을 하나의 단위로 만든다.
    즉, 자주 등장하는 단어는 그 자체가 단위가 되고, 자주 등장하지 않는 단어는 더 작은 sub-word로 쪼개어진다. 
    이는 자주 등장하지 않은 단어를 전부out-of-vocabulary로 처리하여 모델링의 성능을 저하시키는 문제를 해결한다.
    입력받은 모든 문장의 시작으로 [CLS] 토큰(special classification token)이 주어지며 이 [CLS] 토큰은 모델의 전체 계층을 다 거친 후 토큰 시퀀스의 결합된 의미를 가지게 된다.
    
    - **Segmint Embeddings**
    
    토큰으로 나누어진 단어들을 다시 하나의 문장으로 만들고 
    첫 번째 [SEP] 토큰까지 0으로 그 이후 [SEP]토큰까지는 1값으로 마스크를 만들어 각 문장들을 구분한다.
    
    - **Position Embeddings**
    
    토큰의 순서를 인코딩한다. 
    그 이유는 BERT는 transformer의 encoder를 사용하는데 Transformer는 self-attenrion모델을 사용한다. 
    Self-Attention은 입력 위치에 대해 고려하지 ㅇ못하므로 입력 토큰의 위치 정보를 주어야한다. 
    그래서 Transformer에서는 Sigsoid함수를 이용한 Positional 인코딩을 사용하였고, BERT에서는 이를 변형하여 Position Encoding을 사용한다.
    각 임베딩들의 토큰 별로 모두 더하여 BERT의 입력 벡터로 사용한다.
    
    - **BERT 모델 절차**
    1. 데이터 전처리 : 
    사용할 자연어 처리 작업에 맞게 데이터를 전처리합니다. 이 단계에는 토큰화(tokenization), 패딩(padding), 문장 분리(segmentation) 등이 포함될 수 있다.
    2. BERT 모델 불러오기 : 
    BERT 모델을 불러옵니다. 일반적으로 사전 학습된 BERT 모델의 가중치를 다운로드하고, 해당 모델의 구조를 정의하여 사용한다.
    3. Fine-tuning 레이어 추가 : 
    BERT 모델 뒷부분에 작업에 맞는 추가적인 레이어를 추가한다. 
    예를 들어, 텍스트 분류 작업을 수행한다면, 추가적인 분류 레이어를 모델에 추가할 수 있다.
    4. 학습 : 
    Fine-tuning 데이터셋을 사용하여 모델을 학습시킨다. 사전 학습된 가중치를 초기값으로 사용하면서, 추가된 레이어와 함께 모델의 파라미터를 업데이트한다.
    
    <aside>


    💡 Fine-tuning   
    BERT 모델을 구체적인 자연어 처리 작업에 맞게 세부 조정한다. 
    예를 들어, 텍스트 분류, 질의 응답, 개체명 인식 등의 작업에 BERT모델을 적용할 수 있다. 
    Fine-tunning 단계에서는 작업에 맞는 추가적인 레이어를 코델에 추가하고 사전 학습된 가중치를 초기 값으로 사용하여 모델을 학습시켜 최적화된 표현을 얻을 수 있다.
    
    </aside>
    
    - **BERT기반 임베딩** 
    참고 사이트 : [18-08 BERT의 문장 임베딩(SBERT)을 이용한 한국어 챗봇 - 딥 러닝을 이용한 자연어 처리 입문 (wikidocs.net)](https://wikidocs.net/154530)
    - BERT 코드 짜기
        - 키워드 추출
        
        ```python
        !pip install sentence_transformers
        # 키버트 실습을 위해서 우선 sbrt를 위한 패키지를 설치해야한.
        ```
        
    
    1. 텍스트를 모델에 넣는다.
    
    1. 모델 공부
    2. 가장 작은 s모델의 9천개에 기준을 맞추어 진행
    3. 파이토치      
    
    
<br/>

1. **BERT 임베딩**
- 분류 모델 , 예측 모델 
- 다중 분류 
- 2진 분류 모델 4번  / 한번에 MBTI가 나오는지 예측 방식 선정
- 파이토치 활용 / 텐서플로우 활용 / 케라스 활용
- 파인튜닝 작업 처리 방안 논의
2. 전처리된 텍스트 토큰화 작업
3. 학습 모델 활용 방안 논의
4. 주말에 스스로 학습 진행 
- 파이토치로 BERT진행
- 안되는 경우 다른 모델 찾아보기 ( RNL,순환신경망 등 활용 해보기 )
5. - 참고자료 :
[WONA_IN_IT (tistory.com)](https://wonhwa.tistory.com/35)
[https://youtube.com/playlist?list=PL5o6DaWWFfX_dwRuNT3XIkHqBFHYuUIt7](https://youtube.com/playlist?list=PL5o6DaWWFfX_dwRuNT3XIkHqBFHYuUIt7)
[https://www.kaggle.com/code/clebermarques/mbti-500-84-accuracy](https://www.kaggle.com/code/clebermarques/mbti-500-84-accuracy)
[https://colab.research.google.com/github/mrm8488/shared_colab_notebooks/blob/master/Try_mrm8488_xquad_finetuned_model.ipynb](https://colab.research.google.com/github/mrm8488/shared_colab_notebooks/blob/master/Try_mrm8488_xquad_finetuned_model.ipynb)

<br/>

### 6일차 ( 8시간 )

- BERT 사용하여 데이터셋 훈련 
- tokenizer = BertTokenizer.from_pretrained('bert-base-cased') ⇒ BertModel.from_pretrained('bert-large-uncased') 코드 수정  
**`정확도 약 77%`**
    
- 코드해석

```python
import torch
import numpy as np
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
# 버트 모델의 토크나이저를 'bert-large-uncased' 사전에 훈련된 모델로부터 가져옴

labels = {"INFJ" : 0, "INTJ" : 1, "INFP" : 2, "INTP" : 3, "ENFJ" : 4, "ENTJ" : 5, 
              "ENFP" : 6, "ENTP" : 7, "ISFJ" : 8, "ISTJ" : 9, "ISFP" : 10, "ISTP" : 11,
                "ESFJ" : 12, "ESTJ" : 13, "ESFP" : 14, "ESTP" : 15}
# 클래스 레이블과 인덱스를 매핑한 딕셔너리

class Dataset(torch.utils.data.Dataset):  
# dataset클래스를 상속하여 새로운 dataset클래스를 정의
    
    def __init__(self, df):
# 데이터 프레임을 입력 받아 dataset 객체를 초기화

        self.labels = [labels[label] for label in df['type']]
# 데이터프레임의 type열에 해당하는 클래스 레이블을 labels딕셔너리를 사용하여 숫자로 변환하여 저장
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['posts']]
# 데이터프레임의 'posts' 열에 해당하는 텍스트를 BERT 토크나이저를 사용하여 토큰화하고, 
# 패딩과 트러케이션을 적용하여 최대 길이 512로 맞춘 뒤, 텐서 형태로 변환하여 저장합니다.
    
    def classes(self):
        return self.labels
#클래스 레이블을 반환하는 메서드입니다.
    def __len__(self):
        return len(self.labels)
#데이터셋의 샘플 개수를 반환하는 메서드입니다.
    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])
#인덱스 idx에 해당하는 레이블 배치를 가져오는 메서드입니다.
    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]
#인덱스 idx에 해당하는 텍스트 배치를 가져오는 메서드입니다.
    def __getitem__(self, idx):
#인덱스 idx에 해당하는 텍스트 배치와 레이블을 반환하는 메서드입니다.
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
```

<br/>

### 7일차 ( 8시간 )

- 데이이터가 부족한 부분 검증을 위한 데이터 취합
    
    검증데이터 자료 = fail_5Mbti.csv
    
- ppt 작성을 위한 데이터 시각화 작업
- 한글 데이터 번역하여 검증하는 작업 시행
- ppt 자료 구상
- ppt 템플릿 선정

<br/>

### 8일차 ( 8시간 )

- PPT작업
- 데이터 시각화 작업

<br/>

### 9일차 ( 8시간 )

- PPT작업
- 발표 대본 작성
- ppt 설명
    1. 메인
    2. 목차
    3. mbti별 텍스트 분류를 선정한 이유 3가지 
    4. 수집한 데이터에 대한 설명 - 컬럼이 두개였고 type는 16개의 클래스 가짐, posts는 한 행의 글자길이는 3천개정도 있는데 토큰화하여 단어로 확인하면 500개로 줄어든다.
    5. 원래 우리가 하려고 했던 전처리가 방안 설명을하지만 사용하지는 않았다. 
    6. 불용어 처리 또한 진행했지만 사용하지는 않았다.
    7. 선정된 데이터의 시각화 (클래스별 개수의 불균형이 심했음)
    8. 불균형에 대한 해결방안 **알파벳별로 진행하지 않은 이유는 2진분류는 정확도가 오르겠지만 16개의 클래스에 대해서는 신뢰도가 떨어진다.
    9. 버트에 대한 설명
    10. 버트에 대한 자세한설명
    11. 파인튜닝
    12. 데이터 불러오기 _ 그룹별 정렬
    13. ~ 22코드설명
    
     23.  ~27모델 분석 & 활용

<br/>


## :bar_chart: 프로젝트 내용

### BERT(Bidirectional Encoder Representations from Transformers) 모델이란?
![미니프로젝트_2조_1](https://github.com/AhnHz/Algorithm-study/assets/132975657/76271296-24b7-4a8f-9d00-4ef82f8cfb9e)

![미니프로젝트_2조_2](https://github.com/AhnHz/Algorithm-study/assets/132975657/8eb31233-c310-4d33-853e-2e40c67559b8)

![미니프로젝트_2조_3](https://github.com/AhnHz/Algorithm-study/assets/132975657/e766d589-4fb7-4e25-8bab-5315afce99c9)

<br/>

### 모델 학습 결과
![미니프로젝트_2조_4](https://github.com/AhnHz/Algorithm-study/assets/132975657/f0d32816-2ee9-4f4b-96c5-a6c2eda15030)
- 훈련 정확도는 어느 정도 높게 나왔지만 검증 정확도는 에포크 5부터 약 0.79에 머물렀고 테스트 정확도는 0.8 정도로 나왔다.  


<br/>

### 모델 테스트 결과
#### MBTI 유형 별 정확도
![미니프로젝트_2조_5](https://github.com/AhnHz/Algorithm-study/assets/132975657/81df1758-fd4f-4895-8b80-5af75bd3df7a)

#### 잘못 예측된 MBTI 유형 분석
![미니프로젝트_2조_7](https://github.com/AhnHz/Algorithm-study/assets/132975657/511633d8-982e-450c-b466-ac0da3f01616)

![미니프로젝트_2조_8](https://github.com/AhnHz/Algorithm-study/assets/132975657/241608e7-9182-47ca-8d6f-623aa3eb566f)
- 잘못 예측된 MBTI 유형은 단순히 예측에 실패한 것이 아니라 특정 유형끼리 연관성을 보인다는 것을 알 수 있었다.
- 이를 분석해 `연관 패턴` 과 `일대일 패턴` 을 정의했다.

<br/>

### 테스트에 실패한 유형 및 실패 원인 분석
![미니프로젝트_2조_6](https://github.com/AhnHz/Algorithm-study/assets/132975657/2d195359-8631-440e-a15d-24adaca39aca)

<br/>

### 모델의 활용
![미니프로젝트_2조_9](https://github.com/AhnHz/Algorithm-study/assets/132975657/8144f33b-e19c-4c30-a952-c57c7f4c0714)

![미니프로젝트_2조_10](https://github.com/AhnHz/Algorithm-study/assets/132975657/9ba91fc3-e3a5-4a8d-be4c-8b6bb5c1a7ee)

<br/>

## :scissors: Trouble Shooting
### 데이터 불균형 문제
![미니프로젝트_2조_11](https://github.com/AhnHz/Algorithm-study/assets/132975657/50b62ca8-586f-4389-8d48-0815cf7a0387)

- 처음엔 MBTI를 16개의 유형으로 다중 분류하는 것이 아닌 알파벳 자릿수 별로 **4번의 이진 분류**하는 방법을 고안했다.  
  :warning: 중복 데이터가 늘어나고 정확성이 떨어지는 문제로 기각

- 모델 학습 시간과 유형 별 불균형을 완화하기 위해 과다한 유형 데이터를 **1000개로 다운샘플링**했다.

- 1000개 미만의 데이터를 가진 유형은 기존의 데이터를 이용한 **중복 데이터를 추가**했다.

<br/>

## :sweat: 아쉬운 점
- 안현주 : 
- 이성철 : 
- 이경주 :           
- 장수현 : 
- 정승균 : 
- 홍량택 : 
