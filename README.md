# 10장) 회귀분석
### 🖥️사이킷런 다운로드
#### ✅In [1]
    !pip install scikit-learn


### 🖥️데이터 수집, 준비 및 탐색
#### sklearn 1.2 버전부터 load_boston 미제공으로 코드를 변경하였다.

#### ✅ In [2]: 수정전
    import numpy as np
    import pandas as pd
    
    from sklearn.datasets import load_boston 
    boston = load_boston()
#### ✅ In [2]: 수정후
    import numpy as np
    import pandas as pd
    
    from sklearn.datasets import fetch_openml
    boston = fetch_openml(name='boston')

#### ✅ In [3]
    print(boston.DESCR)
boston에 대한 설명(DESCR)을 콘솔창에 보여준다.
###### <img width="437" alt="image" src="https://github.com/LeeSomgyul/Assignment_Bigdata/assets/140570847/0c897265-edc6-4bab-9437-83e252d03a79">

#### ✅ In [4]
    boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
    boston_df.head()
1) pd.DataFrame()은 pandas라이브러리로 데이터를 데이터프레임 형식으로 변환해준다.
2) boston.data은 보스턴 주택 가격 세부 데이터이고 boston.feature_names은 데이터의 열 이름이다.
3) .head()은 기본적으로 5개 행을 출력한다. 즉 boston_df에 저장된 데이터 중 5개 행을 출력한다.

#### ☑️ Out [4]
<img width="800" alt="image" src="https://github.com/LeeSomgyul/Assignment_Bigdata/assets/140570847/292051ba-ac32-4bd6-8a5b-c8717b189651">

#### ✅ In [5]
    boston_df['PRICE'] = boston.target
    boston_df.head()
boston.target은 주택가격이 포함된 배열로, boston_df에 PRICE열을 추가하여 주택가격을 담는다.

#### ☑️ Out [5]
<img width="900" alt="image" src="https://github.com/LeeSomgyul/Assignment_Bigdata/assets/140570847/008b6cce-7dd6-40d1-9722-6e2b55dd1643">

#### ✅ In [6]
    print('보스톤 주택 가격 데이터셋 크기:', boston_df.shape)
.shape는 boston_df의 행,열의 개수를 알려준다.

#### ☑️ Out [6]
행의 개수는 506개, 열의 개수는 14개이다.
14개의 열 중에서 13개(CRIM ~ LSTAT는 독립변수 X이고, PRICE는 X에 영향을 받아 결정되기 때문에 종속변수 Y가 된다.)
###### <img width="400" alt="image" src="https://github.com/LeeSomgyul/Assignment_Bigdata/assets/140570847/63e4b0cf-3aec-4dba-aa03-e4f45ee66be5">


#### ✅ In [7]
    boston_df.info()
boston_df의 정보를 확인한다.

#### ☑️ Out [7]
열 이름 / 506개 행에서 비어 있지 않은 값의 개수 / 데이터 유형 순으로 표기된다.
###### <img width="300" alt="image" src="https://github.com/LeeSomgyul/Assignment_Bigdata/assets/140570847/b311b983-bec7-42a0-a444-9a1664960703">


### 🖥️분석 모델 구축, 결과 분석 및 시각화
#### ✅ In [8]
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
1) LinearRegression: 머닝러신 회귀분석을 위한 함수(회귀 모델 구현 기능)
2) train_test_split: 데이터셋 분리작업을 위한 함수(데이터를 학습 세트와 테스트 세트로 나누는 기능)
3) mean_squared_error: 성능 측정을 위한 함수(예측값과 실제값 사이의 제곱 오차를 평균하여 모델의 예측 성능을 평가, 평가지표 중 MSE로 실제값과 이상값 사이의 차이가 크다)
4) r2_score: 모델의 적합성 평가를 위한 함수

#### ✅ In [9]
    Y = boston_df['PRICE']
    X = boston_df.drop(['PRICE'], axis=1, inplace=False)
X를 독립변수, Y를 종속변수로 설정

#### ✅ In [10]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=156)
train_test_split()함수를 통해 데이터셋을 훈련용 데이터 7, 테스트 데이터 3 비율로 나눈다.
1) X_train: 훈련 입력 데이터(주택 특징에 대한 데이터)
2) Y_train: 훈련 타겟 데이터(특징으로 인한 해당 주택의 실제 판매 가격)
3) X_test: 테스트 입력 데이터(새로운 주택 특징에 대한 데이터로 훈련 데이터로 학습된 모델에게 새로운 데이터를 적용한다)
4) Y_test: 테스트 타겟 데이터(예측값(Y_predict)과 비교하여 모델 성능 확인)

#### ✅ In [11]
    lr = LinearRegression()
선형 회귀 분석 모델로 lr 생성

#### ✅ In [12]
    lr.fit(X_train, Y_train)
.fit함수로 훈련용 데이터로 모델 학습

#### ☑️ Out [12]
<img width="200" alt="image" src="https://github.com/LeeSomgyul/Assignment_Bigdata/assets/140570847/a02ad707-07c1-4e13-8b82-3806a43c27d3">

#### ✅ In [13]
    Y_predict = lr.predict(X_test)
.predict()함수를 사용하여 주택의 특징(X_test)을 입력받아 Y_predict에 예측 가격을 저장한다.
###### 💡Y_test가 아닌 X_test를 사용하는 이유: Y_test는 모델 평가를 위해 사용되는 데이터로 나중에 Y_predict와 비교하여 실제값과 예측값을 비교한는데 사용될 예정이다.

#### ✅ In [14]
    mse = mean_squared_error(Y_test, Y_predict)
    rmse = np.sqrt(mse)
    
    print('MSE : {0:.3f}, RMSE : {1:.3f}'.format(mse, rmse))
    print('R^2(Variance score) : {0:.3f}'.format(r2_score(Y_test, Y_predict)))
1) mean_squared_error함수는 실제값(Y_test)과 예측값(Y_predict)의 평균 제곱 오차를 출력한다.
   (출력값이 작을수록 모델의 예측이 실제 값에 가깝다는 것을 의미)
2) np.sqrt는 제곱근을 계산하는 함수로 MSE에 루트를 씌워 이상값을 줄이는 RMSE값으로 출력한다.
3) MSE와 RMSE값을 소수점 셋째자리까지 출력
4) r2_score()함수로 모델이 얼마나 잘 예측하는지 성능 파악. (출력값이 1에 가까울수록 모델이 데이터를 잘 설명한다는 의미)
###### 💡X_test와 Y_test의 타입이 맞지 않아 <img width="400" alt="image" src="https://github.com/LeeSomgyul/Assignment_Bigdata/assets/140570847/f57c6775-a517-4321-82df-0318c7470284">오류가 발생함으로 In [10] 코드 하단에 아래 코드를 추가한다.
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

#### ☑️ Out [14]
<img width="300" alt="image" src="https://github.com/LeeSomgyul/Assignment_Bigdata/assets/140570847/3ed78f60-5afb-4618-90ef-7bfa6e1a3b36">
    

