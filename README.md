# 10장) 회귀분석
### 🖥️사이킷런 다운로드
#### In [1]
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
###### boston에 대한 설명(DESCR)을 콘솔창에 보여준다.
<img width="437" alt="image" src="https://github.com/LeeSomgyul/Assignment_Bigdata/assets/140570847/0c897265-edc6-4bab-9437-83e252d03a79">

#### ✅ In [4]
    boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
    boston_df.head()
###### 1) pd.DataFrame()은 pandas라이브러리로 데이터를 데이터프레임 형식으로 변환해준다.
###### 2) boston.data은 보스턴 주택 가격 세부 데이터이고 boston.feature_names은 데이터의 열 이름이다.
###### 3) .head()은 기본적으로 5개 행을 출력한다. 즉 boston_df에 저장된 데이터 중 5개 행을 출력한다.

#### ✅ Out [4]
<img width="776" alt="image" src="https://github.com/LeeSomgyul/Assignment_Bigdata/assets/140570847/292051ba-ac32-4bd6-8a5b-c8717b189651">

#### ✅ In [5]
    boston_df['PRICE'] = boston.target
    boston_df.head()
###### boston.target은 주택가격이 포함된 배열로, boston_df에 PRICE열을 추가하여 주택가격을 담는다.

#### ✅ Out [5]
<img width="818" alt="image" src="https://github.com/LeeSomgyul/Assignment_Bigdata/assets/140570847/008b6cce-7dd6-40d1-9722-6e2b55dd1643">

    

