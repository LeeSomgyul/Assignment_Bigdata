import numpy as np
import pandas as pd


#1. 데이터 수집, 준비 및 탐색
from sklearn.datasets import fetch_openml

boston = fetch_openml(name='boston')

print(boston.DESCR)

boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_df.head()

boston_df['PRICE'] = boston.target
boston_df.head()

print('보스톤 주택 가격 데이터셋 크기:', boston_df.shape)

boston_df.info()

#2. 분석 모델 구축, 결과 분석 및 시각화
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

Y = boston_df['PRICE']
X = boston_df.drop(['PRICE'], axis=1, inplace=False)