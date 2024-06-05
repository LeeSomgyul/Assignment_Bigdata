import numpy as np
import pandas as pd

data_df = pd.read_csv('./auto-mpg.csv', header=0, engine='python')

print('데이터셋 크기:' , data_df.shape)
data_df.head()

data_df = data_df.drop(['car_name', 'origin', 'horsepower'], axis = 1, inplace = False)
data_df.head()

print('데이터셋 크기:', data_df.shape)

data_df.info()