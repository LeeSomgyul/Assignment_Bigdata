# 10장) 회귀분석
### 🖥️사이킷런 다운로드
    !pip install scikit-learn
    
### 🖥️데이터 수집, 준비 및 탐색
    import numpy as np
    import pandas as pd
    
    from sklearn.datasets import load_boston //사이킷런에서 보스턴 주택가격 가져오기
    boston = load_boston()
