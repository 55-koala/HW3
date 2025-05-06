# HW03 Data Analysis

## 標題 : Iris 花卉數據分析報告


## 摘要

本報告基於著名的 Iris 花卉資料集，透過探索性資料分析與視覺化，探討不同品種花卉在萼片與花瓣長度與寬度上的差異，藉此找出可區分品種的關鍵特徵，並以邏輯迴歸模型進行分類預測，進一步驗證資料的可分性與分類效果。分析結果可作為植物分類、機器學習教學範例，以及後續模型建構的基礎。

## 引言

- 背景 : 
Iris 資料集由英國統計學家 Ronald Fisher 所提出，是資料科學與機器學習領域中經典的分類問題範例。資料集中包含三種鳶尾花（Setosa、Versicolor、Virginica）共 150 筆樣本，每筆紀錄包含四個特徵（萼片長度與寬度、花瓣長度與寬度）以及其品種標籤。

- 目的 : 
本研究旨在透過資料分析方法，視覺化三種花卉的特徵分布差異，找出最具區別性的特徵組合，並建立分類模型以預測花卉品種。此過程可強化對資料探索與模型評估的理解。

- 待答問題 : 
  - 哪些特徵最能區分三種花卉？

  - 三種花卉在各特徵上的平均值有何差異？

  - 使用邏輯迴歸模型進行分類，其準確率為何？

## 方法

- 數據來源 : 
本研究使用 scikit-learn 套件內建的 Iris 資料集，包含 150 筆花卉樣本，共三種品種，各有 50 筆記錄。

- 分析工具 : 
  - Python：資料分析主要語言

  - Pandas：資料讀取與處理

  - Seaborn / Matplotlib：數據視覺化

  - Scikit-learn：模型建立與分類評估

- 數據處理與分析步驟 :
  - 載入資料並轉為 DataFrame，加入品種對應名稱。
  
  - 使用 describe() 檢視數據統計摘要。
  - 分別以：
   - 箱型圖（Box Plot）：檢視特徵在不同品種下的分布與異常值。

   - 折線圖（Line Plot）：比較三種花卉的平均特徵趨勢。

   - 長條圖（Bar Chart）：呈現各品種四個特徵的平均值。

  - 使用邏輯迴歸進行分類並輸出準確率與分類報告。

## 程式碼

```python
!pip install pandas seaborn matplotlib scikit-learn --quiet

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({i: name for i, name in enumerate(iris.target_names)})

print("前五筆資料：")
print(df.head())

print("\n資料摘要：")
print(df.describe())
```
