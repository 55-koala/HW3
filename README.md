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
  - 載入資料並轉為 DataFrame，加入品種對應名稱  
  - 使用 describe() 檢視數據統計摘要
  - 用圖表將資料視覺化
  - 使用邏輯迴歸進行分類並輸出準確率與分類報告

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
## 結果與分析
- 箱型圖：顯示不同品種下各特徵的分布

```python

for col in df.columns[:-1]:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='species', y=col, data=df)
    plt.title(f'{col} 不同品種的分布')
    plt.tight_layout()
    plt.show()

```

- 折線圖：每個品種的特徵平均值趨勢

```python

mean_features = df.groupby('species').mean()
plt.figure(figsize=(8, 5))
for species in mean_features.index:
    plt.plot(mean_features.columns, mean_features.loc[species], marker='o', label=species)

plt.title('各品種平均特徵值折線圖')
plt.xlabel('特徵')
plt.ylabel('平均值')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

```

- 長條圖：展示各特徵在不同品種的平均值

```python

mean_features.T.plot(kind='bar', figsize=(10, 6))
plt.title('不同品種特徵平均值長條圖')
plt.ylabel('平均值')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

```

- 建立模型並輸出分類報告

```python

X = df.iloc[:, :-1]
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n🔹 分類報告：")
print(classification_report(y_test, y_pred))

```
