import pandas as pd
import matplotlib.pyplot as plt
import seaborn

df = pd.read_csv('data/total.csv')

print(df.columns.values)
arr = []
for col in df.columns.values:
    arr.append(sum(df[col]))
arr = [float(x) / sum(arr) for x in arr]

plt.figure(figsize=(12,8))
plt.barh(range(len(arr)), arr,color='rgb',tick_label=df.columns.values)
plt.show()


import pandas as pd
import matplotlib.pyplot as mp
import seaborn

df = pd.read_csv("data/total.csv")
print(df.head())
df_corr = df.corr()
seaborn.heatmap(df_corr, center=0, annot=True)
mp.show()
