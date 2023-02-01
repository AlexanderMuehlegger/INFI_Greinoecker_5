import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm

df = pd.read_excel('data/bev_meld.xlsx')


sum_data = []
for x in df.columns[3:]:
    sum_data.append(df[x].sum())


plt.scatter(range(int(df.columns[3]), int(df.columns[-1])+1), sum_data)

df_reg = pd.DataFrame({"years": range(int(df.columns[3]), int(df.columns[-1])+1), "popularity": sum_data})
df_reg = df_reg.astype({'years': 'int'})

df_pred = pd.DataFrame({'years': range(1980, 2100)})


def getPred(df, df_pred):
    model = sm.OLS.from_formula('popularity ~ years', df).fit()
    predictions = model.predict(df_pred)
    return predictions

def getModel(y, v):
    df_reg=pd.DataFrame({"years": y, "bev":v})
    model =  sm.OLS.from_formula('bev ~ years', df_reg).fit()
    return model

predictions = getPred(df_reg, df_pred) 
prediction = getPred(df_reg, pd.DataFrame({'years': [2030]}))

plt.plot(df_pred.years, predictions)
plt.plot([2030], [prediction], marker="*", markersize="10", color="red")
plt.show()

df_gem = df.loc[df['Gemeinde'] == 'Kramsach']

gem_sum_data = []

for x in df_gem.columns[3:]:
    gem_sum_data.append(df_gem[x].sum())
    
df_gem_reg = pd.DataFrame({'years': range(int(df.columns[3]), int(df.columns[-1])+1), 'popularity': gem_sum_data})

df_gem_reg = df_gem_reg.astype({'years': 'int'})
predictions_gem = getPred(df_gem_reg, df_pred)

plt.plot(df_gem_reg.years, gem_sum_data)
plt.plot(df_pred.years, predictions_gem)
plt.show()
# print(df_gem)

fig, axes = plt.subplots(2,1)
t1 = df.values[1:24, 3:].sum(axis=0)
d1 = pd.DataFrame({"Bevölkerungszahl":t1})
t2 = df.values[110:139, 3:].sum(axis=0)
d2 = pd.DataFrame({"Bevölkerungszahl":t2})


d1.plot(ax=axes[0], title="Bezirk Innsbruck")
d2.plot(ax=axes[1], title="Bezirk Kufstein")

fig.tight_layout()
plt.show()

plt.plot(df.columns[3:], d1, label="I")
plt.plot(df.columns[3:], d2, label="KU")
plt.legend()
plt.show()

plt.plot(df.columns[3:], getModel(pd.to_numeric(df.columns[3:]), pd.to_numeric(df.values[1:24,3:].sum(axis=0))).predict(pd.DataFrame({"years":np.arange(1993,2022)})), label="IM")
plt.plot(df.columns[3:], getModel(pd.to_numeric(df.columns[3:]), pd.to_numeric(df.values[110:139,3:].sum(axis=0))).predict(pd.DataFrame({"years":np.arange(1993,2022)})), label="KU")
plt.legend()
plt.show()