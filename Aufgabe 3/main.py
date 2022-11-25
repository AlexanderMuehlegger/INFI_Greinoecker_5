import pandas as pd
from tabulate import tabulate
from matplotlib import pyplot as plt
import seaborn as sns

data = pd.read_excel('./data/Zeitreihe-Winter-2022092012.xlsx')

data.columns = ["x" + str(x) if str(x).isdigit() else x for x in data.columns]
#print(data.describe()) # gibt meadian, quantile, maximum, minimum, ... aus

# print(tabulate(data, headers=data.columns))


""" TODO: 1
data = pd.read_excel('./data/Zeitreihe-Winter-2022092012.xlsx')

data.columns = ["x" + str(x) if str(x).isdigit() else x for x in data.columns]
#print(data.describe()) # gibt meadian, quantile, maximum, minimum, ... aus

print(tabulate(data, headers=data.columns))

"""

# print(data.values[0, 3:])

# print(data)
# plt.scatter(range(2000, 2023), data.values[0, 3:], marker="*")
# # plt.show()

# my_bez = data.loc[data['Bez'] == "KU"]
# print(my_bez.sum())

# plt.scatter(range(2000, 2023), my_bez.values[0, 3:])

#3.1
min_data = []
max_data = []
avg_data = []

for x in range(0, len(data)):
    values = data.values[x, 3:]
    min_data.append(values.min())
    max_data.append(values.max())
    avg_data.append(round(values.mean(), 2))

data['min'] = min_data
data['max'] = max_data
data['avg'] = avg_data


# 3.2
sum_data = 0
for x in range(2000, 2023):
    sum_data += data[f'x{x}'].sum()

print(sum_data)

all_bez = list(set(data['Bez']))
sum_bez = {}

for x in all_bez:
    sum_bez[x] = data.loc[data['Bez'] == x].values[:, 3:-3].sum()

print(sum_bez)
plt.bar(range(len(sum_bez)), list(sum_bez.values()), align="center")
plt.xticks(range(len(sum_bez)), list(sum_bez.keys()))

#4.1
data.boxplot(column="avg", by="Bez")

#4.2
sns.barplot(x=data['Bez'], y=data['avg'], data=data)

# plt.show()

#5 
data2 = pd.read_excel('./data/bev_meld.xlsx')

print(data2)

data2.columns = ["y" + str(x) if str(x).isdigit() else x for x in data2.columns]
merged_data = pd.merge(data, data2, how='inner', on='Gemnr')

merged_data = merged_data.drop(columns="Gemnr")

merged_data['std_2018'] = merged_data["y2018"] / merged_data["x2018"]
merged_data['std_2020'] = merged_data["y2020"] / merged_data["x2020"]

merged_data.boxplot(column='std_2018', by='Bez')
merged_data.boxplot(column='std_2020', by='Bez')

merged_data_high = merged_data.sort_values('std_2018', ascending=False)

print(merged_data_high['Gemeinde_x'].head(10))

print(merged_data.loc[merged_data['Gemeinde_y'] == 'Kramsach']['std_2018'])
print(merged_data.loc[merged_data['Gemeinde_y'] == 'Kramsach']['std_2020'])

# print(data)
plt.show()


