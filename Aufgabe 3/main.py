import pandas as pd
from tabulate import tabulate
from matplotlib import pyplot as plt

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

min_data = []
max_data = []
avg_data = []

for x in range(0, len(data)):
    values = data.values[x, 3:]
    min_data.append(values.min())
    max_data.append(values.max())
    avg_data.append(values.mean())



print(data)


# plt.show()


