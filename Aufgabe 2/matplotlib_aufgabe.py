from math import nan
import numpy as np
from matplotlib import pyplot as plt

#source: https://www.kaggle.com/datasets/emmanuelfwerr/london-weather-data
d = np.genfromtxt('data/london_weather.csv', delimiter=",", skip_header=1 )

dt =  d[:,0] #Datum mit folgendem Aufbau: 19790103 (3.Jänner 1979)
# Aufteilen in Tag, Monat, Jahr
day = (dt % 100).astype('i')
month = (dt % 10000 / 100).astype('i')
year = (dt % 100000000 / 10000).astype('i')

mean_temp = d[:,5]

# Check ob es funktioniert hat
print("Jahr:", np.unique(year, return_counts=True))
print("Monat", np.unique(month, return_counts=True))
print("Tag:", np.unique(day, return_counts=True))
print("Jahr MIN MAX" , np.min(year), np.max(year))


temp1979 = mean_temp[year==1979]
temp1979 = temp1979[~np.isnan(temp1979)]

temp1989 = mean_temp[year==1989]
temp1989 = temp1989[~np.isnan(temp1989)]

temp1999 = mean_temp[year==1999]
temp1999 = temp1999[~np.isnan(temp1999)]

temp2009 = mean_temp[year==2009]
temp2009 = temp2009[~np.isnan(temp2009)]

temp2019 = mean_temp[year==2019]
temp2019 = temp2019[~np.isnan(temp2019)]

plt.boxplot([temp1979, temp1989, temp1999, temp2009, temp2019]) #Gegenüberstellung der Sonnenstunden

plt.xticks([1,2,3,4,5],  ["1979","1989","1999", "2009", "2019"])
plt.ylabel("Temperatur")
plt.xlabel("Jahr")
plt.savefig("Temperatur.png")
plt.show()

x = np.arange(1, len(mean_temp[year==1989])+1)
y = mean_temp[year==1989]

plt.plot(x, y, "b.")
plt.xlabel("Tage")
plt.ylabel("Temperatur")
plt.savefig("Verlauf.png")
plt.show()


quant1979 = np.quantile(temp1979, .5)
quant1989 = np.quantile(temp1989, .5)
quant1999 = np.quantile(temp1999, .5)
quant2009 = np.quantile(temp2009, .5)
quant2019 = np.quantile(temp2019, .5)

quantiele = []

for i in range(1979, 2019+1):
    quantiele.append(np.quantile(mean_temp[year==i], .5))

# plt.plot(["1979", "1989", "1999", "2009", "2019"], [quant1979, quant1989, quant1999, quant2009, quant2019], "b.")
plt.plot(np.arange(1979, 2019+1), quantiele, "b.")
plt.ylabel("Temperatur Extremwert")
plt.xlabel("Jahr")
plt.savefig("quantile.png")
plt.show()

mittelwerte = []
years = []

for i in range(2009, 2019+1):
    years.append(str(i))
    mittelwerte.append(np.mean(mean_temp[year==i]))

print(mittelwerte)


plt.bar(years, mittelwerte, align="center")
plt.xlabel("Jahr")
plt.ylabel("Temperatur Mittelwert")
plt.savefig("Mittelwert.png")
plt.show()

radiation = d[:,3]

radiation_ = []
years_ = []

for i in range(1989, 2009+1):
    years_.append(str(i))
    radiation_.append(np.mean(radiation[year==i]))

print(radiation_)

plt.plot(radiation)
plt.ylabel("Strahlung")
plt.xlabel("Jahr")
plt.savefig("Strahlung.png")
plt.show()