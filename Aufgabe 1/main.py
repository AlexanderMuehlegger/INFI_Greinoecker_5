import numpy as np

#1.1.1
a = np.arange(100, 201)

print("1.1.1: {}".format(a))

#1.1.2
a = np.arange(100, 201, 2)

print("1.1.2: {}".format(a))

#1.1.3
a = np.arange(100, 110.5, .5)

print("1.1.2: {}".format(a))

#1.1.4
gleichverteilt = np.random.uniform(size=100)
normalverteilt = np.random.normal(size=100)

print("\ngleichverteilt: {}".format(gleichverteilt))
print("\nNormalverteilt: {}".format(normalverteilt))

#2.1
print("\nMittelwert: {}".format(normalverteilt.mean()))
print("Median: {}".format(np.median(normalverteilt)))
print("Minimum: {}".format(normalverteilt.min()))
print("Maximum: {}".format(np.std(normalverteilt)))

#2.2
print(np.mean(normalverteilt) - np.std(normalverteilt)*2, np.mean(normalverteilt) + np.std(normalverteilt)*2)

#2.3
print(normalverteilt * 100)

#2.4
print(normalverteilt[:10])

#2.5
print(normalverteilt[normalverteilt>0])

