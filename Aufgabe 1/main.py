import numpy as np

hundret = np.arange(100, 201)

hundret2 = np.arange(100, 202, step=2)

hundret3 = np.arange(100, 110.5, step=0.5)

gleichverteilt = np.random.uniform(size=100)

normalverteilt = np.random.normal(size=100)

mittelwert_gleich = np.mean(gleichverteilt)


print(mittelwert_gleich)
