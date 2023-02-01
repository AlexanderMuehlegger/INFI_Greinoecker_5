from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr, pearsonr, chi2_contingency


df = pd.read_csv('./data/ESS8e02.1_F1.csv', sep=",")

df['gndr'] = pd.cut(df['gndr'], [0,1,2,9], labels=['Male', 'Female', 'No Answer'])

df['gndr'].value_counts().plot(kind='bar')
plt.show()

df_at = df.loc[df.cntry == 'AT']
df_at_it = df.loc[df.cntry.isin(['AT', 'IT'])]

df_female = df.loc[df['gndr'] == 'Female']
df_male = df.loc[df['gndr'] == 'Male']

df_gndr_cntry = pd.crosstab(df['gndr'], df['cntry'])

trust_male = df['trstplc'].loc[df['gndr'] == 'Male']
trust_female = df['trstplc'].loc[df['gndr'] == 'Female']

plt.bar(['Male'], trust_male.sum(), label='Male')
plt.bar(['Women'], trust_female.sum(), label='Women')
plt.show()

s, p = mannwhitneyu(trust_male.sum(), trust_female.sum())
print("Ergebnis: ", s)
print("P-Wert: ", p)

#Hypothese Männer vertrauen Polizei mehr:
#   falsch

solar = df["elgsun"]
solar.drop(solar.loc[solar > 0].index)

nuklear = df["elgnuc"]
nuklear.drop(nuklear.loc[nuklear > 0].index)

crossT = pd.crosstab(nuklear, solar, normalize='index').round(decimals = 3)
chi, p, dof, expected = chi2_contingency(crossT)

sns.heatmap(crossT, annot=False, cmap="YlGnBu")
sns.heatmap(crossT, annot=crossT, annot_kws={'va':'bottom'}, fmt="", cbar=False , cmap="YlGnBu")
sns.heatmap(crossT, annot=expected, annot_kws={'va':'top'}, fmt=".2f", cbar=False, cmap="YlGnBu")
plt.show()  

print("P-Wert: " + str(p))
#Hypothese negativer Zusammenhang zwischen 'mehr Strom aus nuklearer Energie' und 'mehr Strom aus Solarenergie'
# richtig

ungarn = df.loc[df.cntry == "HU"]["ccgdbd"]
oesterreich = df.loc[df.cntry == "AT"]["ccgdbd"]

ungarn.drop(ungarn.loc[ungarn > 5].index)
oesterreich.drop(oesterreich.loc[oesterreich > 5].index)

ungarn = ungarn.value_counts(normalize=True)
oesterreich = oesterreich.value_counts(normalize=True)

plt.bar(['Österreich', 'Ungarn'], 
    [pd.Series.median(oesterreich), 
     pd.Series.median(ungarn)])
plt.show()

chi, p, dof, expected = chi2_contingency(pd.crosstab(oesterreich, ungarn))
print ("P-Wert: ", p)

#Hypothese Auswirkung Klimawandel auf Menschen, Eindruck Österreich > Eindruck Ungarn
#   richtig

einkommen = df["basinc"]
male_female = df.loc[(df["gndr"] == "Female") | (df["gndr"] == "Male")]["gndr"]

x = pd.crosstab(einkommen, male_female, normalize='index')
x.plot.bar(rot=0)
plt.show()

chi, p, dof, expected = chi2_contingency(x)
print ("P-Wert:", p)

#Hypothese mehr Frauen für bedingungsloses Grundeinkommen
#   richtig





