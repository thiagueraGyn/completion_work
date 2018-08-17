import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols

df = pd.read_csv('/data/kc_house_data.csv')
df.head()

df.isnull().any()
df.dtypes

df.describe()

fig = plt.figure(figsize=(12, 6))
sqft = fig.add_subplot(121)
cost = fig.add_subplot(122)

sqft.hist(df.sqft_living, bins=80)
sqft.set_xlabel('Ft^2')
sqft.set_title("Histogram of House Square Footage")

cost.hist(df.price, bins=80)
cost.set_xlabel('Price ($)')
cost.set_title("Histogram of Housing Prices")

plt.show()

m = ols('price ~ sqft_living',df).fit()
print (m.summary())

m = ols('price ~ sqft_living + bedrooms + grade + condition',df).fit()
print (m.summary())

sns.jointplot(x="sqft_living", y="price", data=df, kind = 'reg',fit_reg= True, size = 7)
plt.show()