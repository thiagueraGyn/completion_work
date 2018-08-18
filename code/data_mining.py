from seaborn import jointplot as jp
from statsmodels.formula.api import ols
from pandas import read_csv
from matplotlib import pyplot as plt, patches, path

file_path = "/Users/thiagocarvalho/Documents/GitHub/completion_work/data/kc_house_data.csv"

df = read_csv(file_path)
df.head()
df.isnull().any()
df.dtypes
df.describe()

fig = plt.figure(figsize=(15, 8))
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

jp(x="sqft_living", y="price", data=df, kind="reg", fit_reg=True, size=8)
plt.show()