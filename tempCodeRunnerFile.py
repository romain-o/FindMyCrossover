import pandas as pd

df1 = pd.read_csv('catalog_inductors.csv')
#Replace 0.0 by nan
df1['Price'] = df1['Price'].replace(0.0, pd.NA)
df1.to_csv('catalog_inductors.csv', index=False)

df1 = pd.read_csv('catalog_capacitors.csv')
df1['Price'] = df1['Price'].replace(0.0, pd.NA)
df1.to_csv('catalog_capacitors.csv', index=False)

df1 = pd.read_csv('catalog_resistors.csv')
df1['Price'] = df1['Price'].replace(0.0, pd.NA)
df1.to_csv('catalog_resistors.csv', index=False)