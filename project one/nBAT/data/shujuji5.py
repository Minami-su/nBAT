import pandas as pd

# Read the first file into a DataFrame
df1 = pd.read_csv(r"H:\parser-main\data\biio\LNCipedia_neg4.csv")

# Read the second file into a DataFrame
df2 = pd.read_csv(r"H:\parser-main\data\biio\positive4.csv")

df_merged = pd.concat([df1, df2])

df_merged.to_csv(r"H:\parser-main\data\biio\sum3.csv", index=False)
