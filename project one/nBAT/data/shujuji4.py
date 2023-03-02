

import pandas as pd

# Read the data into a pandas DataFrame
df = pd.read_csv(r"H:\parser-main\data\biio\neg_features.txt", names=["yufa"])

# Read the existing LNCipedia_neg.csv file into a DataFrame
df_neg = pd.read_csv(r"H:\parser-main\data\biio\LNCipedia_neg2.csv")
#df_neg = pd.read_csv(r"H:\parser-main\data\biio\088.txt")
# Append the data from the sb.txt file to the third column of the LNCipedia_neg.csv file
#f_neg["yufa"] = pd.concat([df["yufa"].iloc[range(0, len(df), 3)] for i in range(int(20523/2281))], ignore_index=True)
df_neg["target"] ='0'
#df_neg["yufa"] = df["yufa"]
df_neg["yufa"] ='1'
# Save the updated DataFrame to the LNCipedia_neg.csv file
df_neg.to_csv(r"H:\parser-main\data\biio\LNCipedia_neg4.csv", index=False)