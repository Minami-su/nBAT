

import pandas as pd

# Read the data into a pandas DataFrame
df = pd.read_csv(r"H:\parser-main\data\biio\pos_features3.txt", names=["yufa"])

# Read the existing LNCipedia_neg.csv file into a DataFrame
#df_neg = pd.read_csv(r"H:\parser-main\data\biio\188.txt")
df_neg = pd.read_csv(r"H:\parser-main\data\biio\positive2.csv")
# Append the data from the sb.txt file to the third column of the LNCipedia_neg.csv file
# Reindex df["yufa"]
#df_neg["yufa"] = pd.concat([df["yufa"].iloc[range(0, len(df), 3)] for i in range(int(20523/2281))], ignore_index=True)
# Assign the reindexed df["yufa"] to df_neg["yufa"]
#df_neg["yufa"] = df["yufa"]

df_neg["target"] ='1'
df_neg["yufa"] ='1'
# Save the updated DataFrame to the LNCipedia_neg.csv file
df_neg.to_csv(r"H:\parser-main\data\biio\positive4.csv", index=False)