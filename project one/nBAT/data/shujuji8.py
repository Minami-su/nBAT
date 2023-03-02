import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
# 读取数据
df = pd.read_csv(r"H:\parser-main\data\biio\sum3.csv")


X = df["source"]
y = df["target"]
yufa = df["yufa"]
#0.1993
# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test, yufa_train, yufa_test = train_test_split(X, y, yufa, test_size=0.1999, random_state=False)

# 将训练集划分为训练集和验证集
X_val, X_test,y_val, y_test, yufa_val, yufa_test = train_test_split(X_test, y_test, yufa_test, test_size=0.5, random_state=False)



# 将训练集、验证集和测试集保存到 CSV 文件中
pd.concat([X_train,y_train, yufa_train], axis=1).to_csv(r"C:\\Users\\Administrator\\Desktop\\biosfa3L.train.csv", index=False)
pd.concat([X_val, y_val, yufa_val], axis=1).to_csv(r"C:\\Users\\Administrator\\Desktop\\biosfa3L.valid.csv", index=False)
pd.concat([X_test,y_test, yufa_test], axis=1).to_csv(r"C:\\Users\\Administrator\\Desktop\\biosfa3L.test.csv", index=False)
#pd.concat([X, y, yufa], axis=1).to_csv(r"C:\\Users\\Administrator\\Desktop\\bios6.valid.csv", index=False)

# 读取数据
train_df = pd.read_csv(r"C:\\Users\\Administrator\\Desktop\\biosfa3L.train.csv")
val_df = pd.read_csv(r"C:\\Users\\Administrator\\Desktop\\biosfa3L.valid.csv")
test_df = pd.read_csv(r"C:\\Users\\Administrator\\Desktop\\biosfa3L.test.csv")
#
#
# 输出数据条数
print(f"train.csv: {train_df.shape[0]}")
print(f"val.csv: {val_df.shape[0]}")
print(f"test.csv: {test_df.shape[0]}")
print(f"test.csv: {test_df.shape[0]+val_df.shape[0]+train_df.shape[0]}")