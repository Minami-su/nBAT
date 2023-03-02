import pandas as pd

# 读取 csv 文件
df1 = pd.read_csv(r"H:\project one\nBAT\data\case.csv")
df2 = pd.read_csv(r"H:\project one\nBAT\case_study\case_feature2.txt", names=["yufa"])

# 设置循环变量
i = 0
j = 0

# 迭代两个文件
for index, row in df1.iterrows():
    # 将 2.csv 的值写入 1.csv 的 yufa 列
    df1.at[index, 'yufa'] = df2.iloc[j, 0]
    i += 1
    if i % 3 == 0:
        j += 1

# 保存文件
df1.to_csv(r"H:\project one\nBAT\data\case2.csv", index=False)