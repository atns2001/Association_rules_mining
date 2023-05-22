import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import math
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

#中文乱码处理
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

file_path = '../data/anonymous-msweb.data'

# 所有网站表
websites = {}
# 当前的网站ID
website_key = None
# 当前ID对应的网站信息
website_values = []

# 所有的用户访问网站表
users = {}
# 当前的用户ID
user_key = None
# 当前ID对应的访问网站ID
user_values = []

with open(file_path, "r") as myfile:
    for line in myfile:
        line = line.rstrip("\n")
        linedata = line.split(',')

        # 如果是'C'行，则更新当前键和值列表
        if linedata[0] == 'C':
            if user_key is not None:
                users[user_key] = user_values
            user_key = linedata[2]
            user_values = []
        # 如果是'V'行，则将第用户访问的网站信息添加到uservalues
        elif linedata[0] == 'V':
            user_values.append(int(linedata[1]))
        # 如果是'A'行，则将网站信息添加到websites
        elif linedata[0] == 'A':
            website_key = linedata[1]
            website_values.append(linedata[3])
            website_values.append(linedata[4])
            websites[website_key] = website_values
            website_values = []

# 处理最后一个键值对
if user_key is not None:
    users[user_key] = user_values

# # 数据预处理: 清洗数据，处理缺失值，提取用户浏览记录
# # [str, int]
# print('用户数据集长度: ' + str(len(users)))
# for key, values in users.items():
#     print(f"{key}: {values}")
#
# # [str, str]
# print('网站数据集长度: ' + str(len(websites)))
# for key, values in websites.items():
#     print(f"{key}: {values}")

# 统计各个值出现的次数
value_counts = Counter(value for values in users.values() for value in values)

# 按照从大到小的顺序对统计结果排序
sorted_counts = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)

# # 输出网站及点击次数
# for value, count in sorted_counts:
#     if str(value) in websites:
#         print(f"{value}: {count}")
#
# # 输出网站（将ID转换为具体信息）及点击次数
# for value, count in sorted_counts:
#     if str(value) in websites:
#         print(f"{websites[str(value)]}: {count}")
#
# # 输出网站及点击次数柱状图
# pages = [item[0] for item in sorted_counts]
# page_visits = [item[1] for item in sorted_counts]
# plt.bar(pages, page_visits)
# plt.xlabel('Website')
# plt.ylabel('Visits')
# plt.title('Website Visits Distribution')
# plt.xticks(rotation=90)
# plt.show()

transactions = []
for key, values in users.items():
    transactions.append(values)

# 使用TransactionEncoder转换事务列表为布尔矩阵
te = TransactionEncoder()
te_ary = te.fit_transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

df = df.astype(bool)

# 使用Apriori算法计算频繁项集
frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)

frequent_itemsets = frequent_itemsets.sort_values(by='support')

# 输出频繁项集
print(frequent_itemsets)

# 使用关联规则挖掘
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# 输出关联规则
transposed_rules = rules.transpose()
output = transposed_rules.to_string()

print(output)

