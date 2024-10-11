'''

import pandas as pd
data = pd.read_csv('/Users/wangbo/Documents/Time-Series-Library-git/Time-Series-Library-1/datasets/Merged_Data.csv')
print(data.isnull().sum())  # 检查数据中的NaN值

import pandas as pd

# 读取数据
data = pd.read_csv('/Users/wangbo/Documents/Time-Series-Library-git/Time-Series-Library-1/datasets/Merged_Data.csv')

# 删除包含缺失值的行
data_cleaned = data.dropna()

# 保存处理后的数据
data_cleaned.to_csv('/Users/wangbo/Documents/Time-Series-Library-git/Time-Series-Library-1/datasets/Merged_Data_cleaned.csv', index=False)

'''


# 打印模型结构
print(LightTS)

# 计算并打印模型的参数数量
total_params = sum(p.numel() for p in LightTS.parameters() if p.requires_grad)
print(f'Total learnable parameters: {total_params}')