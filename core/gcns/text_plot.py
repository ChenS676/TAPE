import matplotlib.pyplot as plt
import numpy as np

# 生成一些随机数据
data1 = np.random.randn(1000)  # 第一组数据
data2 = np.random.rand(1000) * 100  # 第二组数据

# 创建一个新的图形
fig, ax1 = plt.subplots()

# 绘制第一个直方图 (左轴)
color1 = 'blue'
ax1.hist(data1, bins=30, alpha=0.7, color=color1, edgecolor='black')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency (Data 1)', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

# 创建一个共享x轴的第二个y轴
ax2 = ax1.twinx()

# 绘制第二个直方图 (右轴)
color2 = 'red'
ax2.hist(data2, bins=30, alpha=0.7, color=color2, edgecolor='black')
ax2.set_ylabel('Frequency (Data 2)', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# 添加标题
plt.title('Dual-Axis Histogram')

# 显示图形
plt.savefig('hah')
