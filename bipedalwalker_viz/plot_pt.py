
import torch
import matplotlib.pyplot as plt

# 从.pt文件中加载数据
data = torch.load('./predict.pt')

# 打印数据
print(data)

# 将数据转换为numpy数组
data = data.cpu().numpy()

# 绘制折线图
plt.plot(data)

# # 添加标题和轴标签
# plt.title('Loss Plot')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')

# 显示图形
plt.savefig('predict.png')

