import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


# 生成等距的横坐标数据

x = np.arange(3)  # 生成 3 个等间隔的数值
x_labels = ['512$^3$', '1024$^3$', '2048$^3$']  # 对应的刻度标签

# 定义多组 y 轴数据
#y1 = [0.057200,0.057464,0.058013]
#y2 = [0.056893,0.058046,0.060101]
#y3 = [0.058700,0.058608,0.060218]
#y4 = [0.057793,0.057970,0.060027]

y1 = [0.028995,0.030154,0.029622]
y2 = [0.029475,0.029320,0.029749]
y3 = [0.028065,0.030124,0.029665]
y4 = [0.029531,0.030234,0.029605]

# 创建第一个子图
plt.subplot(2, 1, 1)  # 2 行 1 列，第一个子图
plt.plot(x, y1, marker='o', label='$\eta_0$', color='#007365')
plt.plot(x, y2, marker='s', label='$\eta_1$', color='#E6862F')
plt.plot(x, y3, marker='^', label='$\eta_2$', color='#FFCD39')
plt.plot(x, y4, marker='*', label='$\eta_3$', color='#D95043')
plt.grid(True, linestyle='--', alpha=0.5)
for i in range(len(x)):
    #plt.text(x[i], y1[i], f'{y1[i]*100:.2f}%', fontsize=8, color='#007365', ha='center', va='bottom')
    #plt.text(x[i], y2[i], f'{y2[i]*100:.2f}%', fontsize=8, color='#E6862F', ha='left', va='top')
    #plt.text(x[i], y3[i], f'{y3[i]*100:.2f}%', fontsize=8, color='#FFCD39', ha='right', va='bottom')
    #plt.text(x[i], y4[i], f'{y4[i]*100:.2f}%', fontsize=8, color='#D95043', ha='center', va='bottom')
    plt.annotate(f'{y1[i]*100:.2f}%', (x[i], y1[i]), textcoords="offset points", xytext=(0,0), ha='right', fontsize=8, color='#007365')
    plt.annotate(f'{y2[i]*100:.2f}%', (x[i], y2[i]), textcoords="offset points", xytext=(0,15), ha='right', fontsize=8, color='#E6862F')
    plt.annotate(f'{y3[i]*100:.2f}%', (x[i], y3[i]), textcoords="offset points", xytext=(0,12), ha='right', fontsize=8, color='#FFCD39')
    plt.annotate(f'{y4[i]*100:.2f}%', (x[i], y4[i]), textcoords="offset points", xytext=(0,9), ha='left', fontsize=8, color='#D95043')

#plt.title('β->ω')
#plt.xlabel('Categories')
plt.ylabel('Relative error')
plt.xticks(x, labels=x_labels)  # 设置 x 轴刻度为等距的数值
plt.yticks(np.arange(0.028, 0.031, 0.0006))
plt.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.6, 0.5))  # 将图例放在子图上方，一行排列
# 设置 y 轴显示百分数
def to_percent(y, position):
    return '%1.2f'%(100*y) + '%'

formatter = FuncFormatter(to_percent)
plt.gca().yaxis.set_major_formatter(formatter)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(hspace=2)  # 调整水平方向的间距


# 显示图形
plt.savefig("elastic_relative_error_omega1.pdf",  dpi=600, bbox_inches='tight')
#plt.show()
