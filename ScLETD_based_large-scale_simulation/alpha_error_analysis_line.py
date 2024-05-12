import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


# 生成等距的横坐标数据

x = np.arange(3)  # 生成 3 个等间隔的数值
x_labels = ['512$^3$', '1024$^3$', '2048$^3$']  # 对应的刻度标签

# 定义多组 y 轴数据

#y5 = [0.057689,0.058456,0.058876]
#y6 = [0.057536,0.058302,0.058979]
#y7 = [0.074890,0.075689,0.076577]
#y8 = [0.072578,0.076790,0.076654]
#y9 = [0.056848,0.057679,0.058089]
#y10 = [0.056324,0.057530,0.057890]

y5 = [0.034344,0.031707,0.030199]
y6 = [0.025362,0.030662,0.030938]
y7 = [0.047356,0.041670,0.040212]
y8 = [0.043126,0.038609,0.041419]
y9 = [0.034157,0.032948,0.031388]
y10 = [0.026894,0.031551,0.032244]


# 创建第二个子图
plt.subplot(2, 1, 2)  # 2 行 1 列，第二个子图
plt.plot(x, y5, marker='o', label='$\eta_0$', color='#007365')
plt.plot(x, y6, marker='s', label='$\eta_1$', color='#E6862F')
plt.plot(x, y7, marker='^', label='$\eta_2$', color='#FFCD39')
plt.plot(x, y8, marker='*', label='$\eta_3$', color='#D95043')
plt.plot(x, y9, marker='o', label='$\eta_4$', color='#0099DD')
plt.plot(x, y10, marker='s', label='$\eta_5$', color='#6C5B7B')
#plt.title('β->α')
plt.grid(True, linestyle='--', alpha=0.5)
for i in range(len(x)):
    #plt.text(x[i], y5[i], f'{y5[i]*100:.2f}%', fontsize=8, color='#007365', ha='right', va='bottom')
    #plt.text(x[i], y6[i], f'{y6[i]*100:.2f}%', fontsize=8, color='#E6862F', ha='right', va='bottom')
    #plt.text(x[i], y7[i], f'{y7[i]*100:.2f}%', fontsize=8, color='#FFCD39', ha='right', va='bottom')
    #plt.text(x[i], y8[i], f'{y8[i]*100:.2f}%', fontsize=8, color='#D95043', ha='left', va='top')
    #plt.text(x[i], y9[i], f'{y9[i]*100:.2f}%', fontsize=8, color='#0099DD', ha='right', va='bottom')
    #plt.text(x[i], y10[i], f'{y10[i]*100:.2f}%', fontsize=8, color='#6C5B7B', ha='right', va='bottom')
    plt.annotate(f'{y5[i]*100:.2f}%', (x[i], y5[i]), textcoords="offset points", xytext=(0,20), ha='right', fontsize=8, color='#007365')
    plt.annotate(f'{y6[i]*100:.2f}%', (x[i], y6[i]), textcoords="offset points", xytext=(0,15), ha='right', fontsize=8, color='#E6862F')
    plt.annotate(f'{y7[i]*100:.2f}%', (x[i], y7[i]), textcoords="offset points", xytext=(0,12), ha='right', fontsize=8, color='#FFCD39')
    plt.annotate(f'{y8[i]*100:.2f}%', (x[i], y8[i]), textcoords="offset points", xytext=(0,9), ha='left', fontsize=8, color='#D95043')
    plt.annotate(f'{y9[i]*100:.2f}%', (x[i], y9[i]), textcoords="offset points", xytext=(0,4), ha='right', fontsize=8, color='#0099DD')
    plt.annotate(f'{y10[i]*100:.2f}%', (x[i], y10[i]), textcoords="offset points", xytext=(0,0), ha='right', fontsize=8, color='#6C5B7B')

#plt.xlabel('Categories')
#plt.ylabel('Relative error')
plt.xticks(x, labels=x_labels)  # 设置 x 轴刻度为等距的数值
plt.yticks(np.arange(0.025, 0.048, 0.0048))
def to_percent(y, position):
    return '%1.2f'%(100*y) + '%'
formatter = FuncFormatter(to_percent)
plt.legend(loc='upper center', ncol=6, bbox_to_anchor=(0.55, 0.3))  # 将图例放在子图上方，一行排列
plt.gca().yaxis.set_major_formatter(formatter)  # 设置第二个子图的纵坐标显示百分数


# 调整布局
plt.tight_layout()
plt.subplots_adjust(hspace=2)  # 调整水平方向的间距


# 显示图形
#plt.show()
plt.savefig("elastic_relative_error_alpha1.pdf",  dpi=600, bbox_inches='tight')
