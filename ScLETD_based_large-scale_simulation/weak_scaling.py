import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Number of processes
processes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
x2 = np.array(range(0,10))

# Different sets of PFLOPS data
pflops_data_set1 = [0.01192*1000, 23.83, 0.04765*1000, 0.09511*1000, 189.80, 0.37949*1000, 758.15, 1516.13, 3.02483*1000, 6.04707*1000]
pflops_data_set2 = [0.01048*1000, 0.02093*1000, 0.04182*1000, 0.08360*1000, 0.16698*1000, 0.33258*1000, 0.66448*1000, 1.32832*1000, 2.65431*1000, 5.29668*1000]
real1 = [3.4787,3.4805,3.4811,3.4882,3.4958,3.4968,3.5006,3.5010,3.5096,3.5111]
real2 = [6.1040,6.1100,6.1175,6.1198,6.1277,6.1532,6.1595,6.1625,6.1679,6.1818]
data_o = []
data_a = []

# Theoretical scalability curve
theoretical_pflops1 = [1.490125 * p for p in processes]  # Replace this with your actual theoretical scalability data
theoretical_pflops2 = [1.309625 * p for p in processes]
# Create a new figure
fig, ax = plt.subplots(figsize=(8, 6))


for i in range(0,10):
    data_o.append(real1[0]/real1[i]*100)
    data_a.append(real2[0]/real2[i]*100)

# Plot the first curve
recs1=ax.plot(processes, pflops_data_set1, marker='o', label='β->ω solver performance')

# Plot the second curve
recs2=ax.plot(processes, pflops_data_set2, marker='o', label='β->α solver performance')


# Plot the theoretical curve
ax.plot(processes, theoretical_pflops1, marker='x', linestyle='--', label='β->ω ideal performance')
ax.plot(processes, theoretical_pflops2, marker='x', linestyle='--', label='β->α ideal performance')

# Set log scale for both axes
ax.set_xscale('log')
ax.set_yscale('log')

# Add a dotted grid
ax.grid(True, linestyle=':', linewidth=1)

# Set titles and labels
#ax.set_title('Weak Scalability Curves with Theoretical Curve (log-log)')
plt.yticks(theoretical_pflops1[:10],pflops_data_set1[:10])
plt.xticks(processes[:10],processes[:10])
#ax.xaxis.set_major_formatter('{:.2f}')  # 设置 x 轴的有效位数为两位小数
#ax.yaxis.set_major_formatter('{:.1f}')  # 设置 y 轴的有效位数为三位小数
ax.set_xlabel('Process number')
ax.set_ylabel('TFLOPS')

# Show a legend
ax.legend()

temp = [rec.get_data() for rec in recs1]
x = temp[0][0]
y= temp[0][1]
result = [plt.text(x1,y1+4,str(data1)[:5]+'%') for (x1,y1,data1) in zip(x,y,data_o)]

temp = [rec.get_data() for rec in recs2]
x = temp[0][0]
y= temp[0][1]
result = [plt.text(x1,y1-2,str(data1)[:5]+'%') for (x1,y1,data1) in zip(x,y,data_a)]

# Display the plot
plt.show()
#plt.savefig("weak_scaling.pdf",  dpi=600, bbox_inches='tight')
