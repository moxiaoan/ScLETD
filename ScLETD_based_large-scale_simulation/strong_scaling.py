from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

x2 = np.array(range(0,5))
ideal = [0,1,2,3,4]
speed = [1,2,4,8,16]
ylabel = [256,512,1024,2048,4096]
#pflops = ['9','18','36','71','143','285','570','1,140']
pflops = ['1', '2', '4', '8', '16']
#real = [1,1.871,3.760,6.164,9.641]
#real = [1,1.843,3.445,6.037,9.821]
#real = [1,1.913,3.873,8.316,16.641]
#real = [1,1.927,3.886,8.657,20.875]
#real = [1,1.987,3.988,7.99,15.989,31.996,63.376,126.489]
real_o = [1,3.4968/1.8680,3.4968/0.9766,3.4968/0.5600,3.4968/0.3583]
#real = [1,89.7395/50.3856,89.7395/25.2272,89.7395/11.2299,89.7395/4.9984]
real_a = [1,6.1532/3.3370,6.1532/1.7852,6.1532/1.0099,6.1532/0.6649]
#real = [1,143.699/75.122,143.699/37.0918,143.699/17.2648,143.699/8.6169]
data_o = []
data_a = []

fig, ax = plt.subplots(figsize=(8, 6))

import math
for i in range(0,5):
    data_o.append(real_o[i]/speed[i]*100)
    #print(real[i],speed[i])
#for i in range(0,5):
#    real_o[i] = real_o[i]/speed[i]
for i in range(0,5):
    data_a.append(real_a[i]/speed[i]*100)
    #print(real[i],speed[i])
#for i in range(0,5):
#    real_a[i] = real_a[i]/speed[i]

ax.plot(ylabel[:5], speed[:5],label='ideal performance',marker = 'x')
recs_o = ax.plot(ylabel[:5], real_o[:5] ,marker='o',label = 'β->ω solver performance')
recs_a = ax.plot(ylabel[:5], real_a[:5] ,marker='o',label = 'β->α solver performance')

ax.set_xscale('log')
ax.set_yscale('log')

plt.yticks(speed[:5],speed[:5])
plt.xticks(ylabel[:5],ylabel[:5])
plt.ylabel('Speedup',fontsize=12)
plt.xlabel('Process number',fontsize=12)
ax.grid(True, linestyle=':', linewidth=1)
plt.legend(fontsize=13)
temp = [rec.get_data() for rec in recs_o]
x = temp[0][0]
y= temp[0][1]
result = [plt.text(x1,y1+0.3,str(data1)[:5]+'%') for (x1,y1,data1) in zip(x,y,data_o)]
temp = [rec.get_data() for rec in recs_a]
x = temp[0][0]
y= temp[0][1]
result = [plt.text(x1,y1-0.2,str(data1)[:5]+'%') for (x1,y1,data1) in zip(x,y,data_a)]
#plt.savefig("strong_scaling.pdf",  dpi=600, bbox_inches='tight')
plt.show()
