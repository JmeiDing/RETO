# import packages
import matplotlib.pyplot as plt
import seaborn



plt.rcParams['figure.figsize'] = (6, 5)
epoch = [512, 650, 800, 950, 1024]
# R1_valid = [36.28, 36.71, 37.29, 37.89, 38.29]
# R2_valid = [8.70, 8.95, 9.13, 9.40, 9.79]
# RL_valid = [20.20, 20.61, 20.94, 21.57, 21.62]
# R1_test = [34.07, 34.40, 34.84, 35.41, 35.72]
# R2_test = [7.31, 7.65, 7.95, 8.29, 8.51]
# RL_test = [18.70, 19.15, 19.40, 19.80, 19.97]

#general
R1_valid = [42.32, 43.53, 43.84, 44.94, 47.01]
R2_valid = [13.48, 13.54, 13.76, 14.57, 15.18]
RL_valid = [23.32, 23.54, 23.81, 24.71, 25.27]
R1_test = [42.01, 42.55, 43.61, 44.07, 45.98]
R2_test = [12.56, 12.84, 13.30, 13.81, 14.36]
RL_test = [22.64, 22.91, 23.34, 23.97, 24.16]
plt.plot(epoch, R1_valid, color=seaborn.xkcd_rgb['dark blue'], linestyle='-', marker='v', markevery=1, markersize=5,
         mew=1.25)
plt.plot(epoch, R1_test, color=seaborn.xkcd_rgb['marine blue'], linestyle='--', marker='.', markevery=1,
         markersize=9, mew=1.25)
plt.plot(epoch, R2_valid, color='darkorange', linestyle='-', marker='v', markevery=1, markersize=5, mew=1.25)
plt.plot(epoch, R2_test, color='orange', linestyle='--', marker='.', markevery=1, markersize=9, mew=1.25)
plt.plot(epoch, RL_valid, color=seaborn.xkcd_rgb['darkish purple'], linestyle='-', marker='v', markevery=1,
         markersize=5, mew=1.25)
plt.plot(epoch, RL_test, color=seaborn.xkcd_rgb['purple'], linestyle='--', marker='.', markevery=1,
         markersize=9, mew=1.25)

font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 19,
         }
ax = plt.gca()  # 获得坐标轴的句柄
ax.spines['bottom'].set_linewidth(1.5)  ###设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(1.5)  ####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(0.5)  ###设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(0.5)  ####设置上部坐标轴的粗细
# ax.spines['bottom'].set_position(('data', 0.1))  #data表示通过值来设置x轴的位置，将x轴绑定在y=0的位置
# ax.spines['top'].set_position(('data', 0.1))
plt.xlabel('Input length', font2, labelpad=1)
#plt.ylabel('CTIS task', font2)
plt.tick_params(labelsize=16)
font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 12,
         }

plt.legend(['R-1_valid', 'R-1_test', 'R-2_valid', 'R-2_test', 'R-L_valid', 'R-L_test'], loc="lower right",
           prop=font1, edgecolor='black', bbox_to_anchor=(0.98, 0.42), shadow=True)
plt.xlim((500, 1100))
plt.ylim((10, 50))
plt.yticks(fontfamily="Times New Roman", fontsize=15, weight='normal')
plt.xticks(epoch, fontfamily="Times New Roman", fontsize=15, weight='normal')

plt.grid(axis='y')  # 添加横着的网格线

plt.tight_layout()
# plt.savefig('plt.svg')
plt.savefig('/Users/dingjunmei/code/RL_LLM/rl4lms_exps/4/CTIS_task.pdf')
plt.show()
