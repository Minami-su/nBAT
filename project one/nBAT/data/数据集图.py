import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# load data for coding genes
with open('pos_features3.txt', 'r') as f:
    pos_data = [line.split() for line in f]

pos_hexamer_score = [float(row[30]) for row in pos_data]
pos_fickett_score = [float(row[31]) for row in pos_data]
pos_orf_size = [math.log10(float(row[32])) for row in pos_data]

# load data for non-coding genes
with open('neg_features.txt', 'r') as f:
    neg_data = [line.split() for line in f]

neg_hexamer_score = [float(row[30]) for row in neg_data]
neg_fickett_score = [float(row[31]) for row in neg_data]
neg_orf_size = [float(row[33]) for row in neg_data]

# create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plot coding genes
ax.scatter(pos_fickett_score, pos_hexamer_score, pos_orf_size, c='r', s=0.3, alpha=1)
ax.scatter([], [], [],c='r',s=10, alpha=1, label='coding')

# plot non-coding genes
ax.scatter(neg_fickett_score, neg_hexamer_score, neg_orf_size, c='b', s=0.3, alpha=1)

ax.scatter([], [], [],c='b',s=10, alpha=1, label='non-coding')
# set axis labels
ax.set_xlabel('Fickett Score')
ax.set_ylabel('Hexamer Score')
ax.set_zlabel('ORF Size(bp, log10)')
# 添加图例
ax.legend()

# set axis limits
# ax.set_xlim(-0.3, -0.1)![](C:/Users/Administrator/Desktop/Figure_3d.png)
# ax.set_ylim(1, 1.15)
# ax.set_zlim(-3000, -500)
#ax.set_zlim(-100, 1000)
plt.show()
# ax.view_init(elev=30, azim=-30)

# 保存图片
plt.savefig('3D_scatter2.png', dpi=600)


