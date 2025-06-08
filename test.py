import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname='/home/grealish/summer/KAN_NET/visual/SimHei.ttf')
plt.title("中文标题", fontproperties=font)
plt.plot([1, 2, 3], [4, 5, 6])
plt.xlabel("横轴", fontproperties=font)
plt.ylabel("纵轴", fontproperties=font)
plt.show()
