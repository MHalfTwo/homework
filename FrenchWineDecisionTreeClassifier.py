# coding=utf-8
"""
利用决策树算法进行分类
作者：土木二班陈宇翔
日期：2020/12/2
"""
import pandas as pd  # 调入需要用的库
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sb

# 调入数据
df = pd.read_csv('frenchwine.csv')
df.columns = ['alcohol', 'malic_acid', 'ash', 'alcalinity ash', "magnesium", 'species']
# 查看前6条数据
df.head()
print(df.head())
a={"Zinfandel":"仙粉黛","Syrah":"西拉","Sauvignon":"赤霞珠"}
# 查看数据描述性统计信息
df.describe()
print(df.describe())


def scatter_plot_by_category(feat, x, y):  # 数据的可视化
    alpha = 0.5
    gs = df.groupby(feat)
    cs = cm.rainbow(np.linspace(0, 1, len(gs)))
    for g, c in zip(gs, cs):
        plt.scatter(g[1][x], g[1][y], color=c, alpha=alpha)


plt.figure(figsize=(20, 5))
plt.subplot(131)
scatter_plot_by_category('species', 'ash', 'malic_acid')
plt.xlabel('ash')
plt.ylabel('malic_acid')
plt.title('species')
plt.show()

plt.figure(figsize=(20, 10))  # 利用seaborn库绘制三种Iris花不同参数图
for column_index, column in enumerate(df.columns):
    if column == 'species':
        continue
    plt.subplot(3, 2, column_index + 1)
    sb.violinplot(x='species', y=column, data=df)
plt.show()

# 首先对数据进行切分，即划分出训练集和测试集
from sklearn.model_selection import train_test_split  # 调入sklearn库中交叉检验，划分训练集和测试集

all_inputs = df[['alcohol', 'malic_acid',
                 'ash', 'alcalinity ash', "magnesium"]].values
all_species = df['species'].values

(X_train,
 X_test,
 Y_train,
 Y_test) = train_test_split(all_inputs, all_species, train_size=0.7, random_state=1)  # 70%的数据选为训练集

# 使用决策树算法进行训练
from sklearn.tree import DecisionTreeClassifier  # 调入sklearn库中的DecisionTreeClassifier来构建决策树

# 定义一个决策树对象
decision_tree_classifier = DecisionTreeClassifier()

# 训练模型
model = decision_tree_classifier.fit(X_train, Y_train)
# 输出模型的准确度
x1=[13.42,3.21,2.62,23.5,95]
x2=[12.32,2.77,2.37,22,90]
x3=[13.75,1.59,2.7,19.5,135]
d=[x1,x2,x3]
e=np.array(d)
#print(decision_tree_classifier.score(X_test, Y_test))
# 使用训练的模型进行预测，为了方便，
# 案例直接把测试集里面的数据拿出来三条；实际使用时，请利用所有测试集数据即X_test对模型进行测试。
#print(X_test)  # 利用3个数据进行测试，即取3个数据作为模型的输入端
#b=model.predict(X_test)
#print(model.predict(X_test))  # 输出测试的结果，即输出模型预测的结果
#b=list(b)
#for i in b:
#    print(a[i])
print("预测对象为（13.42,3.21,2.62,23.5,95）,( 12.32,2.77,2.37,22,90),( 13.75,1.59,2.7,19.5,135)")
yuce=model.predict(e)
yuce=list(yuce)
for i in yuce:
    print(a[i])
##决策树可视化
from IPython.display import Image
# from sklearn.externals.six import StringIO  #sklearn 0.23版本已经删掉了这个包,直接安装six即可
from six import StringIO
from sklearn.tree import export_graphviz

features = list(df.columns[:-1])
print(features)

import pydotplus
import os  # 要安装一个Graphviz软件

os.environ['PATH'] = os.environ['PATH'] + (';d:\\jcs\\Graphviz\\bin\\')  #
dot_data = StringIO()
export_graphviz(decision_tree_classifier, out_file=dot_data, feature_names=features, filled=True, rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph[0].create_png())
graph.write_pdf("frenchwine.pdf")  # 将iris数据集利用决策树算法可视化结果保持到iris.pdf文件中
print("保存完毕")


