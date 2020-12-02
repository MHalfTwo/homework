"""
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sb

df = pd.read_csv('frenchwine.csv')
df.columns = ['alcohol', 'malic_acid', 'ash', 'alcalinity ash', "magnesium", 'species']
df.head()
print(df.head())
df.describe()
print(df.describe())


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

for column_index, column in enumerate(df.columns):
    if column == 'species':
        continue
    plt.subplot(3, 2, column_index + 1)
    sb.violinplot(x='species', y=column, data=df)
plt.show()


all_inputs = df[['alcohol', 'malic_acid',
                 'ash', 'alcalinity ash', "magnesium"]].values
all_species = df['species'].values

(X_train,
 X_test,
 Y_train,


decision_tree_classifier = DecisionTreeClassifier()

model = decision_tree_classifier.fit(X_train, Y_train)
x1=[13.42,3.21,2.62,23.5,95]
x2=[12.32,2.77,2.37,22,90]
x3=[13.75,1.59,2.7,19.5,135]
d=[x1,x2,x3]
e=np.array(d)
yuce=model.predict(e)
yuce=list(yuce)
for i in yuce:
    print(a[i])
from IPython.display import Image
from six import StringIO
from sklearn.tree import export_graphviz

features = list(df.columns[:-1])
print(features)

import pydotplus

os.environ['PATH'] = os.environ['PATH'] + (';d:\\jcs\\Graphviz\\bin\\')  #
dot_data = StringIO()
export_graphviz(decision_tree_classifier, out_file=dot_data, feature_names=features, filled=True, rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph[0].create_png())

