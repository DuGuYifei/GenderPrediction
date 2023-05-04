from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.data_handler import handle_data
import pandas as pd
import pickle

# 读取数据，已删除所有空格
data = pd.read_csv('gender.csv')

# 处理数据
del data['Unnamed: 9']
data = handle_data(data)

# 分离特征和目标变量
X = data.drop(['Gender'], axis=1)
y = data['Gender']

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建random forest和decision tree分类器
rfc = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
dtc = DecisionTreeClassifier(max_depth=5, random_state=42)

# 训练模型
rfc.fit(X_train, y_train)
dtc.fit(X_train, y_train)

# 在测试集上进行预测
y_pred_rf = rfc.predict(X_test)
y_pred_dt = dtc.predict(X_test)

# 计算准确率
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

print("Random forest Accuracy:", accuracy_rf)
print("Decision tree Accuracy:", accuracy_dt)

# 输出 1.0和1.0，数据集较少，所以使用决策树模型
with open('dtc_model.pkl', 'wb') as f:
    pickle.dump(dtc, f)
