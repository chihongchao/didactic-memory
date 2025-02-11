from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, \
    ConfusionMatrixDisplay, classification_report
from imblearn.over_sampling import SMOTE
import os
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# 定义数据目录
cache_dir = './data'

# 确保数据目录存在
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# 数据文件路径
data_file_path = os.path.join(cache_dir, 'industry_type_analysis.json')

# 加载数据
with open(data_file_path, 'r', encoding='utf-8') as jsonfile:
    data = json.load(jsonfile)

# 数据预处理
def preprocess_data(data):
    df = pd.DataFrame()

    for industry, years in data.items():
        if len(years) < 7:
            continue

        for year, metrics in years.items():
            # 确保年份在2015年及之后
            if int(year) < 2015:
                continue

            # 确保当前年份有完整的前5年数据
            if int(year) - 5 < 2015:
                continue

            # 检查当前年份和前5年的数据是否完整
            if any(str(int(year) - i) not in years for i in range(5)):
                continue

            # 检查当前年份和下一年的数据是否完整
            next_year = str(int(year) + 1)
            if next_year not in years or years[next_year]['平均市净率'] is None or metrics['平均市净率'] is None:
                continue

            features = {
                '行业': industry,
                '年份': int(year),
                '平均市盈率': metrics['平均市盈率'],
                '平均市净率': metrics['平均市净率'],
                '平均负债率': metrics['平均负债率'],
                '平均ROE': metrics['平均ROE'],
                '近5年平均市净率': metrics['近5年平均市净率']
            }

            # 计算目标变量
            n = years[next_year]['平均市净率'] / metrics['平均市净率']
            if n >= 1.15:
                features['Label'] = 1
            elif n >= 1.1 and n < 1.15:
                features['Label'] = 2
            elif n >= 1.0 and n < 1.1:
                features['Label'] = 3
            else:
                features['Label'] = 0

            df = pd.concat([df, pd.DataFrame([features])], ignore_index=True)

    return df

# 预处理数据
df = preprocess_data(data)
df.dropna(inplace=True)

# 预处理数据
df = preprocess_data(data)
df.dropna(inplace=True)

# 特征选择
features = ['平均市盈率', '平均市净率', '平均负债率', '平均ROE', '近5年平均市净率']
X = df[features]
y = df['Label']

# 处理类别不平衡
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# 模型选择：XGBoost
xgb_model = XGBClassifier(random_state=42, objective='multi:softmax', num_class=4)

# 超参数调优
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 2, 3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

random_search = RandomizedSearchCV(xgb_model, param_grid, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                                   scoring='precision_macro', n_jobs=-1, n_iter=20, random_state=42)
random_search.fit(X_train, y_train)

# 最佳模型
best_model = random_search.best_estimator_

# 交叉验证评估
cv_scores = cross_val_score(best_model, X_train, y_train, cv=StratifiedKFold(n_splits=30, shuffle=True, random_state=42),
                            scoring='precision_macro')
print(f"交叉验证平均精确率: {np.mean(cv_scores):.4f}")

# 测试集评估
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"测试集准确率: {accuracy:.4f}")
print(f"测试集精确率: {precision:.4f}")
print(f"测试集召回率: {recall:.4f}")
print(f"测试集F1分数: {f1:.4f}")

# 输出分类报告
print("\n分类报告：")
print(classification_report(y_test, y_pred))

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# 预测2024年的数据
def predict_2024(data, best_model, scaler):
    df_2024 = pd.DataFrame()

    for industry, years in data.items():
        if len(years) < 7:
            continue

        if '2024' in years:
            metrics = years['2024']
            if metrics['平均市净率'] is None or metrics['近5年平均市净率'] is None:
                continue

            features_dict = {
                '行业': industry,
                '年份': 2024,
                '平均市盈率': metrics['平均市盈率'],
                '平均市净率': metrics['平均市净率'],
                '平均负债率': metrics['平均负债率'],
                '平均ROE': metrics['平均ROE'],
                '近5年平均市净率': metrics['近5年平均市净率']
            }

            df_2024 = pd.concat([df_2024, pd.DataFrame([features_dict])], ignore_index=True)

    X_2024 = df_2024[features]
    X_2024_scaled = scaler.transform(X_2024)
    y_2024_pred = best_model.predict(X_2024_scaled)
    df_2024['预测Label'] = y_2024_pred

    return df_2024

# 预测2024年的数据
df_2024 = predict_2024(data, best_model, scaler)

# 输出预测结果
print("\n2025年预测结果：")
print("\n预测市净率增长大于15%的行业（Label=1）：")
print(df_2024[df_2024['预测Label'] == 1][['行业', '平均市净率']])

print("\n预测市净率增长在10%-15%之间的行业（Label=2）：")
print(df_2024[df_2024['预测Label'] == 2][['行业', '平均市净率']])

print("\n预测市净率增长在0%-10%之间的行业（Label=3）：")
print(df_2024[df_2024['预测Label'] == 3][['行业', '平均市净率']])

print("\n预测市净率增长小于0%的行业（Label=0）：")
print(df_2024[df_2024['预测Label'] == 0][['行业', '平均市净率']])
