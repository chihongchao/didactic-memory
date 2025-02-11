import json
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# 读取JSON文件
def load_data(file_path):
    with open(file_path, mode='r', encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
    return data

# 提取特征和标签
def extract_features_and_labels(data):
    features = []
    labels = []
    years = []
    for combination in data:
        for attributes in combination["组合属性"]:
            # 提取特征
            feature = [
                float(attributes["组合收益"].strip('%')),
                float(attributes["组合收益标准差"].strip('%')),
                float(attributes["组合前年收益"].strip('%')),
                float(attributes["平均ROE"].strip('%')),
                float(attributes["ROE标准差"].strip('%')),
                float(attributes["平均营业额增长率"].strip('%')),
                float(attributes["营业额增长率标准差"].strip('%')),
                float(attributes["平均利润增长率"].strip('%')),
                float(attributes["利润增长率标准差"].strip('%')),
                float(attributes["平均市值"]),
                float(attributes["市值标准差"]),
                float(attributes["平均动态市盈率TTM"]),
                float(attributes["动态市盈率TTM标准差"]),
                float(attributes["平均静态市盈率"]),
                float(attributes["静态市盈率标准差"]),
                float(attributes["平均市净率"]),
                float(attributes["市净率标准差"]),
                float(attributes["平均市现率"]),
                float(attributes["市现率标准差"])
            ]
            # 提取标签
            label = 1 if int(attributes["次年组合收益排名"]) < 100 else 0
            features.append(feature)
            labels.append(label)
            years.append(int(attributes["data_date"]))  # 假设每个属性都有一个 "data_date" 字段表示年份
    return np.array(features), np.array(labels), np.array(years)

# 训练随机森林模型
def train_model(features, labels, years):
    # 数据标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=5)

    # 数据平衡
    over_sampler = BorderlineSMOTE(random_state=42)
    under_sampler = RandomUnderSampler(random_state=42)
    pipeline = Pipeline([
        ('over', over_sampler),
        ('under', under_sampler)
    ])

    # 参数网格
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }

    # 初始化随机森林分类器
    rf = RandomForestClassifier(random_state=42)

    # 初始化 GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tscv, scoring='f1_weighted', n_jobs=-1)

    # 训练模型
    grid_search.fit(features_scaled, labels)

    # 输出最佳参数
    print("Best parameters found: ", grid_search.best_params_)

    # 使用最佳参数训练模型
    best_model = grid_search.best_estimator_

    # 评估模型
    for train_index, test_index in tscv.split(features_scaled):
        X_train, X_test = features_scaled[train_index], features_scaled[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train, y_train)

        best_model.fit(X_train_resampled, y_train_resampled)
        y_pred = best_model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

    # 输出模型内容
    print("\nModel Details:")
    print("Number of trees in the forest:", best_model.n_estimators)
    print("Depth of the forest:", best_model.max_depth)
    print("Minimum samples split:", best_model.min_samples_split)
    print("Minimum samples leaf:", best_model.min_samples_leaf)
    print("Class weights:", best_model.class_weight)
    print("Feature importances:", best_model.feature_importances_)

    return best_model

# 主程序
if __name__ == "__main__":
    # 加载数据
    data = load_data("./data/combinations.json")

    # 提取特征和标签
    features, labels, years = extract_features_and_labels(data)

    # 训练模型
    model = train_model(features, labels, years)