import json
import numpy as np
from sklearn.model_selection import train_test_split
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
    return np.array(features), np.array(labels)

# 训练随机森林模型
def train_model(features, labels):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 数据平衡
    over_sampler = BorderlineSMOTE(random_state=42)
    under_sampler = RandomUnderSampler(random_state=42)
    pipeline = Pipeline([
        ('over', over_sampler),
        ('under', under_sampler)
    ])
    X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train_scaled, y_train)

    # 初始化并训练模型
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train_resampled, y_train_resampled)

    # 评估模型
    y_pred = model.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

    # 输出模型内容
    print("\nModel Details:")
    print("Number of trees in the forest:", model.n_estimators)
    print("Depth of the forest:", model.max_depth)
    print("Minimum samples split:", model.min_samples_split)
    print("Minimum samples leaf:", model.min_samples_leaf)
    print("Class weights:", model.class_weight)
    print("Feature importances:", model.feature_importances_)

    return model

# 主程序
if __name__ == "__main__":
    # 加载数据
    data = load_data("./data/combinations.json")

    # 提取特征和标签
    features, labels = extract_features_and_labels(data)

    # 训练模型
    model = train_model(features, labels)