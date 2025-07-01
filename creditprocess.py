import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings

warnings.filterwarnings('ignore')


# ======================
# 1. 数据加载与列名定义（基于原始German Credit Data）
# ======================
def load_german_credit(file_path):
    """加载原始German Credit数据集并添加列名"""
    # 定义原始数据集的列名（依据UCI文档）
    column_names = [
        'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
        'savings_account', 'employment', 'installment_rate', 'personal_status_sex',
        'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
        'housing', 'existing_credits', 'job', 'dependents', 'telephone', 'foreign_worker',
        'credit_risk'
    ]

    # 加载数据（原始数据为空格分隔，无表头）
    df = pd.read_csv(file_path, sep='\s+', header=None, names=column_names)

    # 目标变量编码：1=Good（良好），2=Bad（不良） → 转换为1和0
    df['credit_risk'] = df['credit_risk'].map({1: 1, 2: 0})

    return df


# ======================
# 2. 特征工程与预处理
# ======================
def preprocess_features(df):
    """对原始German Credit特征进行预处理"""
    # 定义分类特征（13个）和数值特征（7个）
    categorical_features = [
        'checking_account', 'credit_history', 'purpose', 'savings_account',
        'employment', 'personal_status_sex', 'other_debtors', 'property',
        'other_installment_plans', 'housing', 'job', 'telephone', 'foreign_worker'
    ]

    numerical_features = [
        'duration', 'credit_amount', 'installment_rate',
        'residence_since', 'age', 'existing_credits', 'dependents'
    ]

    # 创建预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    # 分离特征与目标变量
    X = df.drop('credit_risk', axis=1)
    y = df['credit_risk'].values

    # 应用预处理
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor


# ======================
# 3. 数据集划分与保存
# ======================
def save_datasets(X, y, test_size=0.3, random_state=42):
    """划分训练集/测试集并保存为TXT文件"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 保存为TXT（保留4位小数）
    np.savetxt('trainx.txt', X_train, fmt='%.4f')
    np.savetxt('trainy.txt', y_train, fmt='%d')
    np.savetxt('testx.txt', X_test, fmt='%.4f')
    np.savetxt('testy.txt', y_test, fmt='%d')

    print(f"数据集已保存：")
    print(f"- 训练特征：trainx.txt ({X_train.shape[0]}样本, {X_train.shape[1]}特征)")
    print(f"- 测试特征：testx.txt ({X_test.shape[0]}样本)")


# ======================
# 主程序
# ======================
if __name__ == "__main__":
    # 配置参数
    DATA_PATH = 'german.data'  # 原始文件名（需与下载文件一致）

    # 1. 加载数据
    print("步骤1: 加载原始German Credit数据...")
    df = load_german_credit(DATA_PATH)
    print(f"数据维度：{df.shape}，目标变量分布：\n{df['credit_risk'].value_counts()}")

    # 2. 预处理特征
    print("步骤2: 特征预处理（标准化数值特征 + 独热编码分类特征）...")
    X_processed, y, preprocessor = preprocess_features(df)

    # 3. 划分并保存
    print("步骤3: 划分训练集/测试集并保存...")
    save_datasets(X_processed, y)
    print("处理完成！文件可直接用于机器学习模型")