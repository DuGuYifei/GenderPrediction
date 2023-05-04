import pandas as pd
from sklearn.preprocessing import LabelEncoder


def label_encoder(df):
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = df[c].fillna('N')
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)
        elif df[c].dtype.name == 'category':
            df[c] = df[c].cat.codes
    return df


def handle_data(df):
    df['Age'] = df['Age'].astype(int)
    df['Height(cm)'] = df['Height(cm)'].astype(int)
    df['Weight(kg)'] = df['Weight(kg)'].astype(int)
    df['Income(USD)'] = df['Income(USD)'].astype(int)
    # 将年龄每5岁分一组
    df['age_group'] = pd.cut(df['Age'], bins=range(0, 101, 5), right=False, labels=range(20))
    # 将身高每5cm分一组
    df['height_group'] = pd.cut(df['Height(cm)'], bins=range(100, 226, 5), right=False, labels=range(25))
    # 将体重每5kg分一组
    df['weight_group'] = pd.cut(df['Weight(kg)'], bins=range(40, 156, 5), right=False, labels=range(23))
    # 将收入每5000分一组
    df['income_group'] = pd.cut(df['Income(USD)'], bins=range(0, 500001, 5000), right=False, labels=range(100))
    # 删除原始特征
    df.drop(['Age', 'Height(cm)', 'Weight(kg)', 'Income(USD)'], axis=1, inplace=True)
    return label_encoder(df.copy())
