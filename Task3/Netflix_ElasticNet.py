import urllib.request
import os
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_dir, "netflix_titles.csv")

url = "https://raw.githubusercontent.com/aniketanraje/assignment-dataset/main/netflix_titles.csv"

if not os.path.exists(save_path):
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"数据下载完成，已保存至：{save_path}")
    except Exception as e:
        print("下载失败：", e)
        raise
else:
    print(f"数据文件已存在，跳过下载：{save_path}")

df = pd.read_csv(save_path)
print(f"原始数据共 {df.shape[0]} 行、{df.shape[1]} 列。")

#数据概览
print("\n数据概览")
print(f"列名包括：{', '.join(df.columns.tolist())}。")
missing = df.isnull().sum()
missing_desc = "；".join([f"{col} 缺失 {int(n)} 条" for col, n in missing.items() if n > 0])
print(f"存在缺失值的字段有：{missing_desc}。")
type_counts = df['type'].value_counts()
type_desc = "；".join([f"{k} {int(v)} 条" for k, v in type_counts.items()])
print(f"内容类型分布为：{type_desc}。")
print(f"duration 字段的前几个示例值为：{', '.join(df['duration'].head(5).astype(str).tolist())}。")

#清洗 duration
def parse_duration(row):
    val = str(row['duration'])
    if row['type'] == 'Movie':
        if 'min' in val:
            return float(val.replace(' min', '').strip())
        return np.nan
    else:
        if 'Season' in val:
            return float(val.split(' ')[0])
        return np.nan

df['duration_num'] = df.apply(parse_duration, axis=1)
df_clean = df.dropna(subset=['duration_num', 'release_year']).copy()
df_clean = df_clean[df_clean['duration_num'] > 0]

print("\n数据清洗")
print(f"将 duration 拆分为数值后，有效样本共 {len(df_clean)} 条。")
for t in df_clean['type'].unique():
    sub = df_clean[df_clean['type'] == t]['duration_num']
    unit = "分钟" if t == "Movie" else "季"
    print(f"{t} 类共 {len(sub)} 条，平均时长 {sub.mean():.2f} {unit}，"
          f"最短 {sub.min():.0f} {unit}，最长 {sub.max():.0f} {unit}。")

#构造特征矩阵（以电影为例）
movies = df_clean[df_clean['type'] == 'Movie'].copy()
rating_dummies = pd.get_dummies(movies['rating'], prefix='rating', drop_first=True)

X = pd.concat([
    movies[['release_year']].reset_index(drop=True),
    rating_dummies.reset_index(drop=True)
], axis=1)
y = movies['duration_num'].values

print("\n特征构造")
print(f"选择电影子集建模，样本数 {len(movies)} 条，目标变量为电影时长（分钟）。")
print(f"特征矩阵共 {X.shape[1]} 个特征，包括 release_year 以及 rating 的 {rating_dummies.shape[1]} 个哑变量。")
print(f"目标变量（电影时长）范围为 {y.min():.0f} 到 {y.max():.0f} 分钟，均值 {y.mean():.2f} 分钟。")

#划分 + 标准化 + 网格搜索
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
}

enet = ElasticNet(max_iter=10000, random_state=42)
grid_search = GridSearchCV(
    enet, param_grid, cv=5,
    scoring='neg_mean_squared_error', n_jobs=-1, verbose=0
)
grid_search.fit(X_train_scaled, y_train)

best_params = grid_search.best_params_
best_cv_mse = -grid_search.best_score_

print("\n模型训练")
print(f"训练集 {len(X_train)} 条，测试集 {len(X_test)} 条。")
print(f"特征已做标准化处理。网格搜索在 alpha 与 l1_ratio 的 {len(param_grid['alpha']) * len(param_grid['l1_ratio'])} 种组合上进行 5 折交叉验证。")
print(f"最优参数为 alpha = {best_params['alpha']}，l1_ratio = {best_params['l1_ratio']}，"
      f"对应交叉验证均方误差为 {best_cv_mse:.2f}（即 RMSE 约 {np.sqrt(best_cv_mse):.2f} 分钟）。")

#测试集评估
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
nonzero = int(np.sum(best_model.coef_ != 0))
total_feat = len(best_model.coef_)

print("\n模型评估")
print(f"在测试集上，模型的均方根误差 RMSE 为 {rmse:.2f} 分钟，"
      f"决定系数 R² 为 {r2:.4f}。")
print(f"这意味着模型对电影时长的解释力较弱，R² 接近 0 说明 release_year 与 rating 这两类特征"
      f"对电影时长的线性预测能力有限。这是数据本身的特性，而非建模流程的问题。")
print(f"Elastic Net 共保留 {nonzero} 个非零系数（总特征 {total_feat} 个），"
      f"其余 {total_feat - nonzero} 个特征被正则化压缩为 0，体现了 Elastic Net 的特征选择作用。")

#特征系数文字叙述
feature_names = X.columns.tolist()
coef_pairs = [(name, coef) for name, coef in zip(feature_names, best_model.coef_) if coef != 0]
coef_pairs.sort(key=lambda x: x[1], reverse=True)

print("\n特征系数解读")
if coef_pairs:
    print("被模型保留的非零特征及其系数如下（按系数从大到小排列）：")
    for name, coef in coef_pairs:
        direction = "正相关" if coef > 0 else "负相关"
        print(f"特征 {name} 的系数为 {coef:.4f}，与电影时长呈{direction}关系。")
    top_feat, top_coef = coef_pairs[0]
    print(f"其中影响最大的是 {top_feat}，系数 {top_coef:.4f}。")
else:
    print("所有特征系数均被压缩为 0，说明在当前正则化强度下没有特征对目标有稳定贡献。")

zero_feats = [name for name, coef in zip(feature_names, best_model.coef_) if coef == 0]
if zero_feats:
    print(f"被压缩为 0 的特征共 {len(zero_feats)} 个，包括：{', '.join(zero_feats)}。")

#总结
print("\n分析总结")
print(f"本次分析使用 Elastic Net 回归，以 Netflix 电影数据为对象，"
      f"用上映年份 release_year 和内容分级 rating 预测电影时长。"
      f"经过标准化与网格搜索调参，最优模型为 alpha = {best_params['alpha']}、"
      f"l1_ratio = {best_params['l1_ratio']}，测试集 RMSE 为 {rmse:.2f} 分钟，R² 为 {r2:.4f}。")
print(f"Elastic Net 从 {total_feat} 个特征中保留了 {nonzero} 个非零系数，"
      f"说明它同时具备 L1 的稀疏选择能力和 L2 的稳定性。")
print("R² 偏低表明仅靠年份和分级难以解释电影时长，后续可引入国家、类型标签等文本特征以提升效果。")
print("\n全部分析完成。")