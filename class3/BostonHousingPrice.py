# ============ Boston 房价：Ridge / Lasso / Elastic Net 对比（纯文字展示版） ============
# 特点：结果只在屏幕上用文字显示 —— 不保存任何表格文件、不画图
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score

line = "=" * 58
print(line)
print("Boston 房价数据：Ridge / Lasso / Elastic Net 对比")
print(line)

# ---------- 1. 读取本地数据集（自动定位到本文件所在文件夹，不会找不到文件） ----------
current_dir = os.path.dirname(os.path.abspath(__file__))
raw_df = pd.read_csv(os.path.join(current_dir, "BostonHousing.csv"))

X = raw_df.drop("medv", axis=1)      # 13 个特征
y = raw_df["medv"].values            # 要预测的房价中位数
feature_names = X.columns.tolist()

print(f"数据规模：{X.shape[0]} 个样本，{X.shape[1]} 个特征")
print(f"特征：{', '.join(feature_names)}")

# ---------- 2. 划分训练集 / 测试集 ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"划分结果：训练集 {X_train.shape[0]} 个样本，测试集 {X_test.shape[0]} 个样本")

# ---------- 3. Z-score 标准化（只在训练集上算均值和方差） ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("预处理：已做 Z-score 标准化（每个特征均值为 0、标准差为 1）")

# ---------- 4. 5 折交叉验证挑选最优正则化强度 ----------
print("\n正在用 5 折交叉验证挑选最优正则化强度 ...")
ridge_cv = RidgeCV(alphas=np.logspace(-2, 3, 100), cv=5).fit(X_train_scaled, y_train)
lasso_cv = LassoCV(alphas=np.logspace(-3, 3, 100), cv=5, max_iter=100000).fit(X_train_scaled, y_train)
elastic_cv = ElasticNetCV(alphas=np.logspace(-3, 3, 100),
                          l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                          cv=5, max_iter=100000).fit(X_train_scaled, y_train)

print("\n【最优正则化强度 alpha】")
print(f"  Ridge        alpha = {ridge_cv.alpha_:.3f}")
print(f"  Lasso        alpha = {lasso_cv.alpha_:.4f}")
print(f"  Elastic Net  alpha = {elastic_cv.alpha_:.4f}   （L1 占比 = {elastic_cv.l1_ratio_:.1f}）")

# ---------- 5. 在测试集上评估，结果直接文字输出 ----------
models = [("Ridge", ridge_cv), ("Lasso", lasso_cv), ("Elastic Net", elastic_cv)]

print("\n【测试集表现】")
print(f"  {'模型':<10}{'MSE':>8}{'R²':>9}{'非零特征数':>11}")
rows = []
for name, model in models:
    pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    nz = int(np.sum(np.abs(model.coef_) > 1e-8))   # 系数不为 0 的特征个数
    nz_str = str(nz) + "/" + str(len(feature_names))
    rows.append((name, mse, r2, nz))
    print(f"  {name:<13}{mse:>8.2f}{r2:>9.4f}{nz_str:>12}")

# ---------- 6. 结果速览 ----------
best = max(rows, key=lambda r: r[2])
print("\n【结果】")
print(f"  精度最高：{best[0]}，测试集 R² = {best[2]:.4f}，MSE = {best[1]:.2f}")
print(f"  三个模型的 R² 差距：{max(r[2] for r in rows) - min(r[2] for r in rows):.4f}")
print(f"  非零特征数：Ridge {rows[0][3]} 个，Lasso {rows[1][3]} 个，Elastic Net {rows[2][3]} 个")