
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pasta.base.annotate import space_left
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os


# 设置中文显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

import warnings

# 忽略所有警告
warnings.filterwarnings('ignore')

import pandas as pd
import os
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class LogisticRegressionHonest:

    def __init__(self, calc_cols, model_features, train_data_path, pred_data_path, save_path, threshold):
        '''
        init对象
        :param calc_cols:
        :param model_features:
        :param train_data_path:
        :param pred_data_path:
        :param save_path:
        '''
        self.calc_cols = calc_cols
        self.model_features = model_features
        self.train_data_path = train_data_path
        self.pred_data_path = pred_data_path
        self.save_path = save_path
        self.threshold = threshold

    @staticmethod
    def read_check_file_data(file_path):
        """
        读取excel数据
        :param file_path:
        :return:
        """
        try:
            df_train = pd.read_excel(file_path)
            # 采购端成本差异
            model_features_check = model_features[:-1]
            missing_check_col = [col for col in model_features_check if col not in df_train.columns]
            if missing_check_col:
                print(f"错误：训练数据缺少必需列：{missing_check_col}")
                return
            print(f"训练数据原始形状：{df_train.shape}")
            print("\n训练数据列名：")
            print(df_train.columns.tolist())
            return df_train
        except Exception as e:
            print(f"读取文件 {file_path} 失败：{e}")
            return None

    def process_data(self, df_train):
        """
        数据预处理
        :param df_train:
        :return:
        """
        # 步骤1：提前校验计算必需列（移出循环，仅执行一次，避免冗余）
        missing_calc_cols = [col for col in self.calc_cols if col not in df_train.columns]
        if missing_calc_cols:
            raise ValueError(f"训练数据缺少采购端成本差异计算必需列：{missing_calc_cols}")

        cost_variance_null = df_train['采购端成本差异'].isna()
        # 无缺失值直接返回
        if not cost_variance_null.any():
            return df_train
        # 正确获取缺失值的行索引（两种方式二选一，推荐pandas原生索引）
        # 方式2：pandas原生索引（更简洁、可读性更高，推荐）
        need_handle_rows = df_train[cost_variance_null].index
        # 步骤3：矢量化计算分子和分母（避免逐行循环，大幅提升效率）
        # 定义全球采购相关成本列
        global_cost_cols = [
            '全球采购物料价格', '全球物流CPB影响', '全球库存成本',
            '全球税损', '全球质量风险'
        ]
        # 定义本地采购相关成本列
        local_cost_cols = [
            '本地采购物料价格', '本地物流CPB影响', '本地库存成本',
            '本地税损', '本地质量风险'
        ]

        # 矢量化计算全球总成本（分母）和分子
        global_total = df_train.loc[need_handle_rows, global_cost_cols].sum(axis=1)
        local_total = df_train.loc[need_handle_rows, local_cost_cols].sum(axis=1)
        numerator = global_total - local_total
        denominator = global_total

        # 步骤4：矢量化赋值（合并条件，高效处理特殊情况）
        # 条件：分母为0 或 全球采购物料价格缺失 → 赋值-1，否则赋值分子/分母
        global_material_price = df_train.loc[need_handle_rows, '全球采购物料价格']
        valid_condition = (denominator != 0) & (~global_material_price.isna())
        df_train.loc[need_handle_rows, '采购端成本差异'] = np.where(
            valid_condition,
            numerator / denominator,
            -1
        )
        # 按规则计算采购端成本差异：全球采购为空填-1，否则（分子/分母）
        df_train = df_train[model_features].fillna(0)
        print(f"自变量空值处理：所有空值已用0填充\n")

        return df_train


    def train_model(self, df_train):
        """
        训练模型
        :param df_train:
        :return:
        """

        # 步骤2：处理模型自变量的空值（其他列空值用0填充）
        # 提取模型自变量列
        X_train = df_train[model_features].copy()
        # 用0填充所有空值
        X_train = X_train.fillna(0)
        print(f"\n模型自变量空值处理完成：所有空值已用0填充")

        # 步骤3：提取目标变量y（假设y列为'y'，若列名不同需修改）
        if 'Local Buy决策' not in df_train.columns:
            raise ValueError("训练数据缺少目标变量列'y'")
        y_train = df_train['Local Buy决策']

        # 划分训练集和测试集
        X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # 构建并训练模型
        model = LogisticRegression(max_iter=1000)  # 增加迭代次数避免收敛警告
        model.fit(X_train_split, y_train_split)

        # 应用阈值（按你之前的0.58阈值，可调整）

        y_proba_test = model.predict_proba(X_test_split)[:, 1]  # y=1的概率
        y_pred_test = (y_proba_test >= self.threshold).astype(int)

        # 评估模型性能
        accuracy = accuracy_score(y_test_split, y_pred_test)
        precision = precision_score(y_test_split, y_pred_test, zero_division=0)
        recall = recall_score(y_test_split, y_pred_test, zero_division=0)
        f1 = f1_score(y_test_split, y_pred_test, zero_division=0)

        # 输出训练结果
        print(f"\n================ 模型训练性能（阈值={self.threshold}） ================")
        print(f"模型准确率：{accuracy:.4f}")
        print(f"模型精确率：{precision:.4f}")
        print(f"模型召回率：{recall:.4f}")
        print(f"模型 F1 分数：{f1:.4f}")
        print(f"模型自变量列（共{len(model_features)}个）：{model_features}")

        return model, accuracy, precision, recall, f1

    @staticmethod
    def eval_feature(model):
        """
        评估模型重要性
        :param model:
        :return:
        """
        # ---------------------- 2. 计算特征重要性（逻辑回归系数绝对值） ----------------------
        # 提取模型系数（每个特征对应一个系数）
        feature_coef = model.coef_[0]
        # 计算特征重要性（系数绝对值，消除正负影响）
        feature_importance = pd.DataFrame({
            '特征名称': model_features,
            '系数': feature_coef,
            '重要性': np.abs(feature_coef)  # 重要性=系数绝对值
        })
        # 按重要性降序排序（便于可视化）
        feature_importance_sorted = feature_importance.sort_values('重要性', ascending=False).reset_index(drop=True)

        print("特征重要性排序（按重要性降序）：")
        print(feature_importance_sorted[['特征名称', '系数', '重要性']])

        # ---------------------- 3. 可视化特征重要性（柱状图） ----------------------
        # 设置中文字体（避免乱码）
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建画布（适中尺寸，便于查看）
        fig, ax = plt.subplots(figsize=(10, 6))

        # 定义颜色：突出“采购端成本差异”（用红色，其他用蓝色）
        colors = ['#FF6B6B' if feat == '采购端成本差异' else '#4ECDC4' for feat in
                  feature_importance_sorted['特征名称']]

        # 绘制柱状图
        bars = ax.bar(
            x=range(len(feature_importance_sorted)),
            height=feature_importance_sorted['重要性'],
            color=colors,
            alpha=0.8,  # 透明度，避免过于刺眼
            edgecolor='#2C3E50',  # 柱子边框颜色，增强轮廓
            linewidth=0.8
        )

        # 设置x轴标签（特征名称，旋转45度避免重叠）
        ax.set_xticks(range(len(feature_importance_sorted)))
        ax.set_xticklabels(feature_importance_sorted['特征名称'], rotation=45, ha='right', fontsize=10)

        # 设置y轴和标题
        ax.set_ylabel('特征重要性（系数绝对值）', fontsize=12, fontweight='bold')
        ax.set_title('逻辑回归模型 - 特征重要性排序', fontsize=14, fontweight='bold', pad=20)

        # 在每个柱子上添加数值标签（显示重要性具体值）
        for i, (bar, importance) in enumerate(zip(bars, feature_importance_sorted['重要性'])):
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # x坐标（柱子中心）
                bar.get_height() + 0.01,  # y坐标（柱子顶部+微小偏移，避免重叠）
                f'{importance:.4f}',  # 显示的数值（保留4位小数）
                ha='center', va='bottom', fontsize=9
            )

        # 添加网格线（y轴方向，便于对比数值）
        ax.yaxis.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)  # 网格线置于柱子下方

        # 调整布局（避免标签被截断）
        plt.tight_layout()

        print(f"\n关键结论：")
        print(f"1. 最重要的3个特征：{', '.join(feature_importance_sorted['特征名称'][:3])}")
        print(
            f"2. 采购端成本差异的重要性排名：第{feature_importance_sorted[feature_importance_sorted['特征名称'] == '采购端成本差异'].index[0] + 1}位")
        print(
            f"3. 采购端成本差异的重要性值：{feature_importance_sorted[feature_importance_sorted['特征名称'] == '采购端成本差异']['重要性'].values[0]:.4f}")
        plt.show()

    def predict_with_model(self, model):
        """
        使用模型进行预测
        :param model:
        :return:
        """

        df_new = pd.read_excel(self.pred_data_path)
        df_new = self.process_data(df_new)
        original_cols = df_new.columns.tolist()
        print(f"新数据原始信息：")
        print(f"形状：{df_new.shape}")
        print(f"列名：{original_cols}\n")

        X_new = df_new[model_features].fillna(0)
        print(f"自变量空值处理：所有空值已用0填充\n")

        # 步骤3：预测并生成结果
        y_new_proba = model.predict_proba(X_new)[:, 1]
        y_new_pred = (y_new_proba >= self.threshold).astype(int)

        # 合并结果（保留原始列，新增预测相关列）
        df_new_result = df_new.copy()
        df_new_result['预测y值'] = y_new_pred
        df_new_result['y=1的概率'] = y_new_proba.round(4)
        return df_new_result

    def save_pred_data(self, df_new_result):
        """
        保存结果
        :param df_new_result:
        :return:
        """
        # 步骤4：保存结果
        df_new_result.to_excel(self.save_path, index=False)
        # 输出结果预览
        print("================ 预测结果预览（前10行关键列） ================")
        preview_cols = ['预测y值', 'y=1的概率']
        print(df_new_result[preview_cols].head(10))
        print(f"\n预测完成！结果文件保存至：{self.save_path}")
        print(f"结果包含列：{df_new_result.columns.tolist()}")

    def train_and_predict(self):
        """
        训练和预测模型
        :return:
        """
        train_data = LogisticRegressionHonest.read_check_file_data(self.train_data_path)
        process_train_data = self.process_data(train_data)

        model, accuracy, precision, recall, f1 = self.train_model(process_train_data)
        #LogisticRegressionHonest.eval_feature(model)
        df_new_result = self.predict_with_model(model)
        self.save_pred_data(df_new_result)
        # 输出结果预览
        print("================ 预测结果预览（前10行关键列） ================")
        preview_cols = ['预测y值', 'y=1的概率']
        print(df_new_result[preview_cols].head(10))
        print(f"\n预测完成！结果文件保存至：{self.save_path}")
        print(f"结果包含列：{df_new_result.columns.tolist()}")

    def predict(self):
        return None


if __name__ == '__main__':
    # 类对象
    # 1.公式计算所需列（用于采购端成本差异）
    calc_cols = [
        '全球采购物料价格',
        '全球物流CPB影响',
        '全球库存成本',
        '全球税损',
        '全球质量风险',
        '本地采购物料价格',
        '本地物流CPB影响',
        '本地库存成本',
        '本地税损',
        '本地质量风险',
    ]
    # 2. 模型自变量
    model_features = [
        '政策适配性',
        '管理层决策',
        'JIT需求',
        '采购端时效差异',
        '资源基础成熟度',
        '供应商生态成熟度',
        '供应商数量',
        '采购端成本差异'
    ]
    # 训练数据集地址
    train_file_path = r"C:\Users\honest.zhao\Desktop\AI\采购模型\LR_data\20251231\预测数据\trainData20251231.xlsx"
    # 预测数据集地址
    pred_file_path = r"C:\Users\honest.zhao\Desktop\AI\采购模型\LR_data\20251231\预测数据\新-ME越南采购模型20251231.xlsx"
    # 预测结果保存地址
    save_file_path = r"C:\Users\honest.zhao\Desktop\AI\采购模型\LR_data\20251231\预测数据\预测结果.xlsx"
    #threshold = 0.58
    threshold = 0.50
    lr = LogisticRegressionHonest(calc_cols, model_features, train_file_path, pred_file_path, save_file_path, threshold)
    lr.train_and_predict()
