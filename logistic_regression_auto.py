# -*- coding: utf-8 -*-
"""
逻辑回归 - 自动检测版
导入 Excel 后，自动识别哪些列是自变量 X、哪个列是目标变量 Y，然后训练并预测。

自动检测规则：
0. 智能表头：自动跳过 Excel 前置的说明/凡例行，定位真正的表头行，
   并对合并单元格造成的空表头向上回填（普通首行表头也照常工作）。
1. 目标变量 Y（在所有二值列中按优先级选）：
   - 最高优先：列名精确等于 y / target / label / 标签 / 目标 等；
   - 次高优先：列名包含高辨识度关键词（如 Local Buy决策）；
   - 兜底：取最右侧的二值列。
   - 非 0/1 的二值列（是/否、Yes/No、Y/N 等）会自动映射成 0/1。
2. 自变量 X：
   - Y 以外的所有“数值型”列自动作为 X；
   - 自动排除文本列、ID/编号/名称类列、以及取值恒定的列。
3. 采购端成本差异（可选逻辑）：
   - 若全球/本地成本列齐全，则按原公式计算缺失值；
   - 若列不齐，则自动跳过，保证任意 Excel 都能运行。
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

# matplotlib 为可选依赖：仅在需要画图时才用。没安装也不影响训练/预测/网页版。
try:
    import matplotlib.pyplot as plt
    # 中文显示（找不到字体时自动忽略，不影响计算）
    for _f in ['SimHei', 'WenQuanYi Zen Hei', 'Microsoft YaHei', 'Arial Unicode MS']:
        try:
            plt.rcParams['font.sans-serif'] = [_f]
            break
        except Exception:
            continue
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    plt = None


# ==================== 自动检测相关配置 ====================
# 目标变量 Y：精确列名（不区分大小写，完全相等才算）——最高优先级
TARGET_EXACT = ['y', 'target', 'label', '标签', '目标', '目标变量']
# 目标变量 Y：高辨识度关键词（包含即匹配）——次高优先级
# 注意：不使用泛化的 '决策'，因为它会误伤特征列 '管理层决策'
TARGET_KEYWORDS = ['buy决策', 'local buy', 'localbuy', 'local_buy', '本地化决策', '是否本地', 'y/n']
# 需要排除、不作为自变量的列名关键词（ID / 编号类）
ID_KEYWORDS = ['id', '编号', '序号', '编码', 'code', 'name', '名称', '料号', '物料号', '责任人', '类别']
# 常见的二值文本映射
BINARY_MAP = {
    '是': 1, '否': 0, 'yes': 1, 'no': 0, 'y': 1, 'n': 0,
    'true': 1, 'false': 0, 'local': 1, 'global': 0, '本地': 1, '全球': 0,
    '通过': 1, '不通过': 0, '成功': 1, '失败': 0,
}


def _to_binary_series(s):
    """尝试把一列转成 0/1 二值。成功返回转换后的 Series，失败返回 None。"""
    vals = s.dropna().unique()
    if len(vals) != 2:
        return None
    # 已经是数值 0/1
    try:
        num = pd.to_numeric(s, errors='raise')
        uniq = set(num.dropna().unique())
        if uniq.issubset({0, 1}):
            return num.astype('Int64')
        # 其它两种数值 → 映射为 0/1（小的为0，大的为1）
        lo, hi = sorted(uniq)
        return num.map({lo: 0, hi: 1}).astype('Int64')
    except Exception:
        pass
    # 文本二值 → 按映射表转换
    lowered = [str(v).strip().lower() for v in vals]
    if all(v in BINARY_MAP for v in lowered):
        return s.astype(str).str.strip().str.lower().map(BINARY_MAP).astype('Int64')
    # 未知文本二值：按出现顺序映射为 0/1
    mapping = {vals[0]: 0, vals[1]: 1}
    return s.map(mapping).astype('Int64')


def _is_number(v):
    """判断单个值是否是数字（用于识别表头行：表头多为文字）。"""
    try:
        float(str(v).replace(',', '').strip())
        return True
    except Exception:
        return False


def _numeric_frac(series):
    """一列中能转成数值的比例（用于判断某行是否像真正的数据行）。"""
    n = series.notna().sum()
    if n == 0:
        return 0.0
    num = pd.to_numeric(series, errors='coerce')
    return num.notna().sum() / n


def smart_read_excel(file_path, max_scan=25, sheet_name=0):
    """
    智能读取 Excel：自动跳过前置的说明/凡例行，定位真正的表头行，
    并对合并单元格造成的空表头向上回填，最后清理无名/空列。
    对普通规整的表格（表头在第一行）也能正常工作。
    :param sheet_name: 要读取的工作表（名称或序号，默认第 1 个）
    """
    raw = pd.read_excel(file_path, header=None, sheet_name=sheet_name)
    if raw.empty:
        return pd.DataFrame()

    def _row_num_frac(row):
        nn = [v for v in row if pd.notna(v)]
        return (sum(1 for v in nn if _is_number(v)) / len(nn)) if nn else 0.0

    n_rows = len(raw)
    best_r, best_score = None, -1     # 满足“文字表头+下一行是数据”的最佳行
    fb_r, fb_score = 0, -1            # 兜底：单纯下方数值列最多的行
    for r in range(min(max_scan, n_rows - 1)):
        data = raw.iloc[r + 1:]
        # 该行作为表头时，下方为“数值列”的数量
        numeric_below = sum(1 for c in range(raw.shape[1])
                            if data[c].notna().sum() >= 3 and _numeric_frac(data[c]) >= 0.6)
        # 表头行本身应以文字为主（字段名是文字）
        hdr = raw.iloc[r].dropna()
        header_text_frac = (sum(1 for v in hdr if not _is_number(v)) / len(hdr)) if len(hdr) else 0.0
        # 紧接的下一行应是数据行（以数字为主）——排除“说明行/分组标题行”
        next_num_frac = _row_num_frac(raw.iloc[r + 1])
        if numeric_below > fb_score:
            fb_score, fb_r = numeric_below, r
        if (numeric_below >= 2 and header_text_frac >= 0.5 and next_num_frac >= 0.5
                and numeric_below > best_score):
            best_score, best_r = numeric_below, r

    # 没有合适候选时退回兜底行
    if best_r is None:
        best_r = fb_r

    # 构造表头：空表头向上回填（处理合并单元格：标签在上一行）
    header = []
    for c in range(raw.shape[1]):
        name = raw.iat[best_r, c]
        rr = best_r
        while (pd.isna(name) or str(name).strip() == '') and rr > 0:
            rr -= 1
            name = raw.iat[rr, c]
        header.append(str(name).replace('\n', ' ').strip() if pd.notna(name) else f'列{c}')

    df = raw.iloc[best_r + 1:].copy()
    df.columns = header
    df = df.reset_index(drop=True)

    # 清理：删掉整列全空的列，以及表头为空/占位名的列
    df = df.dropna(axis=1, how='all')
    drop_cols = [c for c in df.columns
                 if str(c).strip().lower() in ('', 'nan', 'none')
                 or str(c).startswith('列') and df[c].notna().sum() == 0]
    df = df.drop(columns=drop_cols, errors='ignore')
    # 删掉整行全空的行
    df = df.dropna(axis=0, how='all').reset_index(drop=True)

    if best_r > 0:
        print(f"智能表头识别：检测到前 {best_r} 行为说明/凡例，已自动定位第 {best_r + 1} 行为表头。")
    return df


def auto_detect_xy(df, target_col=None, feature_cols=None):
    """
    自动检测自变量 X 和目标变量 Y。
    :param df: 原始 DataFrame
    :param target_col: 可手动指定 Y 列名（None 则自动检测）
    :param feature_cols: 可手动指定 X 列名列表（None 则自动检测）
    :return: (target_name, feature_list)
    """
    cols = list(df.columns)

    # ---------- 1. 确定目标变量 Y ----------
    if target_col is not None:
        if target_col not in cols:
            raise ValueError(f"指定的目标列 '{target_col}' 不在数据中。现有列：{cols}")
        y_name = target_col
    else:
        y_name = None
        binary_cols = [c for c in cols if _to_binary_series(df[c]) is not None]
        if not binary_cols:
            raise ValueError(
                "未能自动识别目标变量 Y：数据中没有找到二值（两种取值）的列。\n"
                "请通过 target_col 参数手动指定目标列。"
            )
        # 1a. 最高优先：列名与精确目标名完全相等（如 y / label / target）
        for c in binary_cols:
            if str(c).strip().lower() in TARGET_EXACT:
                y_name = c
                break
        # 1b. 次高优先：列名包含高辨识度关键词（如 Local Buy决策）
        if y_name is None:
            for c in binary_cols:
                name = str(c).lower()
                if any(k.lower() in name for k in TARGET_KEYWORDS):
                    y_name = c
                    break
        # 1c. 兜底：取最后一个二值列（Y 通常在最右侧）
        if y_name is None:
            y_name = binary_cols[-1]
            if len(binary_cols) > 1:
                print(f"提示：检测到多个二值列 {binary_cols}，已自动选用最后一个 '{y_name}' 作为 Y。")
                print(f"      如需指定其它列，请设置 target_col 参数。")

    # ---------- 2. 确定自变量 X ----------
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in cols]
        if missing:
            raise ValueError(f"指定的自变量列不存在：{missing}")
        x_list = [c for c in feature_cols if c != y_name]
    else:
        x_list = []
        for c in cols:
            if c == y_name:
                continue
            name = str(c).lower()
            # 排除 ID / 名称类列
            if any(k.lower() in name for k in ID_KEYWORDS):
                continue
            # 只保留数值型列
            col_num = pd.to_numeric(df[c], errors='coerce')
            if col_num.notna().sum() == 0:
                continue  # 完全无法转成数值 → 文本列，跳过
            # 排除取值恒定的列（无信息量）
            if col_num.nunique(dropna=True) <= 1:
                continue
            x_list.append(c)

    if not x_list:
        raise ValueError("未能自动识别任何自变量 X（数值列）。请检查数据或手动指定 feature_cols。")

    return y_name, x_list


class LogisticRegressionAuto:

    # 采购端成本差异计算所需列
    GLOBAL_COST_COLS = ['全球采购物料价格', '全球物流CPB影响', '全球库存成本', '全球税损', '全球质量风险']
    LOCAL_COST_COLS = ['本地采购物料价格', '本地物流CPB影响', '本地库存成本', '本地税损', '本地质量风险']
    COST_DIFF_COL = '采购端成本差异'

    def __init__(self, train_data_path, pred_data_path, save_path,
                 threshold=0.5, target_col=None, feature_cols=None):
        """
        :param train_data_path: 训练数据 Excel 路径
        :param pred_data_path:  预测数据 Excel 路径（可为 None，仅训练）
        :param save_path:       预测结果保存路径
        :param threshold:       判定为 1 的概率阈值
        :param target_col:      手动指定 Y 列名（默认 None = 自动检测）
        :param feature_cols:    手动指定 X 列名列表（默认 None = 自动检测）
        """
        self.train_data_path = train_data_path
        self.pred_data_path = pred_data_path
        self.save_path = save_path
        self.threshold = threshold
        self.target_col = target_col
        self.feature_cols = feature_cols
        # 训练时确定，供预测复用
        self.model_features = None
        self.target_name = None

    # ---------------- 读取 ----------------
    @staticmethod
    def read_excel(file_path):
        try:
            df = smart_read_excel(file_path)
            print(f"读取成功：{os.path.basename(file_path)}  形状={df.shape}")
            print(f"列名：{df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"读取文件失败 {file_path}：{e}")
            return None

    # ---------------- 采购端成本差异（可选） ----------------
    def _fill_cost_diff(self, df):
        """若成本列齐全且存在成本差异列的缺失，则按原公式补算；否则跳过。"""
        has_cost_cols = all(c in df.columns for c in self.GLOBAL_COST_COLS + self.LOCAL_COST_COLS)
        if self.COST_DIFF_COL not in df.columns or not has_cost_cols:
            return df  # 不具备条件，直接跳过
        null_mask = df[self.COST_DIFF_COL].isna()
        if not null_mask.any():
            return df
        rows = df[null_mask].index
        global_total = df.loc[rows, self.GLOBAL_COST_COLS].sum(axis=1)
        local_total = df.loc[rows, self.LOCAL_COST_COLS].sum(axis=1)
        numerator = global_total - local_total
        denominator = global_total
        global_price = df.loc[rows, '全球采购物料价格']
        valid = (denominator != 0) & (~global_price.isna())
        df.loc[rows, self.COST_DIFF_COL] = np.where(valid, numerator / denominator, -1)
        print(f"已按公式补算 {self.COST_DIFF_COL} 的 {int(null_mask.sum())} 个缺失值。")
        return df

    # ---------------- 训练 ----------------
    def train_model(self, df_train):
        # 可选：补算成本差异
        df_train = self._fill_cost_diff(df_train)

        # 自动检测 X / Y
        self.target_name, self.model_features = auto_detect_xy(
            df_train, target_col=self.target_col, feature_cols=self.feature_cols
        )
        print("\n================ 自动检测结果 ================")
        print(f"目标变量 Y：{self.target_name}")
        print(f"自变量 X（共 {len(self.model_features)} 个）：{self.model_features}")

        # 组织 X、y
        X = df_train[self.model_features].apply(pd.to_numeric, errors='coerce').fillna(0)
        y = _to_binary_series(df_train[self.target_name])
        if y is None:
            raise ValueError(f"目标列 '{self.target_name}' 无法转成二值 0/1。")
        # 对齐并去掉 y 缺失的行
        mask = y.notna()
        X, y = X[mask], y[mask].astype(int)
        print(f"有效样本数：{len(y)}（正类 {int(y.sum())} / 负类 {int((y == 0).sum())}）")

        # ---------- 交叉验证评估（用全部数据，结果更可信） ----------
        self.cross_validate(X, y)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_tr, y_tr)

        y_proba = model.predict_proba(X_te)[:, 1]
        y_pred = (y_proba >= self.threshold).astype(int)

        acc = accuracy_score(y_te, y_pred)
        pre = precision_score(y_te, y_pred, zero_division=0)
        rec = recall_score(y_te, y_pred, zero_division=0)
        f1 = f1_score(y_te, y_pred, zero_division=0)

        print(f"\n============ 模型性能（阈值={self.threshold}） ============")
        print(f"准确率 Accuracy ：{acc:.4f}")
        print(f"精确率 Precision：{pre:.4f}")
        print(f"召回率 Recall   ：{rec:.4f}")
        print(f"F1 分数         ：{f1:.4f}")
        return model, (acc, pre, rec, f1)

    # ---------------- 交叉验证 ----------------
    def cross_validate(self, X, y):
        """
        用分层 K 折交叉验证评估模型，避免小样本单次切分带来的偶然性。
        自动根据样本量选择折数（最多 5 折，且不超过最小类别样本数）。
        """
        min_class = int(min((y == 0).sum(), (y == 1).sum()))
        if min_class < 2:
            print("\n（样本太少，某一类别不足 2 个，跳过交叉验证）")
            return None
        k = max(2, min(5, min_class))
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        model = LogisticRegression(max_iter=1000)
        # 用交叉验证得到每个样本的“留出预测”，再按阈值统一评估
        proba = cross_val_predict(model, X, y, cv=skf, method='predict_proba')[:, 1]
        pred = (proba >= self.threshold).astype(int)

        acc = accuracy_score(y, pred)
        pre = precision_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)

        print(f"\n============ 交叉验证结果（{k} 折，阈值={self.threshold}） ============")
        print(f"准确率 Accuracy ：{acc:.4f}")
        print(f"精确率 Precision：{pre:.4f}")
        print(f"召回率 Recall   ：{rec:.4f}")
        print(f"F1 分数         ：{f1:.4f}")
        print("（交叉验证用全部样本轮流验证，比单次切分更能反映真实水平）")
        return acc, pre, rec, f1

    # ---------------- 特征重要性 ----------------
    def eval_feature(self, model, show_plot=False):
        coef = model.coef_[0]
        fi = pd.DataFrame({
            '特征名称': self.model_features,
            '系数': coef,
            '重要性': np.abs(coef),
        }).sort_values('重要性', ascending=False).reset_index(drop=True)
        print("\n特征重要性排序（按系数绝对值降序）：")
        print(fi.to_string(index=False))
        if show_plot and plt is None:
            print("（未安装 matplotlib，跳过画图。可运行: pip install matplotlib）")
        if show_plot and plt is not None:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(len(fi)), fi['重要性'], color='#4ECDC4', edgecolor='#2C3E50')
            ax.set_xticks(range(len(fi)))
            ax.set_xticklabels(fi['特征名称'], rotation=45, ha='right')
            ax.set_ylabel('重要性（系数绝对值）')
            ax.set_title('逻辑回归 - 特征重要性')
            ax.yaxis.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.show()
        return fi

    # ---------------- 预测 ----------------
    def predict_with_model(self, model):
        if not self.pred_data_path:
            return None
        df_new = self.read_excel(self.pred_data_path)
        if df_new is None:
            return None
        df_new = self._fill_cost_diff(df_new)

        # 检查预测数据是否包含训练用到的自变量
        missing = [c for c in self.model_features if c not in df_new.columns]
        if missing:
            raise ValueError(f"预测数据缺少训练所需的自变量列：{missing}")

        X_new = df_new[self.model_features].apply(pd.to_numeric, errors='coerce').fillna(0)
        proba = model.predict_pr