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
    vals = s.dropna().unique() #dropna去掉空值，unique去重，看这一列是否只有0/1
    if len(vals) != 2:
        return None
    # 已经是数值 0/1
    try:
        num = pd.to_numeric(s, errors='raise') #尝试转成数字
        uniq = set(num.dropna().unique())
        if uniq.issubset({0, 1}): #检查二值
            return num.astype('Int64')
        # 其它两种数值 → 映射为 0/1（小的为0，大的为1）
        lo, hi = sorted(uniq)
        return num.map({lo: 0, hi: 1}).astype('Int64')
    except Exception: #转换失败抛数字
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
    raw = pd.read_excel(file_path, header=None, sheet_name=sheet_name) #读整张表
    if raw.empty:
        return pd.DataFrame() #得到一个dataframe，类似矩阵

    def _row_num_frac(row): #判断此行为数据行的概率
        nn = [v for v in row if pd.notna(v)]
        return (sum(1 for v in nn if _is_number(v)) / len(nn)) if nn else 0.0

    n_rows = len(raw)
    best_r, best_score = None, -1     # 满足“文字表头+下一行是数据”的最佳行
    fb_r, fb_score = 0, -1            # 兜底：单纯下方数值列最多的行
    for r in range(min(max_scan, n_rows - 1)): #扫描行
        data = raw.iloc[r + 1:] # 假设r是表头时，他下面的所有行
        # 该行作为表头时，下方为“数值列”的数量
        numeric_below = sum(1 for c in range(raw.shape[1]) #计数，满足+1
                            if data[c].notna().sum() >= 3 and _numeric_frac(data[c]) >= 0.6)
                            # 非空值 >= 3，有足够的有效数据    # 行中能转换成数值的比例大于 60%
        # 表头行本身应以文字为主（字段名是文字）
        hdr = raw.iloc[r].dropna() # 取第r行，去掉空值
        header_text_frac = (sum(1 for v in hdr if not _is_number(v)) / len(hdr)) if len(hdr) else 0.0
                            # 计算不是数值（是文字）的比例
        # 紧接的下一行应是数据行（以数字为主）——排除“说明行/分组标题行”
        next_num_frac = _row_num_frac(raw.iloc[r + 1])
        if numeric_below > fb_score: # 若下方数值参数最多，把它记成兜底行
            fb_score, fb_r = numeric_below, r
        if (numeric_below >= 2 and header_text_frac >= 0.5 and next_num_frac >= 0.5 #下方至少有两个数值列，这一行至少一般是文字，下一行至少一半是数字
                and numeric_below > best_score): #数值比之前的好
            best_score, best_r = numeric_below, r

    # 没有合适候选时退回兜底行
    if best_r is None:
        best_r = fb_r

    # 构造表头：空表头向上回填（处理合并单元格：标签在上一行）
    header = []
    for c in range(raw.shape[1]):
        name = raw.iat[best_r, c]
        rr = best_r
        while (pd.isna(name) or str(name).strip() == '') and rr > 0: # 当前格为空(NaN或空字符串) 且 还没到第0行 → 继续往上找
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


def detect_categorical(df, exclude=None, max_card=20):
    """
    自动挑出"像类别"的文本列，作为 one-hot 编码的候选（只推荐，不编码）。
    规则：
      - 非数值列（数值列走原来的 X 逻辑，不在这里）；
      - 取值种类适中：2 ~ max_card 种（默认最多 20 种，太多不适合 one-hot）；
      - 排除近乎唯一的列（取值数≈非空行数，属于名称/ID，不是类别）。
    :param exclude: 要跳过的列名集合（通常是目标列 Y 和已选的数值 X）
    :return: 候选类别列名列表
    """
    exclude = set(str(c) for c in (exclude or []))
    cats = []
    for c in df.columns:
        if str(c) in exclude:
            continue
        s = df[c]
        nonnull = s.dropna()
        if len(nonnull) == 0:
            continue
        # 主要能转成数值的列 → 当数值特征处理，不算类别
        num = pd.to_numeric(s, errors='coerce')
        if num.notna().sum() >= max(3, int(0.6 * len(nonnull))):
            continue
        k = nonnull.astype(str).str.strip().nunique()
        # 取值种类在 [2, max_card] 之间，且不是近乎唯一（k 不能等于非空行数）
        if 2 <= k <= max_card and k < len(nonnull):
            cats.append(c)
    return [str(c) for c in cats]

