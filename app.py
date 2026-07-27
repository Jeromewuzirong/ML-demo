# -*- coding: utf-8 -*-
"""
智能二分类建模平台 - 网页版后端（Flask）· 双文件流程
流程：
  1) 上传【训练文件】→ 自动识别 X/Y → 训练 → 保存模型（并可导出训练打分）
  2) 上传【预测文件】→ 选择/确认字段映射 → 用已保存模型输出一列预测结果 → 导出

运行：
    pip install flask pandas scikit-learn openpyxl
    python app.py
浏览器打开 http://127.0.0.1:5000
"""

import os
import re
import ast
import json
import uuid
import pickle
import traceback

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file, Response

try:
    from openai import OpenAI          # pip install openai —— 兼容所有 OpenAI 格式的服务商
except Exception:
    OpenAI = None

from logistic_regression_auto import smart_read_excel, auto_detect_xy, _to_binary_series, detect_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "_uploads")
RESULT_DIR = os.path.join(BASE_DIR, "_results")
MODEL_PATH = os.path.join(BASE_DIR, "_model.pkl")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024
STATE = {}

# ---------------- 大模型配置（AI 生成特征）----------------
# 换服务商只改这三个环境变量，代码不动：
#   LLM_API_KEY  你的 key
#   LLM_BASE_URL 例如 https://api.deepseek.com、https://api.openai.com/v1 等
#   LLM_MODEL    例如 deepseek-chat、gpt-4o-mini、qwen-plus 等
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash")
_llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) if (OpenAI and LLM_API_KEY) else None


# ---------------- 工具 ----------------
KEEP_UPLOADS = 10   # _uploads 里最多保留的最近文件数
KEEP_RESULTS = 10   # _results 里最多保留的最近文件数


def _prune_dir(d, keep, protect=None):
    """只保留最近修改的 keep 个文件，其余删除；protect 里的文件永不删（防止误删在用的文件）。"""
    protect = {os.path.abspath(p) for p in (protect or []) if p}
    try:
        files = [os.path.join(d, f) for f in os.listdir(d)]
        files = [p for p in files if os.path.isfile(p)]
        files.sort(key=os.path.getmtime, reverse=True)
        for p in files[keep:]:
            if os.path.abspath(p) in protect:
                continue
            try:
                os.remove(p)
            except Exception:
                pass
    except Exception:
        pass


def _save_upload(f):
    fid = uuid.uuid4().hex[:8]
    path = os.path.join(UPLOAD_DIR, f"{fid}_{f.filename}")
    f.save(path)
    # 保留最近若干个，且不删当前训练/预测正在用的文件
    _prune_dir(UPLOAD_DIR, KEEP_UPLOADS,
               protect=[path, STATE.get('train_path'), STATE.get('predict_path')])
    return path


def _sheets(path):
    return pd.ExcelFile(path).sheet_names


def _best_sheet(path, feats):
    """选包含指定字段最多的工作表。返回 (sheet, df)。"""
    sheets = _sheets(path)
    best_name, best_df, best_cov = sheets[0], None, -1
    for s in sheets:
        try:
            d = smart_read_excel(path, sheet_name=s)
        except Exception:
            continue
        cov = sum(1 for c in feats if c in d.columns) if feats else 0
        if cov > best_cov:
            best_name, best_df, best_cov = s, d, cov
    if best_df is None:
        best_df = smart_read_excel(path, sheet_name=sheets[0])
    return best_name, best_df


def build_X(df, features, impute='mean', scaler=None, fit=True):
    X = df[features].apply(pd.to_numeric, errors='coerce')
    if impute == 'zero':
        X = X.fillna(0)
    elif impute == 'median':
        X = X.fillna(X.median(numeric_only=True)).fillna(0)
    else:
        X = X.fillna(X.mean(numeric_only=True)).fillna(0)
    if scaler is not None:
        X = pd.DataFrame(scaler.fit_transform(X) if fit else scaler.transform(X),
                         columns=features, index=X.index)
    return X


def compute_fills(Xn, num_feats, default='mean', impute_map=None, fill_values=None):
    """
    计算每个数值特征的缺失填充常数。
    - fill_values 若提供（预测阶段）：直接沿用训练时算好的常数，保证训练/预测一致。
    - 否则按 impute_map 里该特征的设置（method: mean/median/zero/custom + value）计算；
      未单独设置的特征退回全局默认 default。
    """
    impute_map = impute_map or {}
    fills = {}
    for c in num_feats:
        if fill_values is not None and c in fill_values and fill_values[c] is not None:
            fills[c] = fill_values[c]
            continue
        spec = impute_map.get(c) or {}
        method = spec.get('method') or default
        if method == 'zero':
            fills[c] = 0.0
        elif method == 'custom':
            try:
                fills[c] = float(spec.get('value'))
            except (TypeError, ValueError):
                fills[c] = 0.0
        elif method == 'median':
            m = Xn[c].median()
            fills[c] = float(m) if pd.notna(m) else 0.0
        else:  # mean
            m = Xn[c].mean()
            fills[c] = float(m) if pd.notna(m) else 0.0
    return fills


def assemble_X(df, num_feats, categorical, impute='mean', scaler=None, fit=True,
               impute_map=None, fill_values=None):
    """
    组装最终特征矩阵 = 数值特征（缺失填充+可选标准化） + 类别列的 one-hot（0/1，不标准化）。
    :param num_feats:    数值自变量列名列表
    :param categorical:  {类别列名: [类别值1, 类别值2, ...]}，训练时确定、预测时复用
    :param impute:       全局默认缺失值处理（mean/median/zero），未单独设置的特征用它
    :param impute_map:   {特征: {'method': ..., 'value': ...}} 按特征覆盖的缺失值处理
    :param fill_values:  {特征: 常数}，预测阶段传入训练时算好的填充值，保证对齐
    列顺序固定：先所有数值列，再按 categorical 的顺序依次展开每个类别 → 保证训练/预测对齐。
    预测时遇到训练没见过的新类别，其各 one-hot 列自然全为 0（相当于"其他"）。
    """
    # 1) 数值部分
    if num_feats:
        Xn = df[num_feats].apply(pd.to_numeric, errors='coerce')
        fills = compute_fills(Xn, num_feats, default=impute,
                              impute_map=impute_map, fill_values=fill_values)
        Xn = Xn.fillna(fills).fillna(0)
        if scaler is not None:
            Xn = pd.DataFrame(scaler.fit_transform(Xn) if fit else scaler.transform(Xn),
                              columns=num_feats, index=Xn.index)
    else:
        Xn = pd.DataFrame(index=df.index)
    # 2) 类别部分（one-hot，0/1，不做标准化）
    cat_parts = []
    for col, cats in (categorical or {}).items():
        if col in df.columns:
            s = df[col].astype(str).str.strip()
        else:
            s = pd.Series(['__MISSING__'] * len(df), index=df.index)
        for cat in cats:
            cat_parts.append((s == cat).astype(float).rename(f"{col}={cat}"))
    if cat_parts:
        return pd.concat([Xn] + cat_parts, axis=1)
    return Xn


# ================= AI 生成特征 =================
def _extract_json(text):
    m = re.search(r'\{.*\}', text, re.S)
    if not m:
        raise ValueError("模型未返回 JSON")
    return json.loads(m.group(0))


def _get_client(api_key=None, base_url=None):
    """凭据优先用请求里用户填的；没有再退回服务器环境变量。"""
    if api_key:
        if OpenAI is None:
            raise ValueError("服务器未安装 openai 库")
        return OpenAI(api_key=api_key, base_url=(base_url or LLM_BASE_URL))
    if _llm_client is not None:
        return _llm_client
    raise ValueError("请先在页面右上「AI 设置」里填入你的 API Key")


def llm_expression(description, columns, api_key=None, base_url=None, model=None):
    """自然语言 → 派生特征结构化 spec。约定用 col("列名") 引用现有列。"""
    client = _get_client(api_key, base_url)
    use_model = model or LLM_MODEL
    sys = (
        "你是特征工程助手。把用户描述翻译成一个派生特征的结构化 JSON，先判断属于哪种 kind：\n"
        '1) numeric —— 逐行用现有列算一个数：\n'
        '   {"kind":"numeric","name":"新列名","expression":"col(\\"A\\")/col(\\"B\\")","explanation":"..."}\n'
        '   expression 只能用 col("列名")、+ - * / ( )、比较运算、'
        'np.where/np.abs/np.log/np.sqrt/np.minimum/np.maximum。\n'
        '2) group —— 按列分组做聚合：\n'
        '   {"kind":"group","name":"...","by":["城市"],"target":"成本",'
        '"agg":"mean|sum|max|min|count","mode":"value|ratio|share","explanation":"..."}\n'
        '   value=组内聚合值填回每行；ratio=该行值÷组内聚合值；share=该行值÷组内合计。\n'
        '3) cat —— 多个文本列拼成新类别列：\n'
        '   {"kind":"cat","name":"...","cols":["城市","类型"],"sep":"-","explanation":"..."}\n'
        "只返回一个 JSON 对象。所有列名必须来自给定的现有列。"
    )
    user = f"现有列：{list(columns)}\n用户描述：{description}"
    resp = client.chat.completions.create(
        model=use_model, max_tokens=600, temperature=0,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": user}])
    return _extract_json(resp.choices[0].message.content)


def llm_explain_error(error_text, description, api_key=None, base_url=None, model=None):
    """把技术报错翻成一两句中文说明 + 建议；失败则返回 None（前端退回原始报错）。"""
    try:
        client = _get_client(api_key, base_url)
    except Exception:
        return None
    short = (error_text or '').strip().split('\n')[-1][:300]
    sys = ("你是助手。用一到两句简体中文，向不懂编程的用户解释这次「AI 生成特征」为什么失败，"
           "并给一句可操作的建议（比如换种说法、检查某列是不是文本等）。"
           "不要出现英文报错原文或代码，语气平实简短。")
    user = f"用户想生成的特征：{description}\n技术报错：{short}"
    try:
        resp = client.chat.completions.create(
            model=(model or LLM_MODEL), max_tokens=200, temperature=0,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}])
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


_ALLOWED_NP = {'where', 'abs', 'log', 'sqrt', 'minimum', 'maximum'}


def _validate_ast(expr):
    """AST 白名单：只放行算术/比较/col()/np.<允许的函数>，挡掉 import、属性、危险调用。"""
    tree = ast.parse(expr, mode='eval')
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            ok = (isinstance(f, ast.Name) and f.id == 'col') or \
                 (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                  and f.value.id == 'np' and f.attr in _ALLOWED_NP)
            if not ok:
                raise ValueError("不允许的函数调用")
        elif isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == 'np'):
                raise ValueError("不允许的属性访问")
        elif isinstance(node, (ast.Import, ast.ImportFrom, ast.Lambda)):
            raise ValueError("不允许的语法")
        elif isinstance(node, ast.Name) and node.id.startswith('__'):
            raise ValueError("不允许的名称")
    return tree


def eval_feature(df, expression):
    """在 df 上安全求出一列数值，返回 pd.Series(float)。"""
    tree = _validate_ast(expression)
    cache = {}

    def col(name):
        if name not in df.columns:
            raise ValueError(f"未知列：{name}")
        if name not in cache:
            s = df[name]
            num = pd.to_numeric(s, errors='coerce')
            # 只要列里有可解析的数字（含"存成文本的数字"），就当数值用；
            # 完全无法转数值的（如类型 A/B）才保留文本，供 == 比较。
            cache[name] = num if num.notna().any() else s.astype(str).str.strip()
        return cache[name]

    safe = {"__builtins__": {}, "np": np, "col": col}
    res = eval(compile(tree, '<feat>', 'eval'), safe, {})
    return pd.Series(res, index=df.index).astype(float)


_AGG_OK = {'mean', 'sum', 'max', 'min', 'count'}


def build_derived(df, spec):
    """按 spec 生成一列，返回 (Series, is_categorical)。"""
    kind = spec.get('kind', 'numeric')

    if kind == 'numeric':
        return eval_feature(df, spec['expression']), False

    if kind == 'group':
        by, target = spec['by'], spec['target']
        agg = spec.get('agg', 'mean')
        mode = spec.get('mode', 'value')
        if agg not in _AGG_OK:
            raise ValueError(f"不支持的聚合：{agg}")
        for c in list(by) + [target]:
            if c not in df.columns:
                raise ValueError(f"未知列：{c}")
        tgt = pd.to_numeric(df[target], errors='coerce')
        keys = [df[c].astype(str).str.strip() for c in by]
        agg_series = tgt.groupby(keys).transform(agg)
        if mode == 'ratio':
            out = tgt / agg_series
        elif mode == 'share':
            out = tgt / tgt.groupby(keys).transform('sum')
        else:  # value
            out = agg_series
        return out.astype(float), False

    if kind == 'cat':
        cols = spec['cols']
        sep = spec.get('sep', '-')
        for c in cols:
            if c not in df.columns:
                raise ValueError(f"未知列：{c}")
        out = df[cols[0]].astype(str).str.strip()
        for c in cols[1:]:
            out = out + sep + df[c].astype(str).str.strip()
        return out, True

    raise ValueError(f"未知的特征类型：{kind}")


def apply_derived(df, derived, skip_errors=False):
    """生成所有派生列；返回 (新df, 类别派生列名, 跳过的派生列名)。训练与预测共用。
    skip_errors=True 时（预测阶段），若某派生列所需的原始字段在表中缺失，则跳过该列而不报错，
    交由下游按"缺失特征"用训练均值/全0 处理。"""
    df = df.copy()
    cat_added, skipped = [], []
    for d in (derived or []):
        try:
            series, is_cat = build_derived(df, d)
        except Exception:
            if skip_errors:
                skipped.append(str(d.get('name')))
                continue
            raise
        df[d['name']] = series
        if is_cat:
            cat_added.append(d['name'])
    return df, cat_added, skipped


def metrics_dict(yt, yp):
    return {'acc': round(float(accuracy_score(yt, yp)), 4),
            'pre': round(float(precision_score(yt, yp, zero_division=0)), 4),
            'rec': round(float(recall_score(yt, yp, zero_division=0)), 4),
            'f1': round(float(f1_score(yt, yp, zero_division=0)), 4)}


def make_preview(df, pred, proba, threshold, limit=None, actual=None):
    """limit=None 显示全部行；actual 给定(0/1 序列)时逐行附上实际结果与是否预测正确。"""
    id_col = None
    for c in df.columns:
        if any(k in str(c).lower() for k in ('id', '编号', '序号', '料号', 'name', '名称')):
            id_col = c
            break
    n = len(df) if limit is None else min(limit, len(df))
    av = None
    if actual is not None:
        av = pd.to_numeric(actual, errors='coerce').reset_index(drop=True)
    rows = []
    for i in range(n):
        rid = df[id_col].iloc[i] if id_col is not None else (i + 1)
        p = float(proba[i]); lab = int(pred[i])
        row = {'id': str(rid), 'label': '正类(1)' if lab == 1 else '负类(0)',
               'proba': round(p, 4),
               'judge': f"≥ {threshold} → 1" if p >= threshold else f"< {threshold} → 0"}
        if av is not None:
            a = av.iloc[i] if i < len(av) else None
            if a is None or pd.isna(a):
                row['actual'] = '—'; row['correct'] = None
            else:
                ai = int(a)
                row['actual'] = '正类(1)' if ai == 1 else '负类(0)'
                row['correct'] = bool(ai == lab)
        rows.append(row)
    return rows


def _export(df, pred, proba, threshold, actual=None):
    out = df.copy()
    out['预测结果'] = np.where(pred == 1, '正类', '负类')
    out['预测y值'] = pred
    out['y=1的概率'] = np.round(proba, 4)
    if actual is not None:
        av = pd.to_numeric(actual, errors='coerce').reset_index(drop=True)
        out['预测是否正确'] = ['' if pd.isna(av.iloc[i]) else ('✓' if int(av.iloc[i]) == int(pred[i]) else '✗')
                          for i in range(len(pred))]
    rid = uuid.uuid4().hex[:8]
    p = os.path.join(RESULT_DIR, f"预测结果_{rid}.xlsx")
    out.to_excel(p, index=False)
    _prune_dir(RESULT_DIR, KEEP_RESULTS, protect=[p])
    n = int(len(pred)); pos = int((pred == 1).sum()); neg = n - pos
    return {'n': n, 'pos': pos, 'neg': neg,
            'pos_pct': round(pos / n * 100, 1) if n else 0,
            'neg_pct': round(neg / n * 100, 1) if n else 0,
            'preview': make_preview(df, pred, proba, threshold, actual=actual),
            'download_id': rid}


def _binary_for_compare(s):
    """把对比列转成 0/1；允许只有一种取值（如全 0 或全 Y）。无法转换返回 None。"""
    from logistic_regression_auto import BINARY_MAP
    num = pd.to_numeric(s, errors='coerce')
    if num.notna().sum() > 0 and set(int(v) for v in num.dropna().unique()).issubset({0, 1}):
        return num
    low = s.astype(str).str.strip().str.lower()
    mapped = low.map(BINARY_MAP)
    if mapped.notna().sum() > 0 and set(mapped.dropna().unique()).issubset({0, 1}):
        return mapped
    vals = list(pd.Series(s).dropna().unique())
    if len(vals) == 2:
        return pd.Series(s).map({vals[0]: 0, vals[1]: 1})
    return None


def _model_info():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with open(MODEL_PATH, 'rb') as mf:
            b = pickle.load(mf)
        return {'version': b.get('version', '-'), 'y_name': str(b.get('y_name', '')),
                'features': [str(c) for c in b.get('features', [])],
                'categorical': {str(k): [str(x) for x in v]
                                for k, v in (b.get('categorical', {}) or {}).items()},
                'derived': b.get('derived', []) or [],
                'threshold': b.get('threshold', 0.5)}
    except Exception:
        return None


# ---------------- 页面 ----------------
@app.route('/')
def index():
    with open(os.path.join(BASE_DIR, 'index.html'), encoding='utf-8') as f:
        return Response(f.read(), mimetype='text/html')


@app.route('/api/model_info')
def api_model_info():
    info = _model_info()
    return jsonify(ok=True, has_model=info is not None, model_info=info)


# ---------------- 步骤1：上传训练文件 ----------------
@app.route('/api/upload_train', methods=['POST'])
def api_upload_train():
    try:
        f = request.files.get('file')
        if f is None:
            return jsonify(ok=False, msg="未收到文件")
        path = _save_upload(f)
        sheet, df = _best_sheet(path, None)
        df = smart_read_excel(path, sheet_name=_sheets(path)[0])
        cols = [str(c) for c in df.columns]
        try:
            y_name, feats = auto_detect_xy(df)
        except Exception:
            y_name, feats = None, []
        # 自动推荐"类别字段"（文本、2~20 种取值），排除已作 Y / 数值 X 的列
        try:
            cat_fields = detect_categorical(df, exclude=([y_name] if y_name else []) + list(feats))
        except Exception:
            cat_fields = []
        STATE['train_path'] = path
        return jsonify(ok=True, filename=f.filename, size=os.path.getsize(path),
                       n_rows=int(len(df)), columns=cols,
                       input_fields=[str(c) for c in feats],
                       output_field=str(y_name) if y_name is not None else '',
                       cat_fields=[str(c) for c in cat_fields])
    except Exception:
        return jsonify(ok=False, msg=traceback.format_exc())


# ---------------- 步骤2：训练并保存模型 ----------------
@app.route('/api/train', methods=['POST'])
def api_train():
    try:
        cfg = request.get_json(force=True)
        threshold = float(cfg.get('threshold', 0.5))
        impute = cfg.get('impute', 'mean')
        impute_map = cfg.get('impute_map') or {}   # {特征: {method, value}} 按特征覆盖
        use_scaler = bool(cfg.get('scaler', False))
        target = cfg.get('target') or None
        features = cfg.get('features') or None
        cat_sel = cfg.get('categorical') or []   # 用户勾选要 one-hot 的类别列
        derived = cfg.get('derived') or []       # AI 生成的派生特征 spec 列表

        path = STATE.get('train_path')
        if not path or not os.path.exists(path):
            return jsonify(ok=False, msg="请先上传训练文件")
        df = smart_read_excel(path, sheet_name=_sheets(path)[0])
        # 先把 AI 派生列算出来加进 df（拼类别的自动并入类别列选择）
        df, cat_added, _ = apply_derived(df, derived)
        cat_sel = list(cat_sel) + [c for c in cat_added if c not in cat_sel]
        y_name, feats = auto_detect_xy(df, target_col=target, feature_cols=features)

        # 类别列 → 记下每列训练时出现的类别（排序去空），预测时按这套对齐
        categorical = {}
        for col in cat_sel:
            if col in df.columns and col != y_name and col not in feats:
                cats = sorted(v for v in df[col].dropna().astype(str).str.strip().unique() if v != '')
                if len(cats) >= 2:
                    categorical[col] = cats

        y = _to_binary_series(df[y_name])
        if y is None:
            nvals = int(df[y_name].dropna().nunique())
            return jsonify(ok=False, msg=(
                f"目标列『{y_name}』有 {nvals} 种取值，不是二值(0/1)，无法作为 Y。"
                f"请在「目标字段(Y)」里改选二值列（如 0/1、是/否、Y/N），"
                f"本模型只支持二分类。"))
        scaler = StandardScaler() if use_scaler else None
        # 按特征算好每列的缺失填充常数（训练时确定，预测时复用 → 完全对齐）
        raw_feat = df[feats].apply(pd.to_numeric, errors='coerce')
        fill_values = compute_fills(raw_feat, feats, default=impute, impute_map=impute_map)
        Xfull = assemble_X(df, feats, categorical, impute=impute, scaler=scaler, fit=True,
                           fill_values=fill_values)
        final_cols = list(Xfull.columns)
        mask = y.notna()
        Xfull, yv = Xfull[mask], y[mask].astype(int)

        cv = None
        min_class = int(min((yv == 0).sum(), (yv == 1).sum()))
        if min_class >= 2:
            k = max(2, min(5, min_class))
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            cvp = cross_val_predict(LogisticRegression(max_iter=1000), Xfull, yv,
                                    cv=skf, method='predict_proba')[:, 1]
            cv = metrics_dict(yv, (cvp >= threshold).astype(int)); cv['k'] = k

        X_tr, X_te, y_tr, y_te = train_test_split(Xfull, yv, test_size=0.2, random_state=42)
        m = LogisticRegression(max_iter=1000); m.fit(X_tr, y_tr)
        holdout = metrics_dict(y_te, (m.predict_proba(X_te)[:, 1] >= threshold).astype(int))

        final = LogisticRegression(max_iter=1000); final.fit(Xfull, yv)
        version = 'V' + uuid.uuid4().hex[:4]
        with open(MODEL_PATH, 'wb') as mf:
            pickle.dump({'model': final, 'features': feats, 'y_name': y_name,
                         'categorical': categorical, 'scaler': scaler, 'impute': impute,
                         'impute_map': impute_map, 'threshold': threshold, 'version': version,
                         'derived': derived, 'fill_values': fill_values}, mf)

        # 特征重要性：数值列 + one-hot 展开列一起排
        coef = final.coef_[0]
        imp = sorted([{'name': str(final_cols[i]), 'coef': round(float(coef[i]), 4),
                       'imp': round(float(abs(coef[i])), 4)} for i in range(len(final_cols))],
                     key=lambda d: d['imp'], reverse=True)

        # 训练文件自身打分，便于导出查看
        Xall = assemble_X(df, feats, categorical, impute=impute, scaler=scaler, fit=False,
                          fill_values=fill_values)
        pp = final.predict_proba(Xall)[:, 1]
        pr = (pp >= threshold).astype(int)

        return jsonify(ok=True, y_name=str(y_name), features=[str(c) for c in feats],
                       categorical={k: v for k, v in categorical.items()},
                       n_train=int(len(yv)), n_pos=int((yv == 1).sum()), n_neg=int((yv == 0).sum()),
                       cv=cv, holdout=holdout, importance=imp, model_version=version,
                       threshold=threshold, predict=_export(df, pr, pp, threshold),
                       predict_source="训练文件自身打分")
    except Exception:
        return jsonify(ok=False, msg=traceback.format_exc())


# ---------------- AI 生成新特征 ----------------
@app.route('/api/gen_feature', methods=['POST'])
def api_gen_feature():
    # 先解析凭据/描述，放在 try 外，便于失败时用同一 key 生成中文解释
    try:
        cfg = request.get_json(force=True)
    except Exception:
        return jsonify(ok=False, msg="请求解析失败")
    desc = (cfg.get('description') or '').strip()
    # 凭据由用户在页面填写、随请求传来；后端只临时使用、不保存
    api_key = (cfg.get('api_key') or '').strip()
    base_url = (cfg.get('base_url') or '').strip()
    model = (cfg.get('model') or '').strip()
    try:
        if not desc:
            return jsonify(ok=False, msg="请填写特征逻辑描述")
        path = STATE.get('train_path')
        if not path or not os.path.exists(path):
            return jsonify(ok=False, msg="请先上传训练文件")
        df = smart_read_excel(path, sheet_name=_sheets(path)[0])
        spec = llm_expression(desc, [str(c) for c in df.columns],
                              api_key=api_key, base_url=base_url, model=model)
        series, is_cat = build_derived(df, spec)                    # 立即试算，出错即抛
        if is_cat:
            preview = [str(v) for v in series.head(8)]
            n_nan = 0
        else:
            preview = [None if pd.isna(v) else round(float(v), 4) for v in series.head(8)]
            n_nan = int(series.isna().sum())
        return jsonify(ok=True, spec=spec, is_cat=is_cat,
                       name=str(spec.get('name', '新特征')),
                       explanation=str(spec.get('explanation', '')),
                       preview=preview, n_nan=n_nan)
    except Exception:
        raw = traceback.format_exc()
        friendly = llm_explain_error(raw, desc, api_key=api_key, base_url=base_url, model=model)
        return jsonify(ok=False, msg=raw, friendly=friendly)


# ---------------- 预览导入的训练文件 ----------------
@app.route('/api/preview_train')
def api_preview_train():
    try:
        path = STATE.get('train_path')
        if not path or not os.path.exists(path):
            return jsonify(ok=False, msg="请先上传训练文件")
        df = smart_read_excel(path, sheet_name=_sheets(path)[0])
        limit = 50
        head = df.head(limit)
        cols = [str(c) for c in df.columns]
        rows = [['' if pd.isna(v) else str(v) for v in r]
                for r in head.itertuples(index=False, name=None)]
        return jsonify(ok=True, columns=cols, rows=rows,
                       n_rows=int(len(df)), n_cols=len(cols), shown=len(rows))
    except Exception:
        return jsonify(ok=False, msg=traceback.format_exc())


# ---------------- 预览"应用新特征后"的数据表 ----------------
@app.route('/api/preview_derived', methods=['POST'])
def api_preview_derived():
    try:
        cfg = request.get_json(force=True)
        derived = cfg.get('derived') or []
        path = STATE.get('train_path')
        if not path or not os.path.exists(path):
            return jsonify(ok=False, msg="请先上传训练文件")
        df = smart_read_excel(path, sheet_name=_sheets(path)[0])
        base_cols = [str(c) for c in df.columns]
        df2, _, _ = apply_derived(df, derived)          # 配置阶段原始列都在，正常算
        cols = [str(c) for c in df2.columns]
        new_cols = [c for c in cols if c not in base_cols]
        head = df2.head(50)
        rows = [['' if pd.isna(v) else str(v) for v in r]
                for r in head.itertuples(index=False, name=None)]
        return jsonify(ok=True, columns=cols, rows=rows, new_cols=new_cols,
                       n_rows=int(len(df2)), n_cols=len(cols), shown=len(rows))
    except Exception:
        return jsonify(ok=False, msg=traceback.format_exc())


# ---------------- 步骤3：上传预测文件 ----------------
@app.route('/api/upload_predict', methods=['POST'])
def api_upload_predict():
    try:
        info = _model_info()
        if info is None:
            return jsonify(ok=False, msg="还没有已保存的模型，请先完成训练。")
        f = request.files.get('file')
        if f is None:
            return jsonify(ok=False, msg="未收到文件")
        path = _save_upload(f)
        feats = info['features']
        sheet, df = _best_sheet(path, feats)
        cols = [str(c) for c in df.columns]
        # 派生列由公式自动重算，不需要预测文件提供、也不算缺失
        derived_names = {str(d.get('name')) for d in info.get('derived', [])}
        # 自动匹配：模型字段 → 预测文件里的同名列（派生列标记为自动生成）
        match = {c: (c if (c in cols or c in derived_names) else None) for c in feats}
        # 类别列按同名自动对齐；找出预测文件里缺失的类别列（派生类别列除外）
        cat_cols = list(info.get('categorical', {}).keys())
        cat_missing = [c for c in cat_cols if c not in cols and c not in derived_names]
        STATE['predict_path'] = path
        return jsonify(ok=True, filename=f.filename, size=os.path.getsize(path),
                       n_rows=int(len(df)), columns=cols, sheet=sheet,
                       features=feats, match=match, model_version=info['version'],
                       cat_fields=cat_cols, cat_missing=cat_missing,
                       derived=sorted(derived_names))
    except Exception:
        return jsonify(ok=False, msg=traceback.format_exc())


# ---------------- 步骤4：生成预测结果 ----------------
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        info = _model_info()
        if info is None:
            return jsonify(ok=False, msg="没有已保存的模型，请先训练。")
        with open(MODEL_PATH, 'rb') as mf:
            bundle = pickle.load(mf)
        cfg = request.get_json(force=True)
        threshold = float(cfg.get('threshold', bundle.get('threshold', 0.5)))
        mapping = cfg.get('mapping') or {}   # {模型字段: 预测文件列名}

        path = STATE.get('predict_path')
        if not path or not os.path.exists(path):
            return jsonify(ok=False, msg="请先上传预测文件")
        feats = bundle['features']
        _, df = _best_sheet(path, feats)
        # 用训练时保存的同一套 spec 重算派生列（保证训练/预测一致）；
        # 若预测文件缺少某派生特征所需的原始字段，则跳过它、按缺失特征处理（不报错）
        df, _, skipped_derived = apply_derived(df, bundle.get('derived'), skip_errors=True)

        # 按映射把预测文件的列对齐到模型字段
        # mapping 值可为：某列名 / '__MEAN__'(训练均值) / '__ZERO__'(置0/不采用)
        use = pd.DataFrame(index=df.index)
        fillv = bundle.get('fill_values', {})
        derived_names = {str(d.get('name')) for d in bundle.get('derived', [])}
        used_mean, used_zero = [], []
        for feat in feats:
            # 派生特征：无视映射，直接用公式自动算出的值（缺原始字段时上面已跳过→用训练均值）
            if feat in derived_names:
                use[feat] = df[feat] if feat in df.columns else fillv.get(feat, 0.0)
                continue
            val = mapping.get(feat)
            if val == '__ZERO__':
                use[feat] = 0.0; used_zero.append(feat)
            elif val and val not in ('__MEAN__',) and val in df.columns and df[val].notna().sum() > 0:
                use[feat] = df[val]
            else:
                # __MEAN__ 或 未选/空列 → 用训练均值
                use[feat] = fillv.get(feat, 0.0); used_mean.append(feat)
        if feats and len(used_mean) + len(used_zero) == len(feats):
            return jsonify(ok=False, msg="所有模型字段都没有对应到预测文件的实际列，请检查文件或字段映射。")
        parts = []
        if used_mean: parts.append(f"用训练均值填充：{used_mean}")
        if used_zero: parts.append(f"置0(不采用)：{used_zero}")
        if skipped_derived:
            parts.append(f"派生特征缺少所需原始字段、已按训练均值处理：{skipped_derived}")

        # 类别列（one-hot）：按同名列从预测文件取原始值；缺失列/新类别 → 该行 one-hot 全 0
        categorical = bundle.get('categorical', {}) or {}
        cat_missing, new_cat = [], []
        for col, cats in categorical.items():
            if col in df.columns:
                use[col] = df[col]
                s = df[col].astype(str).str.strip()
                n_new = int((~s.isin(cats) & s.notna() & (s != '') & (s.str.lower() != 'nan')).sum())
                if n_new > 0:
                    new_cat.append(f"{col}({n_new}行)")
            else:
                cat_missing.append(col)
        if cat_missing:
            parts.append(f"类别列在预测文件中缺失、按'其他'处理：{cat_missing}")
        if new_cat:
            parts.append(f"出现训练未见过的新类别、已置0：{new_cat}")
        warn = ("提示（" + "；".join(parts) + "），结果仅供参考") if parts else None

        X = assemble_X(use, feats, categorical, impute=bundle['impute'],
                       scaler=bundle['scaler'], fit=False,
                       fill_values=bundle.get('fill_values'))
        proba = bundle['model'].predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)

        # 可选：与预测文件里指定的对比列比较，给出准确率
        compare = None
        actual_series = None   # 传给 _export/make_preview 做逐行对比；无实际结果列则保持 None
        ccol = cfg.get('compare_col')
        if ccol and ccol in df.columns:
            ab = _binary_for_compare(df[ccol])
            if ab is not None:
                actual_series = ab
            if ab is None:
                compare = {'col': str(ccol), 'error': '该列无法转成 0/1（取值超过两种），无法对比'}
            else:
                m = ab.notna().values
                if int(m.sum()) == 0:
                    compare = {'col': str(ccol), 'error': '该列没有有效值'}
                else:
                    a = ab[m].astype(int).values
                    pv = pred[m]
                    tp = int(((pv == 1) & (a == 1)).sum()); tn = int(((pv == 0) & (a == 0)).sum())
                    fp = int(((pv == 1) & (a == 0)).sum()); fn = int(((pv == 0) & (a == 1)).sum())
                    compare = {'col': str(ccol), 'n': int(m.sum()), 'agree': int((pv == a).sum()),
                               'acc': round(float((pv == a).mean()), 4),
                               'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                               'actual_pos': int((a == 1).sum()), 'actual_neg': int((a == 0).sum())}

        return jsonify(ok=True, threshold=threshold, model_version=bundle.get('version', '-'),
                       predict=_export(df, pred, proba, threshold, actual=actual_series),
                       compare=compare, warn=warn, predict_source="预测文件")
    except Exception:
        return jsonify(ok=False, msg=traceback.format_exc())


@app.route('/api/download/<rid>')
def api_download(rid):
    p = os.path.join(RESULT_DIR, f"预测结果_{rid}.xlsx")
    if not os.path.exists(p):
        return "文件不存在", 404
    return send_file(p, as_attachment=True, download_name="预测结果.xlsx")


# ---------------- 导出当前模型（下载 .pkl） ----------------
@app.route('/api/export_model')
def api_export_model():
    if not os.path.exists(MODEL_PATH):
        return "还没有可导出的模型，请先训练。", 404
    info = _model_info()
    ver = info['version'] if info else 'model'
    return send_file(MODEL_PATH, as_attachment=True, download_name=f"模型_{ver}.pkl")


# ---------------- 导入模型（上传 .pkl，设为当前模型） ----------------
@app.route('/api/import_model', methods=['POST'])
def api_import_model():
    try:
        f = request.files.get('file')
        if f is None:
            return jsonify(ok=False, msg="未收到文件")
        data = f.read()
        try:
            bundle = pickle.loads(data)
        except Exception:
            return jsonify(ok=False, msg="文件无法解析，请上传本平台导出的 .pkl 模型文件。")
        # 校验：必须是本平台的模型包（含 model 和 features）
        if not isinstance(bundle, dict) or 'model' not in bundle or 'features' not in bundle:
            return jsonify(ok=False, msg="这不是有效的模型文件（缺少 model / features 字段）。")
        with open(MODEL_PATH, 'wb') as mf:
            mf.write(data)
        return jsonify(ok=True, model_info=_model_info())
    except Exception:
        return jsonify(ok=False, msg=traceback.format_exc())


if __name__ == '__main__':
    print("网页版已启动: http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
