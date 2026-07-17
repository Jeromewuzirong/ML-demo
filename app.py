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
import uuid
import pickle
import traceback

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file, Response

from logistic_regression_auto import smart_read_excel, auto_detect_xy, _to_binary_series
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


# ---------------- 工具 ----------------
def _save_upload(f):
    fid = uuid.uuid4().hex[:8]
    path = os.path.join(UPLOAD_DIR, f"{fid}_{f.filename}")
    f.save(path)
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


def metrics_dict(yt, yp):
    return {'acc': round(float(accuracy_score(yt, yp)), 4),
            'pre': round(float(precision_score(yt, yp, zero_division=0)), 4),
            'rec': round(float(recall_score(yt, yp, zero_division=0)), 4),
            'f1': round(float(f1_score(yt, yp, zero_division=0)), 4)}


def make_preview(df, pred, proba, threshold, limit=10):
    id_col = None
    for c in df.columns:
        if any(k in str(c).lower() for k in ('id', '编号', '序号', '料号', 'name', '名称')):
            id_col = c
            break
    rows = []
    for i in range(min(limit, len(df))):
        rid = df[id_col].iloc[i] if id_col is not None else (i + 1)
        p = float(proba[i]); lab = int(pred[i])
        rows.append({'id': str(rid), 'label': '正类(1)' if lab == 1 else '负类(0)',
                     'proba': round(p, 4),
                     'judge': f"≥ {threshold} → 1" if p >= threshold else f"< {threshold} → 0"})
    return rows


def _export(df, pred, proba, threshold):
    out = df.copy()
    out['预测结果'] = np.where(pred == 1, '正类', '负类')
    out['预测y值'] = pred
    out['y=1的概率'] = np.round(proba, 4)
    rid = uuid.uuid4().hex[:8]
    p = os.path.join(RESULT_DIR, f"预测结果_{rid}.xlsx")
    out.to_excel(p, index=False)
    n = int(len(pred)); pos = int((pred == 1).sum()); neg = n - pos
    return {'n': n, 'pos': pos, 'neg': neg,
            'pos_pct': round(pos / n * 100, 1) if n else 0,
            'neg_pct': round(neg / n * 100, 1) if n else 0,
            'preview': make_preview(df, pred, proba, threshold),
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
        STATE['train_path'] = path
        return jsonify(ok=True, filename=f.filename, size=os.path.getsize(path),
                       n_rows=int(len(df)), columns=cols,
                       input_fields=[str(c) for c in feats],
                       output_field=str(y_name) if y_name is not None else '')
    except Exception:
        return jsonify(ok=False, msg=traceback.format_exc())


# ---------------- 步骤2：训练并保存模型 ----------------
@app.route('/api/train', methods=['POST'])
def api_train():
    try:
        cfg = request.get_json(force=True)
        threshold = float(cfg.get('threshold', 0.5))
        impute = cfg.get('impute', 'mean')
        use_scaler = bool(cfg.get('scaler', False))
        target = cfg.get('target') or None
        features = cfg.get('features') or None

        path = STATE.get('train_path')
        if not path or not os.path.exists(path):
            return jsonify(ok=False, msg="请先上传训练文件")
        df = smart_read_excel(path, sheet_name=_sheets(path)[0])
        y_name, feats = auto_detect_xy(df, target_col=target, feature_cols=features)

        y = _to_binary_series(df[y_name])
        scaler = StandardScaler() if use_scaler else None
        Xfull = build_X(df, feats, impute=impute, scaler=scaler, fit=True)
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
        raw_feat = df[feats].apply(pd.to_numeric, errors='coerce')
        fill_values = {c: (float(raw_feat[c].mean()) if raw_feat[c].notna().any() else 0.0)
                       for c in feats}
        with open(MODEL_PATH, 'wb') as mf:
            pickle.dump({'model': final, 'features': feats, 'y_name': y_name,
                         'scaler': scaler, 'impute': impute, 'threshold': threshold,
                         'version': version, 'fill_values': fill_values}, mf)

        coef = final.coef_[0]
        imp = sorted([{'name': str(feats[i]), 'coef': round(float(coef[i]), 4),
                       'imp': round(float(abs(coef[i])), 4)} for i in range(len(feats))],
                     key=lambda d: d['imp'], reverse=True)

        # 训练文件自身打分，便于导出查看
        Xall = build_X(df, feats, impute=impute, scaler=scaler, fit=False)
        pp = final.predict_proba(Xall)[:, 1]
        pr = (pp >= threshold).astype(int)

        return jsonify(ok=True, y_name=str(y_name), features=[str(c) for c in feats],
                       n_train=int(len(yv)), n_pos=int((yv == 1).sum()), n_neg=int((yv == 0).sum()),
                       cv=cv, holdout=holdout, importance=imp, model_version=version,
                       threshold=threshold, predict=_export(df, pr, pp, threshold),
                       predict_source="训练文件自身打分")
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
        # 自动匹配：模型字段 → 预测文件里的同名列
        match = {c: (c if c in cols else None) for c in feats}
        STATE['predict_path'] = path
        return jsonify(ok=True, filename=f.filename, size=os.path.getsize(path),
                       n_rows=int(len(df)), columns=cols, sheet=sheet,
                       features=feats, match=match, model_version=info['version'])
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

        # 按映射把预测文件的列对齐到模型字段
        # mapping 值可为：某列名 / '__MEAN__'(训练均值) / '__ZERO__'(置0/不采用)
        use = pd.DataFrame(index=df.index)
        fillv = bundle.get('fill_values', {})
        used_mean, used_zero = [], []
        for feat in feats:
            val = mapping.get(feat)
            if val == '__ZERO__':
                use[feat] = 0.0; used_zero.append(feat)
            elif val and val not in ('__MEAN__',) and val in df.columns and df[val].notna().sum() > 0:
                use[feat] = df[val]
            else:
                # __MEAN__ 或 未选/空列 → 用训练均值
                use[feat] = fillv.get(feat, 0.0); used_mean.append(feat)
        if len(used_mean) + len(used_zero) == len(feats):
            return jsonify(ok=False, msg="所有模型字段都没有对应到预测文件的实际列，请检查文件或字段映射。")
        parts = []
        if used_mean: parts.append(f"用训练均值填充：{used_mean}")
        if used_zero: parts.append(f"置0(不采用)：{used_zero}")
        warn = ("部分字段未取自预测文件（" + "；".join(parts) + "），结果仅供参考") if parts else None

        X = build_X(use, feats, impute=bundle['impute'], scaler=bundle['scaler'], fit=False)
        proba = bundle['model'].predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)

        # 可选：与预测文件里指定的对比列比较，给出准确率
        compare = None
        ccol = cfg.get('compare_col')
        if ccol and ccol in df.columns:
            ab = _binary_for_compare(df[ccol])
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
                       predict=_export(df, pred, proba, threshold),
                       compare=compare, warn=warn, predict_source="预测文件")
    except Exception:
        return jsonify(ok=False, msg=traceback.format_exc())


@app.route('/api/download/<rid>')
def api_download(rid):
    p = os.path.join(RESULT_DIR, f"预测结果_{rid}.xlsx")
    if not os.path.exists(p):
        return "文件不存在", 404
    return send_file(p, as_attachment=True, download_name="预测结果.xlsx")


if __name__ == '__main__':
    print("网页版已启动: http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
