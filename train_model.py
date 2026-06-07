"""
CalTenant AI — Random Forest training + ONNX export
MBA7008 AI Capstone, Sofia University

Trains a Random Forest classifier on the California small claims dataset,
reports honest accuracy (5-fold CV + hold-out), and exports the model to
ONNX so it can run client-side in the browser via onnxruntime-web.

Usage:
    python train_model.py /path/to/caltenant_Dataset.xlsx
"""
import sys, json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

DATA = sys.argv[1] if len(sys.argv) > 1 else "caltenant_Dataset (150).xlsx"

CAT_COLS = ['dispute_type', 'county', 'written_lease', 'has_photos',
            'has_written_communication', 'has_signed_checklist',
            'has_receipts_or_payments', 'has_witness', 'demand_letter_sent',
            'prior_resolution_attempt', 'landlord_is_company']
NUM_COLS = ['dollar_amount', 'evidence_score']
SEED = 42

def main():
    df = pd.read_excel(DATA)
    y = df['outcome']
    X = df.drop(columns=['outcome'])

    # Fixed category order so the browser can reproduce the encoding exactly
    cats = [sorted(df[c].unique().tolist()) for c in CAT_COLS]
    enc = OneHotEncoder(categories=cats, handle_unknown='ignore', sparse_output=False)

    Xc = enc.fit_transform(X[CAT_COLS])
    Xn = X[NUM_COLS].to_numpy(dtype=np.float32)
    Xfull = np.hstack([Xc, Xn]).astype(np.float32)
    feat_order = list(enc.get_feature_names_out(CAT_COLS)) + NUM_COLS

    rf_params = dict(n_estimators=200, max_depth=8, min_samples_leaf=2,
                     random_state=SEED, class_weight='balanced')

    # --- Honest evaluation ---
    Xtr, Xte, ytr, yte = train_test_split(Xfull, y, test_size=0.25,
                                          stratify=y, random_state=SEED)
    rf_eval = RandomForestClassifier(**rf_params).fit(Xtr, ytr)
    holdout = accuracy_score(yte, rf_eval.predict(Xte))
    cv = cross_val_score(RandomForestClassifier(**rf_params), Xfull, y,
                         cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    print(f"Hold-out test accuracy : {holdout*100:.2f}%")
    print(f"5-fold CV accuracy     : {cv.mean()*100:.2f}% (+/- {cv.std()*100:.2f}%)")
    print("\nClassification report (hold-out):\n",
          classification_report(yte, rf_eval.predict(Xte)))

    # --- Final model on all data, exported to ONNX ---
    rf = RandomForestClassifier(**rf_params).fit(Xfull, y)
    onx = convert_sklearn(
        rf,
        initial_types=[('input', FloatTensorType([None, Xfull.shape[1]]))],
        options={id(rf): {'zipmap': False}},
        target_opset=17,
    )
    with open("model.onnx", "wb") as f:
        f.write(onx.SerializeToString())

    json.dump({"cat_cols": CAT_COLS, "cats": cats, "num_cols": NUM_COLS,
               "feat_order": feat_order, "classes": list(rf.classes_)},
              open("model_meta.json", "w"), indent=2)
    print("\nExported model.onnx and model_meta.json")

if __name__ == "__main__":
    main()
