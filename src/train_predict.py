# ============================================================
# TWO-STAGE FINAL (STRICT GATE)
#
# Stage 1 (fixed):
#   - StandardScaler + Logistic Regression
#   - Features: Baseline + engineered
#   - Task: class2 (event vs nonevent)
#
# Stage 2 (fixed):
#   - StandardScaler + SVM (RBF, probability=True)
#   - Features: Baseline + engineered
#   - Task: class4 (Ia/Ib/II/nonevent) trained on FULL labels
#
# STRICT GATING (NO OVERRIDE):
#   - If Stage 1 predicts NONEVENT => final class4 = nonevent
#   - Stage 2 is used ONLY when Stage 1 predicts EVENT
#   - When used, Stage 2 chooses among Ia/Ib/II only
#
# PROBABILITIES:
#   - Kaggle p uses ONLY Stage 1 probability: P(event)
#   - Optional combined class4 CV log-loss uses hierarchical probs:
#       P(nonevent) = 1 - P(event)
#       P(Ia/Ib/II) = P(event) * P_SVM(Ia/Ib/II | X) renormalized over event classes
# ============================================================

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True, help="Path to train.csv")
    p.add_argument("--test", type=Path, required=True, help="Path to test.csv")
    p.add_argument("--out", type=Path, default=Path("submission.csv"))
    return p.parse_args()


# -----------------------
# Feature engineering
# -----------------------
def add_hyvonen_physics_features_strict(df_in: pd.DataFrame):
    df_fe = df_in.copy()
    eps = 1e-6

    cs_col = "CS.mean"

    rh_cols = [
        "RHIRGA42.mean",
        "RHIRGA84.mean",
        "RHIRGA168.mean",
        "RHIRGA336.mean",
        "RHIRGA504.mean",
        "RHIRGA672.mean",
    ]

    o3_cols = ["O342.mean", "O384.mean", "O3168.mean", "O3504.mean", "O3672.mean"]
    no_cols = ["NO42.mean", "NO84.mean", "NO168.mean", "NO336.mean", "NO504.mean", "NO672.mean"]
    nox_cols = ["NOx42.mean", "NOx84.mean", "NOx168.mean", "NOx336.mean", "NOx504.mean", "NOx672.mean"]
    so2_cols = ["SO2168.mean"]

    required = [cs_col] + rh_cols + o3_cols + no_cols + nox_cols + so2_cols
    missing = [c for c in required if c not in df_fe.columns]
    if missing:
        raise KeyError(f"Missing required columns for engineered features: {missing}")

    df_fe["log_CS"] = np.log(df_fe[cs_col].clip(lower=eps))

    df_fe["RH_mean_all"] = df_fe[rh_cols].mean(axis=1)
    df_fe["RH_std_all"] = df_fe[rh_cols].std(axis=1)

    df_fe["RH_gt_77"] = (df_fe["RH_mean_all"] > 77).astype(int)
    df_fe["logCS_gt_-5p5"] = (df_fe["log_CS"] > -5.5).astype(int)

    df_fe["Pnucl_lit"] = 1.0 / (1.0 + np.exp(1.7 * df_fe["log_CS"] + 0.13 * df_fe["RH_mean_all"]))

    df_fe["O3_mean_all"] = df_fe[o3_cols].mean(axis=1)
    df_fe["NO_mean_all"] = df_fe[no_cols].mean(axis=1)
    df_fe["NOx_mean_all"] = df_fe[nox_cols].mean(axis=1)
    df_fe["SO2_mean_all"] = df_fe[so2_cols].mean(axis=1)

    engineered_cols = [
        "log_CS",
        "RH_mean_all",
        "RH_std_all",
        "RH_gt_77",
        "logCS_gt_-5p5",
        "Pnucl_lit",
        "O3_mean_all",
        "NO_mean_all",
        "NOx_mean_all",
        "SO2_mean_all",
    ]

    return df_fe, engineered_cols


# -----------------------
# CV helpers (NO globals)
# -----------------------
def combined_cv_predict_hard_strict_gate_full(
    X_all: np.ndarray,
    y2: np.ndarray,
    y4: np.ndarray,
    stage1_model,
    stage2_model,
    cv: StratifiedKFold,
    classes4_order,
    int_to_event,
    threshold: float = 0.5,
) -> np.ndarray:
    y4_pred_all = np.empty(len(y4), dtype=object)

    for train_idx, val_idx in cv.split(X_all, y2):
        X_tr, X_val = X_all[train_idx], X_all[val_idx]
        y2_tr = y2[train_idx]
        y4_tr = y4[train_idx]

        # Fit Stage 1 on train fold
        m1 = clone(stage1_model)
        m1.fit(X_tr, y2_tr)

        p_event = m1.predict_proba(X_val)[:, 1]
        is_event = p_event >= threshold

        # Default: nonevent
        pred = np.array(["nonevent"] * len(val_idx), dtype=object)

        # Fit Stage 2 on FULL class4 labels
        m2 = clone(stage2_model)
        m2.fit(X_tr, y4_tr)

        if np.any(is_event):
            probs2 = m2.predict_proba(X_val[is_event])
            cls2 = m2.named_steps["svm"].classes_

            # Reorder to classes4_order
            idx = [list(cls2).index(c) for c in classes4_order]
            probs2 = probs2[:, idx]

            # Choose only among event classes (Ia/Ib/II)
            event_argmax = np.argmax(probs2[:, :3], axis=1)
            pred[is_event] = np.array([int_to_event[i] for i in event_argmax], dtype=object)

        y4_pred_all[val_idx] = pred

    return y4_pred_all


def combined_cv_predict_proba_strict_gate_full(
    X_all: np.ndarray,
    y2: np.ndarray,
    y4: np.ndarray,
    stage1_model,
    stage2_model,
    cv: StratifiedKFold,
    classes4_order,
) -> np.ndarray:
    """
    Returns hierarchical 4-class probs in order:
      [Ia, Ib, II, nonevent]

    Stage2 is trained on FULL 4-class labels,
    but we convert its output to P(subtype|event) by
    renormalizing over Ia/Ib/II.
    """
    proba_all = np.zeros((len(y4), 4), dtype=float)

    for train_idx, val_idx in cv.split(X_all, y2):
        X_tr, X_val = X_all[train_idx], X_all[val_idx]
        y2_tr = y2[train_idx]
        y4_tr = y4[train_idx]

        # Stage 1
        m1 = clone(stage1_model)
        m1.fit(X_tr, y2_tr)
        p_event = m1.predict_proba(X_val)[:, 1]

        # Stage 2 (FULL)
        m2 = clone(stage2_model)
        m2.fit(X_tr, y4_tr)

        probs2 = m2.predict_proba(X_val)
        cls2 = m2.named_steps["svm"].classes_
        idx = [list(cls2).index(c) for c in classes4_order]
        probs2 = probs2[:, idx]

        # Renormalize stage2 event probabilities
        event_probs = probs2[:, :3]
        denom = event_probs.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        cond_event = event_probs / denom

        # Hierarchical combine (Stage 1 controls nonevent probability)
        proba_all[val_idx, 0] = p_event * cond_event[:, 0]
        proba_all[val_idx, 1] = p_event * cond_event[:, 1]
        proba_all[val_idx, 2] = p_event * cond_event[:, 2]
        proba_all[val_idx, 3] = 1.0 - p_event

    # Safety normalize
    rs = proba_all.sum(axis=1, keepdims=True)
    proba_all = np.divide(proba_all, rs, where=rs != 0)

    return proba_all


def main():
    # -----------------------
    # 0) Parse args and load data
    # -----------------------
    args = parse_args()

    if not args.train.exists():
        raise FileNotFoundError(f"Train file not found: {args.train}")
    if not args.test.exists():
        raise FileNotFoundError(f"Test file not found: {args.test}")

    df = pd.read_csv(args.train)
    df_test = pd.read_csv(args.test)

    # -----------------------
    # 1) Auto baseline feature columns
    # -----------------------
    non_feature_cols = {"class4", "class2", "id", "date", "datetime", "time", "partlybad"}

    feature_cols = [
        c
        for c in df.columns
        if c not in non_feature_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    print("Baseline feature count:", len(feature_cols))
    print("First 15 baseline features:", feature_cols[:15])

    # -----------------------
    # 2) Labels + mappings
    # -----------------------
    y4 = df["class4"].values
    y2 = np.where(y4 == "nonevent", 0, 1).astype(int)

    classes4_order = ["Ia", "Ib", "II", "nonevent"]
    int_to_event = {0: "Ia", 1: "Ib", 2: "II"}

    # -----------------------
    # 3) Feature engineering (train + test)
    # -----------------------
    df_fe, engineered_cols = add_hyvonen_physics_features_strict(df)
    df_test_fe, _ = add_hyvonen_physics_features_strict(df_test)

    feature_cols_hyv = feature_cols + engineered_cols
    print("Engineered feature count:", len(engineered_cols))
    print("Total baseline+engineered:", len(feature_cols_hyv))

    X_hyv = df_fe[feature_cols_hyv].values
    X_hyv_test = df_test_fe[feature_cols_hyv].values

    # -----------------------
    # 4) Models
    # -----------------------
    lr_gate = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", max_iter=2000),
            ),
        ]
    )

    svm_full = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=0),
            ),
        ]
    )

    # -----------------------
    # 5) CV setup
    # -----------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    # -----------------------
    # 6) Stage 1 CV metrics (class2)
    # -----------------------
    probs_stage1 = cross_val_predict(
        lr_gate, X_hyv, y2, cv=cv, method="predict_proba", n_jobs=-1
    )

    p_event_all = probs_stage1[:, 1]
    y2_pred = (p_event_all >= 0.5).astype(int)

    stage1_acc = accuracy_score(y2, y2_pred)
    stage1_ll = log_loss(y2, probs_stage1)
    stage1_perplex = float(np.exp(stage1_ll))

    print("\nStage 1 (LR gate | baseline+engineered)")
    print("  CV accuracy:", stage1_acc)
    print("  CV log-loss:", stage1_ll)
    print("  CV perplexity:", stage1_perplex)

    # -----------------------
    # 7) Threshold sweep for best combined class4 accuracy
    # -----------------------
    thresholds = np.linspace(0.3, 0.7, 9)
    best_acc, best_t = -1.0, 0.5

    for t in thresholds:
        y4_pred = combined_cv_predict_hard_strict_gate_full(
            X_hyv,
            y2,
            y4,
            lr_gate,
            svm_full,
            cv,
            classes4_order,
            int_to_event,
            threshold=float(t),
        )
        acc = accuracy_score(y4, y4_pred)
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)

    print("\nBest gate threshold by strict class4 accuracy:", best_t)
    print("Combined class4 CV accuracy (strict gate):", best_acc)

    # Final strict-gate reports at best threshold
    y4_pred_best = combined_cv_predict_hard_strict_gate_full(
        X_hyv,
        y2,
        y4,
        lr_gate,
        svm_full,
        cv,
        classes4_order,
        int_to_event,
        threshold=best_t,
    )

    print("\nConfusion matrix (strict gate):")
    print(confusion_matrix(y4, y4_pred_best, labels=classes4_order))

    print("\nClassification report (strict gate):")
    print(
        classification_report(
            y4, y4_pred_best, labels=classes4_order, target_names=classes4_order
        )
    )

    # Combined class4 log-loss from hierarchical probs
    probs4 = combined_cv_predict_proba_strict_gate_full(
        X_hyv, y2, y4, lr_gate, svm_full, cv, classes4_order
    )
    ll4 = log_loss(y4, probs4, labels=classes4_order)
    perplex4 = float(np.exp(ll4))

    print("\nCombined class4 CV log-loss (strict gate, hierarchical probs):", ll4)
    print("Combined class4 CV perplexity:", perplex4)

    # -----------------------
    # 8) Fit full models + create submission
    # -----------------------
    lr_gate.fit(X_hyv, y2)
    svm_full.fit(X_hyv, y4)

    p_event_test = lr_gate.predict_proba(X_hyv_test)[:, 1]
    is_event_test = p_event_test >= best_t

    final_class4 = np.array(["nonevent"] * len(df_test), dtype=object)

    if np.any(is_event_test):
        probs2_test = svm_full.predict_proba(X_hyv_test[is_event_test])
        cls2 = svm_full.named_steps["svm"].classes_
        idx = [list(cls2).index(c) for c in classes4_order]
        probs2_test = probs2_test[:, idx]

        event_argmax = np.argmax(probs2_test[:, :3], axis=1)
        final_class4[is_event_test] = np.array(
            [int_to_event[i] for i in event_argmax], dtype=object
        )

    submission = pd.DataFrame(
        {
            "id": df_test["id"].values if "id" in df_test.columns else np.arange(len(df_test)),
            "class4": final_class4,
            "p": p_event_test,  # ONLY class2 probability from Stage 1
        }
    )

    print("\nSubmission preview:")
    print(submission.head())
    print("Rows:", len(submission))

    submission.to_csv(args.out, index=False)
    print("Saved submission to:", args.out)


if __name__ == "__main__":
    main()
