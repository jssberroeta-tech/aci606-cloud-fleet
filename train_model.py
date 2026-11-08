"""Entrenamiento: EDA -> GroupShuffleSplit por fleet_id -> RF vs GB -> model.joblib."""
import glob
import json
import os
import subprocess
import sys
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

SEED = 42
CLASSES = ["NORMAL", "WARNING", "CRITICAL"]
CAND_FEATURES = ["speed", "battery", "motor_temp", "vibration", "motor_current", "obstacle_distance"]
LABEL_CANDIDATES = ["status", "label", "condition", "target", "predicted_status", "robot_status", "y_true"]
REPO = "https://github.com/DrOscarL/aci606-cloud-fleet-challenge.git"
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)


# ------------------------------------------------------------------ datos
def synthetic_fleets(n_fleets=12, per_fleet=500, seed=SEED) -> pd.DataFrame:
    """Respaldo: flotas sintéticas con ruido/offset por flota (solo si no hay dataset)."""
    rng = np.random.default_rng(seed)
    frames = []
    for f in range(n_fleets):
        off = rng.normal(0, 1, 6) * np.array([0.05, 2.0, 2.0, 0.03, 0.2, 0.05])
        n = per_fleet
        wear = rng.beta(1.2, 4.0, n)  # desgaste latente
        df = pd.DataFrame({
            "fleet_id": f"F{f:02d}", "robot_id": [f"R{(i % 5) + 1:02d}" for i in range(n)],
            "timestamp": 1.78e9 + np.arange(n),
            "x": rng.uniform(0, 20, n), "y": rng.uniform(0, 20, n),
            "speed": np.clip(rng.normal(0.7, 0.2, n) + off[0], 0, None),
            "battery": np.clip(100 - 90 * rng.random(n) + off[1], 0, 100),
            "motor_temp": 40 + 45 * wear + rng.normal(0, 3, n) + off[2],
            "vibration": 0.2 + 1.2 * wear + rng.normal(0, 0.08, n) + off[3],
            "motor_current": 3 + 4 * wear + rng.normal(0, 0.5, n) + off[4],
            "obstacle_distance": np.clip(rng.gamma(2.0, 0.6, n) + off[5], 0.02, None),
        })
        score = (df.motor_temp - 40) / 45 + df.vibration / 1.4 + (df.battery < 15) * 0.6 + (df.obstacle_distance < 0.15) * 0.4
        df["status"] = np.where(score > 1.55, "CRITICAL", np.where(score > 0.9, "WARNING", "NORMAL"))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_dataset(n_fleets: int = 8, samples: int = 3000) -> pd.DataFrame:
    """data/*.csv del repo base. Si no hay, el starter se clona y se generan n flotas (una semilla por flota)."""
    files = sorted(glob.glob("data/**/*.csv", recursive=True))
    if not files:
        try:
            print("data/ vacío -> clonando starter y generando flotas con simulator/generate_dataset.py…")
            if not os.path.isdir("_starter"):
                subprocess.run(["git", "clone", "--depth", "1", REPO, "_starter"], check=True, capture_output=True, timeout=180)
            for i in range(n_fleets):
                out = os.path.abspath(f"data/fleet_{i:02d}.csv")
                subprocess.run([sys.executable, "simulator/generate_dataset.py", "--samples", str(samples),
                                "--seed", str(606 + 101 * i), "--output", out],
                               cwd="_starter", check=True, capture_output=True, timeout=300)
            files = sorted(glob.glob("data/**/*.csv", recursive=True))
        except Exception as exc:
            print("No se pudo generar el dataset con el starter:", exc)
    if files:
        print("Archivos de datos:", files)
        frames = []
        for f in files:
            d = pd.read_csv(f)
            if "fleet_id" not in d.columns:  # una flota por archivo
                d["fleet_id"] = os.path.splitext(os.path.basename(f))[0]
            frames.append(d)
        return pd.concat(frames, ignore_index=True)
    print("⚠ Sin dataset real: usando flotas sintéticas (reemplaza por data/*.csv del starter).")
    df = synthetic_fleets()
    df.to_csv("data/synthetic_fleets.csv", index=False)
    return df


df = load_dataset()
label_col = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
if label_col is None:
    raise SystemExit(f"No encuentro la columna de etiqueta entre {LABEL_CANDIDATES}. Columnas: {list(df.columns)}")
group_col = "fleet_id" if "fleet_id" in df.columns else ("robot_id" if "robot_id" in df.columns else None)
if group_col is None:
    raise SystemExit("No existe fleet_id ni robot_id para separar por grupos.")
df[label_col] = df[label_col].astype(str).str.strip().str.upper()
df = df[df[label_col].isin(CLASSES)].copy()
features = [c for c in CAND_FEATURES if c in df.columns]
for c in features:
    df[c] = pd.to_numeric(df[c], errors="coerce")
print(f"\nEtiqueta={label_col} | grupo={group_col} | features={features}")

# ------------------------------------------------------------------ EDA
print("\n=== EDA ===")
print("Forma:", df.shape, "| grupos (flotas):", df[group_col].nunique())
print("Nulos por feature:\n", df[features].isna().sum().to_string())
print("\nResumen estadístico:\n", df[features].describe().round(3).T.to_string())
dist = df[label_col].value_counts().reindex(CLASSES).fillna(0).astype(int)
print("\nDistribución de clases:\n", pd.DataFrame({"n": dist, "%": (100 * dist / dist.sum()).round(2)}).to_string())

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
dist.plot.bar(ax=ax[0], color=["#2e7d4f", "#c98a00", "#c0392b"], rot=0, title="Distribución de clases")
(df.groupby([group_col, label_col]).size().unstack(fill_value=0).reindex(columns=CLASSES)
   .plot.bar(stacked=True, ax=ax[1], color=["#2e7d4f", "#c98a00", "#c0392b"], title=f"Clases por {group_col}", legend=False))
plt.tight_layout(); plt.savefig("models/eda_class_distribution.png", dpi=120); plt.show()
print("\nMedia por clase:\n", df.groupby(label_col)[features].mean().reindex(CLASSES).round(3).to_string())

# ------------------------------------------------------------------ split por flota
X, y, g = df[features], df[label_col], df[group_col]
if g.nunique() < 2:
    raise SystemExit("Se necesitan al menos 2 grupos para GroupShuffleSplit.")
for rs in range(SEED, SEED + 200):  # primera semilla cuyo test contiene las 3 clases y train también
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=rs).split(X, y, groups=g))
    if y.iloc[te].nunique() == 3 and y.iloc[tr].nunique() == 3 and y.iloc[te].value_counts().min() >= 10:
        break
print("random_state del split:", rs)
Xtr, Xte, ytr, yte, gtr = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], g.iloc[tr]
assert set(g.iloc[tr]).isdisjoint(set(g.iloc[te])), "fuga: grupos compartidos entre train y test"
print(f"\nTrain: {len(Xtr)} filas / {gtr.nunique()} grupos | Test: {len(Xte)} filas / {g.iloc[te].nunique()} grupos (disjuntos)")
print("Clases train:", ytr.value_counts().to_dict(), "| test:", yte.value_counts().to_dict())


# ------------------------------------------------------------------ modelos
def make_pipe(clf):
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", clf)])


candidates = {
    "RandomForest": (make_pipe(RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                                     n_jobs=-1, random_state=SEED)), {}),
    "GradientBoosting": (make_pipe(GradientBoostingClassifier(n_estimators=250, learning_rate=0.08, max_depth=3,
                                                              random_state=SEED)), "sw"),
}


def fit(pipe, Xa, ya, sw):
    kw = {"clf__sample_weight": compute_sample_weight("balanced", ya)} if sw == "sw" else {}
    return pipe.fit(Xa, ya, **kw)


from sklearn.base import clone  # noqa: E402

k = max(2, min(5, gtr.nunique()))
cv_scores, results = {}, {}
print(f"\n=== Comparación (GroupKFold k={k} en train; test = flotas no vistas) ===")
for name, (pipe, sw) in candidates.items():
    folds = []
    for a, b in GroupKFold(n_splits=k).split(Xtr, ytr, groups=gtr):
        m = fit(clone(pipe), Xtr.iloc[a], ytr.iloc[a], sw)
        folds.append(f1_score(ytr.iloc[b], m.predict(Xtr.iloc[b]), average="macro", labels=CLASSES, zero_division=0))
    cv_scores[name] = float(np.mean(folds))
    t0 = time.perf_counter()
    m = fit(clone(pipe), Xtr, ytr, sw)
    pred = m.predict(Xte)
    results[name] = {"model": m, "pred": pred, "f1": f1_score(yte, pred, average="macro", labels=CLASSES, zero_division=0),
                     "ms": (time.perf_counter() - t0) * 1000}
    print(f"{name:17s} CV F1 macro={cv_scores[name]:.4f} ± {np.std(folds):.4f} | TEST F1 macro={results[name]['f1']:.4f}")

majority = yte.map(lambda _: ytr.mode()[0])
base_f1 = f1_score(yte, majority, average="macro", labels=CLASSES, zero_division=0)
print(f"Línea base clase mayoritaria ({ytr.mode()[0]}): F1 macro={base_f1:.4f}")

best = max(cv_scores, key=cv_scores.get)  # se elige por CV en train, no por el test
print(f"\n>>> Modelo elegido: {best} (mayor F1 macro en validación cruzada por grupos)")
bp = results[best]["pred"]
cm = confusion_matrix(yte, bp, labels=CLASSES)
print("\nMatriz de confusión (filas=real, columnas=predicho):\n", pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())
print("\n", classification_report(yte, bp, labels=CLASSES, digits=4, zero_division=0))

# F1_macro = (1/K) * sum_k F1_k, con F1_k = 2 P_k R_k / (P_k + R_k)
tp = np.diag(cm).astype(float)
P = np.divide(tp, cm.sum(axis=0), out=np.zeros(len(CLASSES)), where=cm.sum(axis=0) > 0)
R = np.divide(tp, cm.sum(axis=1), out=np.zeros(len(CLASSES)), where=cm.sum(axis=1) > 0)
F1k = np.divide(2 * P * R, P + R, out=np.zeros(len(CLASSES)), where=(P + R) > 0)
f1_manual = float(F1k.mean())
print(f"F1_k = {dict(zip(CLASSES, F1k.round(4)))}")
print(f"F1_macro manual = {f1_manual:.4f} | sklearn = {results[best]['f1']:.4f}")
assert abs(f1_manual - results[best]["f1"]) < 1e-9

fig, ax = plt.subplots(figsize=(4.6, 4))
ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"{best} · F1 macro={f1_manual:.3f}"); plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=120); plt.show()

# ------------------------------------------------------------------ pipeline final
pipe_final, sw = candidates[best]
final = fit(clone(pipe_final), X, y, sw)  # reentrena con todas las flotas para despliegue
joblib.dump(final, "models/model.joblib", compress=3)
meta = {
    "model_version": "v1", "algorithm": best, "features": features, "classes": list(map(str, final.classes_)),
    "label_column": label_col, "group_column": group_col, "seed": SEED,
    "metrics_unseen_fleets": {"f1_macro": round(f1_manual, 4), "baseline_majority_f1_macro": round(float(base_f1), 4),
                              "cv_f1_macro": {k_: round(v, 4) for k_, v in cv_scores.items()}},
    "sklearn_version": sklearn.__version__, "numpy_version": np.__version__, "pandas_version": pd.__version__,
    "joblib_version": joblib.__version__, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "justification": ("Selección por F1 macro en GroupKFold sobre flotas de train; evaluación en flotas no vistas. "
                      "F1 macro pondera igual a CRITICAL (clase minoritaria)."),
}
with open("models/model_meta.json", "w", encoding="utf-8") as fh:
    json.dump(meta, fh, indent=2, ensure_ascii=False)
print("\n✔ Guardado models/model.joblib y models/model_meta.json")
print(json.dumps(meta["metrics_unseen_fleets"], indent=2))
