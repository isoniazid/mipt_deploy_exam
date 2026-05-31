import pandas as pd
import joblib

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from datetime import datetime

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix
)

# 1. DB CONNECT

DB_URL = f"postgresql://developer:vasich_dev_pg@185.222.161.242:5432/mushrooms"
engine = create_engine(DB_URL)

# 2. LOAD DATA

query = """
SELECT
    cap_texture,
    spore_pattern,
    stem_flexibility,
    ring_thickness,
    cap_shape,
    cap_surface,
    cap_color,
    stalk_shape,
    veil_type,
    veil_color,
    ring_number,
    population,
    habitat,
    class
FROM feature_store;
"""

df = pd.read_sql(query, engine)

# 3. TARGET ENCODING

# p = poisonous = 1 (positive class)
# e = edible = 0 (negative class)

df["class"] = df["class"].map({
    "p": 1,
    "e": 0,
    "c": 1
})

X = df.drop(columns=["class"])
y = df["class"]

# 4. FEATURES

ordinal_features = [
    'stem_flexibility',
    'ring_thickness',
    'ring_number',
    'population'
]

nominal_features = [
    'cap_texture',
    'spore_pattern',
    'cap_shape',
    'cap_surface',
    'cap_color',
    'stalk_shape',
    'veil_type',
    'veil_color',
    'habitat'
]

ordinal_categories_list = [
    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j','s'],
    ['a', 'b', 'c', 'r'],
    ['n', 'o', 't', 'r'],
    ['y', 'v', 's', 'n', 'c', 'a', 'p']
]

# 5. PREPROCESSING

preprocessor = ColumnTransformer([
    (
        "ord",
        OrdinalEncoder(categories=ordinal_categories_list),
        ordinal_features
    ),
    (
        "oh",
        OneHotEncoder(handle_unknown="ignore"),
        nominal_features
    )
])

# 6. SPLIT

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    train_size=0.8,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

# 7. MODEL

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", model)
])

# 8. TRAIN

pipeline.fit(X_train, y_train)

# 9. PREDICTIONS

TOXIC_THRESHOLD = 0.5

y_proba = pipeline.predict_proba(X_test)[:, 1]

y_pred = (y_proba >= TOXIC_THRESHOLD).astype(int)

# 10. METRICS

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

TPR = tp / (tp + fn)  # recall (toxic)
TNR = tn / (tn + fp)  # specificity

ROC_AUC = roc_auc_score(y_test, y_proba)
F1 = f1_score(y_test, y_pred)

print("\n=== METRICS ===")
print(f"TPR (Recall toxic): {TPR:.4f}")
print(f"TNR (Specificity): {TNR:.4f}")
print(f"ROC-AUC: {ROC_AUC:.4f}")
print(f"F1-score: {F1:.4f}")


# 11. SAVE ARTIFACT

joblib.dump(pipeline, f"mushroom_xgb_{datetime.now().strftime('%Y-%m-%d %H_%M_%S')}.pkl")

print("\nPipeline saved successfully.")