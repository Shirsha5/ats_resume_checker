# import os
# import pickle
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import (
#     train_test_split, StratifiedKFold,
#     RandomizedSearchCV, cross_val_score
# )
# from sklearn.metrics import classification_report

# # 1. Load and prepare data
# df = pd.read_csv('training_data_expanded.csv')
# df = df.drop(columns=['filename'])       # remove non-numeric column
# X = df.drop(columns=['hire_decision'])
# y = df['hire_decision']

# # 2. Train/test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# from imblearn.over_sampling import SMOTE
# sm = SMOTE(random_state=42)
# X_train, y_train = sm.fit_resample(X_train, y_train)


# # 3. Base model
# base_clf = RandomForestClassifier(random_state=42, class_weight='balanced')

# from sklearn.feature_selection import RFECV
# selector = RFECV(estimator=base_clf, cv=5, scoring='accuracy', n_jobs=-1)
# selector.fit(X_train, y_train)
# X_train = selector.transform(X_train)
# X_test  = selector.transform(X_test)
# selected_features = X.columns[selector.support_].tolist()
# print("Selected features:", selected_features)


# # 4. Hyperparameter search space
# param_dist = {
#     'n_estimators': [100,200,300,400,500],
#     'max_depth': [None, 10,20,30],
#     'min_samples_split': [2,5,10],
#     'min_samples_leaf': [1,2,4],
#     'max_features': ['sqrt','log2',0.3,0.5],
#     'bootstrap': [True, False]
# }

# # 5. Randomized search with 5‐fold Stratified CV
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# random_search = RandomizedSearchCV(
#     base_clf, param_distributions=param_dist,
#     n_iter=30, cv=cv, scoring='accuracy',
#     n_jobs=-1, verbose=2, random_state=42
# )
# random_search.fit(X_train, y_train)

# # 6. Best model
# best_clf = random_search.best_estimator_
# print("Best hyperparameters:", random_search.best_params_)
# print("CV accuracy: %.4f ± %.4f" % (
#     random_search.cv_results_['mean_test_score'][random_search.best_index_],
#     random_search.cv_results_['std_test_score'][random_search.best_index_]
# ))

# # 7. Evaluate on test set
# y_pred = best_clf.predict(X_test)
# print("\nTest set performance:")
# print(classification_report(y_test, y_pred))

# # 8. Cross‐validation scores on full training set
# cv_scores = cross_val_score(best_clf, X_train, y_train, cv=cv, scoring='accuracy')
# print("Stratified 5-fold CV accuracy: %.4f ± %.4f" % (cv_scores.mean(), cv_scores.std()*2))

# # 9. Feature importance
# importances = best_clf.feature_importances_
# feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
# print("\nTop 10 features by importance:")
# print(feat_imp.head(10))

# # 10. Save the model and feature list
# os.makedirs('ml_models', exist_ok=True)
# model_package = {'model': best_clf, 'selected_features': list(X.columns)}
# with open('ml_models/simple_classifier.pkl', 'wb') as f:
#     pickle.dump(model_package, f)
# print("\n✅ Improved model saved to ml_models/simple_classifier.pkl")

# train_simple_classifier_improved.py
"""
Enhanced Random Forest Training Script with SMOTE, RFECV, LightGBM, Early Stopping, and Hold-Out Set
"""

import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_score
)
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFECV
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

# 1. Load CSV and drop filename
df = pd.read_csv('training_data_expanded.csv')  # Use your expanded dataset
if 'filename' in df.columns:
    df.drop(columns=['filename'], inplace=True)

X = df.drop(columns=['hire_decision'])
y = df['hire_decision']

# 2. Create hold-out set (10%)
X_temp, X_holdout, y_temp, y_holdout = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y
)

# 3. Split remaining into train/test (20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_temp, y_temp, test_size=0.20, random_state=42, stratify=y_temp
)

# 4. Balance classes with SMOTE (preserve DataFrame structure)
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

# Convert back to DataFrame to preserve column names
X_train = pd.DataFrame(X_train_resampled, columns=X_train.columns)
y_train = pd.Series(y_train_resampled, name='hire_decision')

# # 5. RFECV feature selection (preserve feature names)
# base_clf = lgb.LGBMClassifier(
#     n_estimators=100,
#     learning_rate=0.1,
#     max_depth=8,
#     num_leaves=31,
#     random_state=42,
#     verbose=-1  # Suppress warnings
# )

# selector = RFECV(
#     estimator=base_clf,
#     cv=3,  # Reduce CV folds to speed up
#     scoring='accuracy',
#     n_jobs=-1
# )
# selector.fit(X_train, y_train)

# 5. FORCED feature selection (no RFECV - use specific features)
# Force these 4 features instead of automatic selection
FORCED_FEATURES = ['cgpa', 'academic_year', 'company_law', 'contract_law']

# Check if all forced features exist in data
available_features = X_train.columns.tolist()
selected_features = [f for f in FORCED_FEATURES if f in available_features]

if len(selected_features) < len(FORCED_FEATURES):
    missing = [f for f in FORCED_FEATURES if f not in available_features]
    print(f"⚠️ Missing features: {missing}")
    print(f"📋 Available features: {available_features}")

print(f"🎯 FORCED features: {selected_features}")

# # Get selected feature names BEFORE transformation
# selected_features = X_train.columns[selector.support_].tolist()
# print("🔍 Selected features (RFECV):", selected_features)

# Transform to selected features but keep as DataFrames
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
X_holdout_selected = X_holdout[selected_features]

# 6. Hyperparameter tuning (RandomizedSearchCV)
param_dist = {
    'n_estimators': [100,200,300],
    'max_depth': [6,8,10],
    'min_child_samples': [10,20],
    'num_leaves': [31,50],
    'subsample': [0.8,1.0],
    'colsample_bytree': [0.8,1.0]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(
    lgb.LGBMClassifier(random_state=42, verbose=-1),
    param_distributions=param_dist,
    n_iter=15,  # Reduce iterations to speed up
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0,  # Suppress output
    random_state=42
)

print("🔧 Training optimized LightGBM model...")
random_search.fit(X_train_selected, y_train)

best_clf = random_search.best_estimator_
print("✅ Best hyperparameters:", random_search.best_params_)
print("📊 CV accuracy: %.4f ± %.4f" % (
    random_search.cv_results_['mean_test_score'][random_search.best_index_],
    random_search.cv_results_['std_test_score'][random_search.best_index_]*2
))

# 7. Test set evaluation
y_pred = best_clf.predict(X_test_selected)
print("\n🧪 Test set performance:")
print(classification_report(y_test, y_pred))

# 8. Hold-out set evaluation
holdout_acc = best_clf.score(X_holdout_selected, y_holdout)
print(f"\n🎯 Hold-out accuracy: {holdout_acc:.4f}")

# 9. Stratified CV on training data
cv_scores = cross_val_score(
    best_clf, X_train_selected, y_train,
    cv=cv, scoring='accuracy'
)
print("🔄 Stratified CV accuracy: %.4f ± %.4f" % (
    cv_scores.mean(), cv_scores.std()*2
))

# 10. Feature importance (now with proper names)
importances = best_clf.feature_importances_
imp_series = pd.Series(importances, index=selected_features)
imp_series = imp_series.sort_values(ascending=False)
print("\n🌟 Top feature importances:")
print(imp_series)

# 11. Save model and feature list
os.makedirs('ml_models', exist_ok=True)
model_package = {
    'model': best_clf,
    'selected_features': selected_features
}

with open('ml_models/simple_classifier.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print(f"\n💾 Model saved to ml_models/simple_classifier.pkl")
print(f"📈 Selected {len(selected_features)} features from {len(X.columns)} original features")
print(f"🎯 Final hold-out accuracy: {holdout_acc:.4f}")
