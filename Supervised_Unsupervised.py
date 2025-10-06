# %% [markdown]
# # 1. Dataset, libraries upload

# %%
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

# %%
data = open('/content/drive/MyDrive/Colab Notebooks/ML2/Data/data.txt').readlines()
data = [i.split() for i in data]

column_names = [
    "pelvic_incidence",
    "pelvic_tilt",
    "lumbar_lordosis_angle",
    "sacral_slope",
    "pelvic_radius",
    "grade_of_spondylolisthesis",
    "class"
]

data = pd.DataFrame(columns = column_names, data = data)
data.head()

# %%
print(data.info())
print(data.describe())
data.isnull().sum()

# %% [markdown]
# * No missing values

# %% [markdown]
# # 2. Data preprocessing

# %%
data["class"].unique()

# %%
data.loc[data['class'] == 'NO',['class']] = 0
data.loc[data['class'] == 'AB',['class']] = 1
for i in column_names:
  data[i] = pd.to_numeric(data[i])

data.head()

# %% [markdown]
# # 3. EDA

# %% [markdown]
# ## 3.1. Distribution of vairables

# %%
# Column name mapping for display
column_name_mapping = {
    "pelvic_incidence": "Pelvic Incidence",
    "pelvic_tilt": "Pelvic Tilt",
    "lumbar_lordosis_angle": "Lumbar Lordosis Angle",
    "sacral_slope": "Sacral Slope",
    "pelvic_radius": "Pelvic Radius",
    "grade_of_spondylolisthesis": "Grade of Spondylolisthesis"
}

features = [col for col in data.columns if col != 'class']
colors = sns.color_palette("husl", len(features))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, column in enumerate(features):
    data[column].plot.hist(
        ax=axes[i],
        bins=20,
        edgecolor='black',
        color=colors[i],
        alpha=0.8
    )
    fixedname = column_name_mapping[column]
    axes[i].set_title(f"Histogram of {fixedname}")
    axes[i].set_xlabel(fixedname)
    axes[i].set_ylabel("Frequency")
plt.suptitle("Histograms of Numerical Features", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# %%
column_names = ['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle',
       'sacral_slope', 'pelvic_radius', 'grade_of_spondylolisthesis', 'class']

numeric_features = column_names[:-1]
X = data[numeric_features].values
names = numeric_features
sns.pairplot(data, vars=numeric_features, hue="class", diag_kind="kde", palette="Set1", markers=["o", "s"])
plt.savefig("pre_stand.png")
plt.show()

# %%
scaler = StandardScaler()
data2 = data.copy()
data2[numeric_features] = scaler.fit_transform(data2[numeric_features])

numeric_features = column_names[:-1]
X = data2[numeric_features].values
names = numeric_features
sns.pairplot(data2, vars=numeric_features, hue="class", diag_kind="kde", palette="Set1", markers=["o", "s"])
plt.suptitle("Pairplot of Vertebral Column Data by Class", y=1.02)
plt.show()

# %%
boxplot_colors = ['#FF9999', '#99FF99', '#9999FF', '#FFCC99', '#CC99FF', '#66CCCC']

features = [col for col in data.columns if col != 'class']
plt.figure(figsize=(12, 6))
box = plt.boxplot(
    data[features].values,
    patch_artist=True,
    labels=[column_name_mapping[col] for col in features]
)

for patch, color in zip(box['boxes'], boxplot_colors):
    patch.set_facecolor(color)

plt.xticks(rotation=30, ha='right')  # Tilted x-axis labels for readability
plt.title("Boxplot of Numerical Features (Detecting Outliers)", fontsize=14)
plt.ylabel("Values")
plt.show()

# %%
plt.figure(figsize=(3, 4.5))
ax = sns.countplot(x="class", data=data, palette="coolwarm")
plt.title("Class Distribution(NO vs. AB)")

for p in ax.patches:
    height = p.get_height()
    ax.text(
        p.get_x() + p.get_width()/2.,
        height + 0.3,
        f'{int(height)}',
        ha='center',
        va='bottom'
    )
plt.xticks([0, 1], ['Normal', 'Abnormal'])

plt.show()

# %% [markdown]
# ## 3.2. Correlation analysis

# %%
# Feature variables
plt.figure(figsize=(8, 6))
sns.heatmap(data.drop(columns=["class"]).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# %%
# Including target variables
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# %% [markdown]
# ## 3.3. PCA Classification

# %%
X = data[numeric_features]
y = data['class']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['class'] = y

plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', data=pca_df)
plt.title("PCA of Vertebral Column Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# %% [markdown]
# Standardscaler
# * Mean ≈ 0
# Standard Deviation ≈ 1

# %% [markdown]
# # 4. PCA

# %%
# Scaling
values = data.drop(columns=["class"]).values
values = StandardScaler().fit_transform(values)

# %%
# PCA
from sklearn.decomposition import PCA

PCAthreshold = 0.9
Pca = PCA(n_components = PCAthreshold, svd_solver="full")
reducedData = Pca.fit_transform(values)
reducedData.shape

# %% [markdown]
# Setting: 90% variance -> 4 components

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

pca_full = PCA(svd_solver="full")
pca_full.fit(values)

explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

n_components = range(1, len(explained_variance_ratio) + 1)
plt.figure(figsize=(10, 6))

# Scree Plot
plt.subplot(1, 2, 1)
plt.plot(n_components, explained_variance_ratio, 'o-', linewidth=2, markersize=8)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)

# Accumulate
plt.subplot(1, 2, 2)
plt.plot(n_components, cumulative_variance_ratio, 'o-', linewidth=2, markersize=8)
plt.axhline(y=0.9, color='r', linestyle='--', label='90% Threshold')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# elbow point
components_needed_for_90 = np.argmax(cumulative_variance_ratio >= 0.9) + 1
print(f"Components needed for 90% variance: {components_needed_for_90}")
print("Explained variance by component:")
for i, ratio in enumerate(explained_variance_ratio[:10], 1):
    print(f"PC{i}: {ratio:.4f} ({cumulative_variance_ratio[i-1]:.4f} cumulative)")

# %% [markdown]
# # 5. Factoral Analysis

# %%
!pip install factor_analyzer

# %%
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import FactorAnalyzer

max_factors = min(6, X_scaled.shape[1])

variance_explained = []
cumulative_variance = []

for n_factors in range(1, max_factors + 1):
    fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
    fa.fit(X_scaled)

    variance = fa.get_factor_variance()[1]
    variance_explained.append(sum(variance))

    cumulative_variance.append(sum(variance))

plt.figure(figsize=(12, 6))

# Scree Plot
plt.subplot(1, 2, 1)
plt.plot(range(1, max_factors + 1), variance_explained, 'o-', linewidth=2, markersize=8)
plt.title('Scree Plot')
plt.xlabel('Number of Factors')
plt.ylabel('Variance Explained')
plt.grid(True)

# Accumulated
plt.subplot(1, 2, 2)
plt.plot(range(1, max_factors + 1), cumulative_variance, 'o-', linewidth=2, markersize=8)
plt.axhline(y=0.7, color='r', linestyle='--', label='90% Explained Variance')
plt.title('Cumulative Variance Explained')
plt.xlabel('Number of Factors')
plt.ylabel('Cumulative Variance')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

print("Variance explained by number of factors:")
for i, var in enumerate(variance_explained, 1):
    print(f"{i} factor(s): {var:.4f} ({cumulative_variance[i-1]:.4f} cumulative)")

# elbow method
variance_diff = np.diff(variance_explained)
variance_diff_rate = np.diff(variance_diff)
suggested_factors = np.argmin(variance_diff_rate) + 3

print(f"\nSuggested number of factors based on elbow method: {suggested_factors}")

# %% [markdown]
# # 6. Supervised Classification

# %%
X_scaled=pd.DataFrame(X_scaled, columns=data.columns[:-1])
data_scaled=pd.concat([X_scaled, data['class']], axis=1)
data_scaled

# %% [markdown]
# ### 6.1. K-nearst classification

# %% [markdown]
# #### 6.1.1. Original features

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from collections import Counter

# SMOTE (resampling)
train, test = train_test_split(data_scaled, test_size=0.2, random_state=42)
X_train, X_test = train.drop('class', axis=1), test.drop('class', axis=1)
y_train, y_test = train['class'], test['class']

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

from sklearn.model_selection import cross_val_score
k_values = range(5, 30, 2)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')   # imbalanced
    scores = cross_val_score(knn, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# optimal k
optimal_k = k_values[np.argmax(cv_scores)]
best_score = np.max(cv_scores)

print("Optimal k:", optimal_k)
print(f"Best cross-validation score: {best_score:.3f}")

# %% [markdown]
# The way to set the value of k (the number of nearest neighbors) in KNN generally depends on the size and distribution of the data.
# 
# ✅ Common Guidelines for Setting k:
# 
# Set k to an odd number: For binary classification (0/1), it is recommended to set k as an odd number to avoid ties.
# 
# √(number of data points): Empirically, it is suggested that k ≈ √(total number of samples).
# 
# For the current data with 310 samples:
# 
# 𝑘
# ≈
# 310
# ≈
# 17.6
# k≈
# 310
# ​
#  ≈17.6
# Small k value (Risk of Underfitting): If k is too small, the model becomes sensitive to noise (overfitting). (e.g., when k=1, the model only looks at the closest single data point).
# 
# Large k value (Risk of Underfitting): If k is too large, the model may overly simplify the patterns (underfitting), which will reduce its performance.

# %% [markdown]
# knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
# 
# * weights='distance' <br>
# When weights='distance' is set, closer data points receive higher weights.
# In other words, if minority class data points are close, they are not ignored by the majority class.
# The default value weights='uniform' gives equal weight to all neighbors, which is disadvantageous for imbalanced data

# %%
plt.figure(figsize=(8, 5))
plt.plot(k_values, cv_scores, marker='o', color='b', label="Cross-Validation Accuracy")
plt.axvline(optimal_k, color='r', linestyle='--', label=f"Optimal k={optimal_k}")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Cross-Validation Accuracy")
plt.title("KNN: Choosing the Optimal k")
plt.legend()
plt.grid(True)
plt.show()

# %%
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
knn = KNeighborsClassifier(n_neighbors=optimal_k, weights='distance')
knn.fit(X_train_balanced, y_train_balanced)
y_predict = knn.predict(X_test)

confusion = confusion_matrix(y_test, y_predict)
print("Confusion matrix:")
print(confusion)

# Scoring
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy:.3f}")

recall = recall_score(y_test, y_predict)
print(f"Recall: {recall:.3f}")

precision = precision_score(y_test, y_predict)
print(f"Precision: {precision:.3f}")

f1 = f1_score(y_test, y_predict)
print(f"F1-score: {f1:.3f}")

# %%
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap

# Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_balanced)  # Apply PCA to the training data

knn_vis = KNeighborsClassifier(n_neighbors=optimal_k, weights='distance')
knn_vis.fit(X_pca, y_train_balanced)

x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cmap_light = ListedColormap(["#FF9999", "#9999FF"])
cmap_bold = ["red", "blue"]

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

sns.scatterplot( x=X_pca[:, 0],y=X_pca[:, 1], hue=pd.Series(y_train_balanced).map({0: "Normal", 1: "Abnormal"}), palette=cmap_bold, edgecolor="k")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("KNN Decision Boundary (PCA Components)")

plt.tight_layout()
plt.show()

# %%
from matplotlib.colors import ListedColormap

feature_pairs = [
    ("pelvic_incidence", "sacral_slope"),
    ("grade_of_spondylolisthesis", "pelvic_radius"),
    ("pelvic_incidence", "grade_of_spondylolisthesis"),
    ("pelvic_incidence", "pelvic_tilt")]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (feature_x, feature_y) in zip(axes.flatten(), feature_pairs):
    X_vis = X_train_balanced[[feature_x, feature_y]].values
    y_vis = y_train_balanced.values

    knn_vis = KNeighborsClassifier(n_neighbors=optimal_k, weights='distance')
    knn_vis.fit(X_vis, y_vis)

    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(["#FF9999", "#9999FF"])
    cmap_bold = ["red", "blue"]

    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

    sns.scatterplot(x=X_vis[:, 0], y=X_vis[:, 1], hue=pd.Series(y_vis).map({0: "Normal", 1: "Abnormal"}), palette=cmap_bold, edgecolor="k", ax=ax)

    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_title(f"KNN Decision Boundary: {feature_x} vs {feature_y}")

plt.tight_layout()
plt.show()


# %% [markdown]
# #### 6.1.2. PCA

# %%
# train, test = train_test_split(data_scaled, test_size=0.2, random_state=42)
# X_train, X_test = train.drop('class', axis=1), test.drop('class', axis=1)
# y_train, y_test = train['class'], test['class']

# smote = SMOTE(random_state=42)
# X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

PCAthreshold = 0.9    # (or n_components = 4)
pca = PCA(n_components=PCAthreshold, svd_solver="full")
pca.fit(X_train_balanced)

X_train_pca = pca.transform(X_train_balanced)
X_test_pca = pca.transform(X_test)

print(f"Original features: {X_train_balanced.shape[1]}")
print(f"After PCA reduction features: {X_train_pca.shape[1]}")

k_values = range(5, 30, 2)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    scores = cross_val_score(knn, X_train_pca, y_train_balanced, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = k_values[np.argmax(cv_scores)]
best_score = np.max(cv_scores)

print("Optimal k:", optimal_k)
print(f"Best cross-validation score: {best_score:.3f}")

# %%
final_knn = KNeighborsClassifier(n_neighbors=optimal_k, weights='distance')
final_knn.fit(X_train_pca, y_train_balanced)
y_predict = final_knn.predict(X_test_pca)

confusion = confusion_matrix(y_test, y_predict)
print("Confusion matrix:")
print(confusion)

# Scoring
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy:.3f}")

recall = recall_score(y_test, y_predict)
print(f"Recall: {recall:.3f}")

precision = precision_score(y_test, y_predict)
print(f"Precision: {precision:.3f}")

f1 = f1_score(y_test, y_predict)
print(f"F1-score: {f1:.3f}")

# %%
from matplotlib.colors import ListedColormap

X_train_vis = X_train_pca[:, :2] # first 2 component
X_test_vis = X_test_pca[:, :2] # first 2 component
y_vis = y_train_balanced.values

final_knn_vis = KNeighborsClassifier(n_neighbors=optimal_k, weights='distance')
final_knn_vis.fit(X_train_vis, y_vis)

x_min, x_max = X_train_vis[:, 0].min() - 1, X_train_vis[:, 0].max() + 1
y_min, y_max = X_train_vis[:, 1].min() - 1, X_train_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

Z = final_knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cmap_light = ListedColormap(["#FF9999", "#9999FF"])
cmap_bold = ["red", "blue"]

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

sns.scatterplot(
    x=X_train_vis[:, 0],
    y=X_train_vis[:, 1],
    hue=pd.Series(y_vis).map({0: "Normal", 1: "Abnormal"}),
    palette=cmap_bold,
    edgecolor="k")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("KNN Decision Boundary (PCA Components)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.1.3. FA

# %%
# Train FA
num_factors = 3
fa = FactorAnalysis(n_components=num_factors, random_state=42)
fa.fit(X_train_balanced)

X_train_fa = fa.transform(X_train_balanced)
X_test_fa = fa.transform(X_test)

print(f"After FA features: {X_train_fa.shape[1]}")

k_values = range(5, 30, 2)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    scores = cross_val_score(knn, X_train_fa, y_train_balanced, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

optimal_k = k_values[np.argmax(cv_scores)]
best_score = np.max(cv_scores)

print("Optimal k:", optimal_k)
print(f"Best cross-validation score: {best_score:.3f}")

final_knn = KNeighborsClassifier(n_neighbors=optimal_k, weights='distance')
final_knn.fit(X_train_fa, y_train_balanced)
y_predict = final_knn.predict(X_test_fa)

confusion = confusion_matrix(y_test, y_predict)
print("Confusion matrix:")
print(confusion)

# Scoring
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy:.3f}")

recall = recall_score(y_test, y_predict)
print(f"Recall: {recall:.3f}")

precision = precision_score(y_test, y_predict)
print(f"Precision: {precision:.3f}")

f1 = f1_score(y_test, y_predict)
print(f"F1-score: {f1:.3f}")

# %%
X_train_vis = X_train_fa[:, :2] # first 2 component
X_test_vis = X_test_fa[:, :2] # first 2 component
y_vis = y_train_balanced.values

final_knn_vis = KNeighborsClassifier(n_neighbors=optimal_k, weights='distance')
final_knn_vis.fit(X_train_vis, y_vis)

x_min, x_max = X_train_vis[:, 0].min() - 1, X_train_vis[:, 0].max() + 1
y_min, y_max = X_train_vis[:, 1].min() - 1, X_train_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

Z = final_knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cmap_light = ListedColormap(["#FF9999", "#9999FF"])
cmap_bold = ["red", "blue"]

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

sns.scatterplot(
    x=X_train_vis[:, 0],
    y=X_train_vis[:, 1],
    hue=pd.Series(y_vis).map({0: "Normal", 1: "Abnormal"}),
    palette=cmap_bold,
    edgecolor="k")

plt.xlabel("FA Component 1")
plt.ylabel("FA Component 2")
plt.title("KNN Decision Boundary (FA Components)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ### 6.2. RandomForest Classification

# %% [markdown]
# #### 6.2.1. Original features

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [100, 150, 200], 'max_depth': [10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],}

rfc = RandomForestClassifier(class_weight='balanced', random_state=42)
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')  # same condition(cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print(f"Best Cross-Validation Score: {best_score:.3f}")

# %% [markdown]
# * class_weight='balanced': imbalanced coordination
# * cv = 5: same condition as knn

# %%
rfc = RandomForestClassifier(class_weight='balanced', random_state=42, **best_params)
rfc.fit(X_train, y_train)
y_predict = rfc.predict(X_test)

confusion = confusion_matrix(y_test, y_predict)
print("Confusion matrix:")
print(confusion)

# Scoring
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy:.3f}")

recall = recall_score(y_test, y_predict)
print(f"Recall: {recall:.3f}")

precision = precision_score(y_test, y_predict)
print(f"Precision: {precision:.3f}")

f1 = f1_score(y_test, y_predict)
print(f"F1-score: {f1:.3f}")

# %%
# Checking the feature importance
feature_importance = rfc.feature_importances_

features = X_train.columns
sorted_idx = feature_importance.argsort()[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance[sorted_idx], y=features[sorted_idx], palette="magma")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.show()

# %% [markdown]
# #### 6.2.2. PCA

# %%
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

rfc = RandomForestClassifier(class_weight='balanced', random_state=42)
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

grid_search.fit(X_train_pca, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print(f"Best Cross-Validation Score: {best_score:.3f}")

rfc = RandomForestClassifier(class_weight='balanced', random_state=42, **best_params)
rfc.fit(X_train_pca, y_train)
y_predict = rfc.predict(X_test_pca)

confusion = confusion_matrix(y_test, y_predict)
print("Confusion matrix:")
print(confusion)

# Scoring
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy:.3f}")

recall = recall_score(y_test, y_predict)
print(f"Recall: {recall:.3f}")

precision = precision_score(y_test, y_predict)
print(f"Precision: {precision:.3f}")

f1 = f1_score(y_test, y_predict)
print(f"F1-score: {f1:.3f}")

# %% [markdown]
# #### 6.2.3. FA

# %%
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

rfc = RandomForestClassifier(class_weight='balanced', random_state=42)
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

grid_search.fit(X_train_fa, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print(f"Best Cross-Validation Score: {best_score:.3f}")

rfc = RandomForestClassifier(class_weight='balanced', random_state=42, **best_params)
rfc.fit(X_train_fa, y_train)
y_predict = rfc.predict(X_test_fa)

confusion = confusion_matrix(y_test, y_predict)
print("Confusion matrix:")
print(confusion)

# Scoring
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy:.3f}")

recall = recall_score(y_test, y_predict)
print(f"Recall: {recall:.3f}")

precision = precision_score(y_test, y_predict)
print(f"Precision: {precision:.3f}")

f1 = f1_score(y_test, y_predict)
print(f"F1-score: {f1:.3f}")

# %% [markdown]
# ### 6.3. Support Vector Machine

# %% [markdown]
# #### 6.3.1. Original features

# %%
train, test = train_test_split(data_scaled, test_size=0.2, random_state=42)    # Scaled data
X_train, X_test = train.drop('class', axis=1), test.drop('class', axis=1)
y_train, y_test = train['class'], test['class']

# %%
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE     # imbalanced coordination
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# hyperparameter tunning
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.1, 1], 'kernel': ['rbf', 'linear']}

svm = SVC(class_weight='balanced', random_state=42)
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train_resampled, y_train_resampled)

best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

# %%
svm = SVC(class_weight='balanced', random_state=42, **best_params)
svm.fit(X_train_resampled, y_train_resampled)

y_predict = svm.predict(X_test)

confusion = confusion_matrix(y_test, y_predict)
print("Confusion matrix:")
print(confusion)

# Scoring
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy:.3f}")

recall = recall_score(y_test, y_predict)
print(f"Recall: {recall:.3f}")

precision = precision_score(y_test, y_predict)
print(f"Precision: {precision:.3f}")

f1 = f1_score(y_test, y_predict)
print(f"F1-score: {f1:.3f}")

# %% [markdown]
# #### 6.3.2. PCA

# %%
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled_pca, y_train_resampled = smote.fit_resample(X_train_pca, y_train)

# hyperparameter tunning
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.1, 1], 'kernel': ['rbf', 'linear']}

svm = SVC(class_weight='balanced', random_state=42)
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train_resampled_pca, y_train_resampled)

best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

svm = SVC(class_weight='balanced', random_state=42, **best_params)
svm.fit(X_train_resampled_pca, y_train_resampled)

y_predict = svm.predict(X_test_pca)

confusion = confusion_matrix(y_test, y_predict)
print("Confusion matrix:")
print(confusion)

# Scoring
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy:.3f}")

recall = recall_score(y_test, y_predict)
print(f"Recall: {recall:.3f}")

precision = precision_score(y_test, y_predict)
print(f"Precision: {precision:.3f}")

f1 = f1_score(y_test, y_predict)
print(f"F1-score: {f1:.3f}")

# %% [markdown]
# #### 6.3.3. FA

# %%
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled_fa, y_train_resampled = smote.fit_resample(X_train_fa, y_train)

# hyperparameter tunning
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.1, 1], 'kernel': ['rbf', 'linear']}

svm = SVC(class_weight='balanced', random_state=42)
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train_resampled_fa, y_train_resampled)

best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

svm = SVC(class_weight='balanced', random_state=42, **best_params)
svm.fit(X_train_resampled_fa, y_train_resampled)

y_predict = svm.predict(X_test_fa)

confusion = confusion_matrix(y_test, y_predict)
print("Confusion matrix:")
print(confusion)

# Scoring
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy:.3f}")

recall = recall_score(y_test, y_predict)
print(f"Recall: {recall:.3f}")

precision = precision_score(y_test, y_predict)
print(f"Precision: {precision:.3f}")

f1 = f1_score(y_test, y_predict)
print(f"F1-score: {f1:.3f}")

# %% [markdown]
# ### 6.4. Discriminant

# %% [markdown]
# #### 6.4.1. Original features

# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from collections import Counter

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Original class distribution: {Counter(y_train)}")
print(f"Resampled class distribution: {Counter(y_train_balanced)}")

class_counts = np.bincount(y_train)
class_weights = class_counts / np.sum(class_counts)

param_grid = [{'solver': ['svd'], 'priors': [class_weights]},
    {'solver': ['lsqr'], 'shrinkage': [None, 'auto', 0.1, 0.2], 'priors': [class_weights]},
    {'solver': ['eigen'], 'shrinkage': [None, 'auto'], 'priors': [class_weights]}]

lda = LinearDiscriminantAnalysis()
grid_search = GridSearchCV(estimator=lda, param_grid=param_grid, cv=5, n_jobs=-1,
                           scoring='f1_weighted', error_score='raise')
grid_search.fit(X_train_balanced, y_train_balanced)

best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

lda_best = LinearDiscriminantAnalysis(**best_params)
lda_best.fit(X_train_balanced, y_train_balanced)

y_predict = lda_best.predict(X_test)

confusion = confusion_matrix(y_test, y_predict)
print("Confusion matrix:")
print(confusion)

# Scoring
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy:.3f}")

recall = recall_score(y_test, y_predict)
print(f"Recall: {recall:.3f}")

precision = precision_score(y_test, y_predict)
print(f"Precision: {precision:.3f}")

f1 = f1_score(y_test, y_predict)
print(f"F1-score: {f1:.3f}")

# %%
pca = PCA(n_components=2)
X_pca_balanced = pca.fit_transform(X_train_balanced)

X_vis = X_pca_balanced
y_vis = y_train_balanced

feature_x = 0
feature_y = 1

x_min, x_max = X_vis[:, feature_x].min() - 1, X_vis[:, feature_x].max() + 1
y_min, y_max = X_vis[:, feature_y].min() - 1, X_vis[:, feature_y].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

grid_df = pd.DataFrame(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]), columns=X_train.columns)

Z = lda_best.predict(grid_df)
Z = Z.reshape(xx.shape)

cmap_light = ListedColormap(["#FF9999", "#9999FF"])
cmap_bold = ["red", "blue"]

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

sns.scatterplot(x=X_pca_balanced[:, feature_x], y=X_pca_balanced[:, feature_y],
                hue=pd.Series(y_vis).map({0: "Normal", 1: "Abnormal"}),
                palette=cmap_bold, edgecolor="k")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("LDA Decision Boundary (PCA Components)")

plt.tight_layout()
plt.show()

# %% [markdown]
# * LDA (Linear Discriminant Analysis) is a linear classification model, so its decision boundary always appears as a straight line.

# %% [markdown]
# #### 6.4.2. PCA

# %%
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_pca, y_train)

print(f"Original class distribution: {Counter(y_train)}")
print(f"Resampled class distribution: {Counter(y_train_balanced)}")

# Define Parameter Grid with Priors Adjusted for Class Weights
class_counts = np.bincount(y_train)
class_weights = class_counts / np.sum(class_counts)   # Normalize to sum=1

param_grid = [
    {'solver': ['svd'], 'priors': [class_weights]},
    {'solver': ['lsqr'], 'shrinkage': [None, 'auto', 0.1, 0.2], 'priors': [class_weights]},
    {'solver': ['eigen'], 'shrinkage': [None, 'auto'], 'priors': [class_weights]}
]

lda = LinearDiscriminantAnalysis()
grid_search = GridSearchCV(estimator=lda, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1_weighted', error_score='raise')
grid_search.fit(X_train_balanced, y_train_balanced)

best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

lda_best = LinearDiscriminantAnalysis(**best_params)
lda_best.fit(X_train_balanced, y_train_balanced)

y_predict = lda_best.predict(X_test_pca)

confusion = confusion_matrix(y_test, y_predict)
print("Confusion matrix:")
print(confusion)

# Scoring
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy:.3f}")

recall = recall_score(y_test, y_predict)
print(f"Recall: {recall:.3f}")

precision = precision_score(y_test, y_predict)
print(f"Precision: {precision:.3f}")

f1 = f1_score(y_test, y_predict)
print(f"F1-score: {f1:.3f}")

# %%
X_train_vis = X_train_balanced[:, :2]
X_test_vis = X_test_pca[:, :2]
y_vis = y_train_balanced.values

# Train LDA with best parameters
lda_vis = LinearDiscriminantAnalysis(**best_params)
lda_vis.fit(X_train_vis, y_vis)

# Create mesh grid
x_min, x_max = X_train_vis[:, 0].min() - 1, X_train_vis[:, 0].max() + 1
y_min, y_max = X_train_vis[:, 1].min() - 1, X_train_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Predict on mesh grid
Z = lda_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cmap_light = ListedColormap(["#FF9999", "#9999FF"])
cmap_bold = ["red", "blue"]

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

sns.scatterplot(
    x=X_train_vis[:, 0],
    y=X_train_vis[:, 1],
    hue=pd.Series(y_vis).map({0: "Normal", 1: "Abnormal"}),
    palette=cmap_bold,
    edgecolor="k")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("LDA Decision Boundary (PCA Components)")

plt.tight_layout()
plt.show()


# %% [markdown]
# #### 6.4.3. FA

# %%
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_fa, y_train)

print(f"Original class distribution: {Counter(y_train)}")
print(f"Resampled class distribution: {Counter(y_train_balanced)}")

# Define Parameter Grid with Priors Adjusted for Class Weights
class_counts = np.bincount(y_train)
class_weights = class_counts / np.sum(class_counts)   # Normalize to sum=1

param_grid = [
    {'solver': ['svd'], 'priors': [class_weights]},
    {'solver': ['lsqr'], 'shrinkage': [None, 'auto', 0.1, 0.2], 'priors': [class_weights]},
    {'solver': ['eigen'], 'shrinkage': [None, 'auto'], 'priors': [class_weights]}
]

lda = LinearDiscriminantAnalysis()
grid_search = GridSearchCV(estimator=lda, param_grid=param_grid, cv=5, n_jobs=-1, scoring='f1_weighted', error_score='raise')
grid_search.fit(X_train_balanced, y_train_balanced)

best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

lda_best = LinearDiscriminantAnalysis(**best_params)
lda_best.fit(X_train_balanced, y_train_balanced)

y_predict = lda_best.predict(X_test_fa)

confusion = confusion_matrix(y_test, y_predict)
print("Confusion matrix:")
print(confusion)

# Scoring
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy:.3f}")

recall = recall_score(y_test, y_predict)
print(f"Recall: {recall:.3f}")

precision = precision_score(y_test, y_predict)
print(f"Precision: {precision:.3f}")

f1 = f1_score(y_test, y_predict)
print(f"F1-score: {f1:.3f}")

# %%
# Use first two factor analysis components for visualization
X_train_vis = X_train_balanced[:, :2]
X_test_vis = X_test_fa[:, :2]
y_vis = y_train_balanced.values

# Train LDA with best parameters
lda_vis = LinearDiscriminantAnalysis(**best_params)
lda_vis.fit(X_train_vis, y_vis)

# Create mesh grid
x_min, x_max = X_train_vis[:, 0].min() - 1, X_train_vis[:, 0].max() + 1
y_min, y_max = X_train_vis[:, 1].min() - 1, X_train_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Predict on mesh grid
Z = lda_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cmap_light = ListedColormap(["#FF9999", "#9999FF"])
cmap_bold = ["red", "blue"]

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

sns.scatterplot(
    x=X_train_vis[:, 0],
    y=X_train_vis[:, 1],
    hue=pd.Series(y_vis).map({0: "Normal", 1: "Abnormal"}),
    palette=cmap_bold,
    edgecolor="k")

plt.xlabel("Factor Component 1")
plt.ylabel("Factor Component 2")
plt.title("LDA Decision Boundary (FA Components)")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6.5. Conclusion

# %%
data = {
    'Model': ['KNN', 'Random Forest', 'SVM', 'LDA'],
    'Original Accuracy': [0.839, 0.790, 0.823, 0.855],
    'Original Recall': [0.818, 0.909, 0.818, 0.909],
    'Original Precision': [0.949, 0.816, 0.923, 0.889],
    'Original F1-score': [0.878, 0.860, 0.867, 0.899],
    'PCA Accuracy': [0.823, 0.790, 0.839, 0.839],
    'PCA Recall': [0.818, 0.864, 0.818, 0.818],
    'PCA Precision': [0.923, 0.844, 0.947, 0.947],
    'PCA F1-score': [0.867, 0.854, 0.878, 0.878],
    'FA Accuracy': [0.726, 0.677, 0.661, 0.758],
    'FA Recall': [0.727, 0.795, 0.727, 0.909],
    'FA Precision': [0.865, 0.761, 0.780, 0.784],
    'FA F1-score': [0.790, 0.778, 0.753, 0.842]
}

metrics_df = pd.DataFrame(data)

# Highlight the best values in each metric column
def highlight_best(val, col):
    # Return yellow for the highest value in each column
    return 'background-color: yellow' if val == metrics_df[col].max() else ''

# Apply the highlighting function to the columns of interest
styled_df = metrics_df.style.applymap(lambda val: highlight_best(val, 'Original Accuracy'), subset=['Original Accuracy'])
styled_df = styled_df.applymap(lambda val: highlight_best(val, 'PCA Accuracy'), subset=['PCA Accuracy'])
styled_df = styled_df.applymap(lambda val: highlight_best(val, 'FA Accuracy'), subset=['FA Accuracy'])
styled_df = styled_df.applymap(lambda val: highlight_best(val, 'Original Recall'), subset=['Original Recall'])
styled_df = styled_df.applymap(lambda val: highlight_best(val, 'PCA Recall'), subset=['PCA Recall'])
styled_df = styled_df.applymap(lambda val: highlight_best(val, 'FA Recall'), subset=['FA Recall'])
styled_df = styled_df.applymap(lambda val: highlight_best(val, 'Original Precision'), subset=['Original Precision'])
styled_df = styled_df.applymap(lambda val: highlight_best(val, 'PCA Precision'), subset=['PCA Precision'])
styled_df = styled_df.applymap(lambda val: highlight_best(val, 'FA Precision'), subset=['FA Precision'])
styled_df = styled_df.applymap(lambda val: highlight_best(val, 'Original F1-score'), subset=['Original F1-score'])
styled_df = styled_df.applymap(lambda val: highlight_best(val, 'PCA F1-score'), subset=['PCA F1-score'])
styled_df = styled_df.applymap(lambda val: highlight_best(val, 'FA F1-score'), subset=['FA F1-score'])

# Save the styled DataFrame as an image using matplotlib
fig, ax = plt.subplots(figsize=(12, 6))  # Adjust the figure size as necessary
ax.axis('off')  # Hide the axes

# Create a table and apply the style
table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, loc='center', cellLoc='center')

# Apply custom styles to the table
for i, col in enumerate(metrics_df.columns):
    for j, cell in enumerate(table.get_celld().values()):
        if cell.get_text().get_text() == str(metrics_df[col].max()):
            cell.set_facecolor('yellow')  # Highlight the best values in yellow

# Adjust the layout
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(metrics_df.columns))))

# Save the figure as an image
plt.show()


# %%
results = pd.DataFrame(data)

# Separate Line Plots for each method
methods = ['Original', 'PCA', 'FA']
metrics = ['Accuracy', 'Recall', 'Precision', 'F1-score']

for method in methods:
    plt.figure(figsize=(8, 6))

    for metric in metrics:
        plt.plot(results['Model'], results[f'{method} {metric}'], marker='o', label=metric)

    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title(f'{method} Model Performance Comparison')
    plt.legend()
    plt.grid(True)

    plt.show()


# %%
# KNN and LDA

df = pd.DataFrame(data)

knn_lda_data = df[df['Model'].isin(['KNN', 'LDA'])]

results = pd.DataFrame(knn_lda_data)

# Separate Line Plots for each method
methods = ['Original', 'PCA', 'FA']
metrics = ['Accuracy', 'Recall', 'Precision', 'F1-score']

for method in methods:
    plt.figure(figsize=(8, 6))

    for metric in metrics:
        plt.plot(results['Model'], results[f'{method} {metric}'], marker='o', label=metric)

    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title(f'{method} Model Performance Comparison')
    plt.legend()
    plt.grid(True)

    plt.show()


# %%
results = pd.DataFrame(data)

# Creating the subplots for each metric and clustering method
methods = ['Original', 'PCA', 'FA']
metrics = ['Accuracy', 'Recall', 'Precision', 'F1-score']

for method in methods:
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    for i, metric in enumerate(metrics):
        axes[i].bar(results['Model'], results[f'{method} {metric}'], color=['b', 'g', 'r', 'purple'][i])
        axes[i].set_title(f'{metric} - {method}')
        axes[i].set_ylabel('Score')
        for j, value in enumerate(results[f'{method} {metric}']):
            axes[i].text(j, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
        axes[i].set_ylim(0, max(results[f'{method} {metric}']) + 0.1)

    plt.tight_layout()
    plt.show()


# %%
# KNN and LDA

methods = ['Original', 'PCA', 'FA']
metrics = ['Accuracy', 'Recall', 'Precision', 'F1-score']

results = pd.DataFrame(knn_lda_data)

for method in methods:
    fig, axes = plt.subplots(1, 4, figsize=(10, 5))

    for i, metric in enumerate(metrics):
        axes[i].bar(results['Model'], results[f'{method} {metric}'], color=['b', 'g', 'r', 'purple'][i])
        axes[i].set_title(f'{metric} - {method}')
        axes[i].set_ylabel('Score')
        for j, value in enumerate(results[f'{method} {metric}']):
            axes[i].text(j, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
        axes[i].set_ylim(0, max(results[f'{method} {metric}']) + 0.1)

    plt.tight_layout()
    plt.show()


# %%
models = ['KNN', 'LDA']

x = np.arange(len(methods))
width = 0.3

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]

    for j, model in enumerate(models):
        scores = [df[(df['Model'] == model)][f'{method} {metric}'].values[0] for method in methods]
        ax.bar(x + j * width, scores, width, label=model)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Score")
    ax.set_title(f"{metric} Comparison")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()


# %% [markdown]
# # 7. Unsupervised Clustering
# 

# %%
data = open('/content/drive/MyDrive/Colab Notebooks/ML2/Data/data.txt').readlines()
data = [i.split() for i in data]

column_names = [
    "pelvic_incidence",
    "pelvic_tilt",
    "lumbar_lordosis_angle",
    "sacral_slope",
    "pelvic_radius",
    "grade_of_spondylolisthesis",
    "class"
]

data = pd.DataFrame(columns = column_names, data = data)
data.head()

X = data[numeric_features]
y = data['class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %% [markdown]
# ## 7.1 GMM

# %%
gmm2 = GaussianMixture(n_components=2, random_state=2025)
clusters_gmm2 = gmm2.fit_predict(X_scaled)

data['GMM_cluster2'] = clusters_gmm2

print("Cluster distribution:", np.bincount(clusters_gmm2))

# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
custom_palette = ["#377eb8","#e41a1c"]
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters_gmm2, palette=custom_palette)
plt.title("GMM Clustering After PCA - 2 Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# %%
gmm3 = GaussianMixture(n_components=3, random_state=2025)
clusters_gmm3 = gmm3.fit_predict(X_scaled)

data['GMM_cluster_3'] = clusters_gmm3

print("Cluster distribution:", np.bincount(clusters_gmm3))

# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot GMM clusters
plt.figure(figsize=(8,6))
custom_palette = ["#377eb8","#e41a1c","#E69F00"]
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters_gmm3, palette=custom_palette)
plt.title("GMM Clustering After PCA - 3 Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# %%
n_components = range(1, 11)
bics = []

for n in n_components:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(X_scaled)
    bics.append(gmm.bic(X_scaled))

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(n_components, bics, marker='o', linestyle='-')
ax.set_xlabel('Number of Components')
ax.set_ylabel('BIC')
ax.set_title('BIC for Optimal Number of Components in GMM')


plt.show()

# %% [markdown]
# GMM has a decent level of performance, with three clusters, which suggests that the algorithm might be identifying the three orthopaedic classes detailed in the dataset description: normal, disk hernia, or spondylolisthesis

# %%
# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 10))

# Plot 1: GMM Clustering (2 Clusters)
custom_palette_2 = ["#377eb8","#e41a1c"]
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters_gmm2, palette=custom_palette_2, ax=axes[0])
axes[0].set_title("GMM Clustering After PCA - 2 Clusters")
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")

# Plot 2: GMM Clustering (3 Clusters)
custom_palette_3 = ["#377eb8","#e41a1c","#E69F00"]
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters_gmm3, palette=custom_palette_3, ax=axes[1])
axes[1].set_title("GMM Clustering After PCA - 3 Clusters")
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")

# Plot 3: True Class Labels
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=data["class"], palette="Set1", ax=axes[2])
axes[2].set_title("True Class Labels in PCA Space")
axes[2].set_xlabel("Principal Component 1")
axes[2].set_ylabel("Principal Component 2")

# Adjust layout to avoid overlap
plt.tight_layout()

# Save the combined figure (optional)
fig.savefig("GMM.jpeg", format='jpeg', dpi=300)

# Show the combined plot
plt.show()


# %% [markdown]
# ### 7.2. KMeans

# %%
kmeans2 = KMeans(n_clusters=2, random_state=42, n_init=10)  # n_init=10 for better stability
clusters_kmeans2 = kmeans2.fit_predict(X_scaled)

# Add the clusters to the DataFrame
data['KMeans_cluster2'] = clusters_kmeans2

# %%
# Plot K-Means clusters in PCA space
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters_kmeans2, palette="Set1")
plt.title("K-Means Clustering After PCA - 2 Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# %%
kmeans_3 = KMeans(n_clusters=3, random_state=42, n_init=10)  # n_init=10 for better stability
clusters_kmeans_3 = kmeans_3.fit_predict(X_scaled)

# Add the clusters to the DataFrame
data['KMeans_cluster_3'] = clusters_kmeans_3

# %%
# Plot K-Means clusters in PCA space
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters_kmeans_3, palette=custom_palette)
plt.title("K-Means Clustering After PCA - 3 Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# %%
inertia = []
K_range = range(1, 11)  # Try k from 1 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(K_range, inertia, marker='o', linestyle='-')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method for Optimal k')

plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 10))

# K-Means Clustering - 2 Clusters
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters_kmeans2, palette="Set1", ax=axes[0])
axes[0].set_title("K-Means Clustering After PCA - 2 Clusters")
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")

# K-Means Clustering - 3 Clusters
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters_kmeans_3, palette=custom_palette_3, ax=axes[1])
axes[1].set_title("K-Means Clustering After PCA - 3 Clusters")
axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")

# True Class Labels
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=data["class"], palette="Set1", ax=axes[2])
axes[2].set_title("True Class Labels in PCA Space")
axes[2].set_xlabel("Principal Component 1")
axes[2].set_ylabel("Principal Component 2")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7.3 Clustering Metrics

# %% [markdown]
# Silhouette Score

# %%
from sklearn.metrics import silhouette_score

silhouette_kmeans = silhouette_score(X_scaled, clusters_kmeans2)
silhouette_kmeans3 = silhouette_score(X_scaled, clusters_kmeans_3)

silhouette_gmm = silhouette_score(X_scaled, clusters_gmm2)
silhouette_gmm3 = silhouette_score(X_scaled, clusters_gmm3)

print("Silhouette Score - KMeans (2):", silhouette_kmeans)
print("Silhouette Score - KMeans (3):", silhouette_kmeans3)

print("Silhouette Score - GMM (2):", silhouette_gmm)
print("Silhouette Score - GMM (3):", silhouette_gmm3)

# %% [markdown]
# David-Bouldin Index (DBI)

# %%
from sklearn.metrics import davies_bouldin_score

dbi_kmeans = davies_bouldin_score(X_scaled, clusters_kmeans2)
dbi_kmeans3 = davies_bouldin_score(X_scaled, clusters_kmeans_3)

dbi_gmm = davies_bouldin_score(X_scaled, clusters_gmm2)
dbi_gmm3 = davies_bouldin_score(X_scaled, clusters_gmm3)

print("Davies-Bouldin Index - KMeans (2):", dbi_kmeans)
print("Davies-Bouldin Index - KMeans (3):", dbi_kmeans3)

print("Davies-Bouldin Index - GMM (2):", dbi_gmm)
print("Davies-Bouldin Index - GMM (3):", dbi_gmm3)

# %% [markdown]
# Calinski-Harabasz Index (CHI)

# %%
from sklearn.metrics import calinski_harabasz_score

chi_kmeans = calinski_harabasz_score(X_scaled, clusters_kmeans2)
chi_kmeans3 = calinski_harabasz_score(X_scaled, clusters_kmeans_3)

chi_gmm = calinski_harabasz_score(X_scaled, clusters_gmm2)
chi_gmm3 = calinski_harabasz_score(X_scaled, clusters_gmm3)

print("Calinski-Harabasz Index - KMeans (2):", chi_kmeans)
print("Calinski-Harabasz Index - KMeans (3):", chi_kmeans3)

print("Calinski-Harabasz Index - GMM (2):", chi_gmm)
print("Calinski-Harabasz Index - GMM (3):", chi_gmm3)

# %% [markdown]
# Log-likelihood of GMM models

# %%
# 2 clusters
log_likelihood_gmm_2 = gmm2.score(X_scaled)
total_log_likelihood_gmm_2 = gmm2.score_samples(X_scaled).sum()

# 3 clusters
log_likelihood_gmm_3 = gmm3.score(X_scaled)
total_log_likelihood_gmm_3 = gmm3.score_samples(X_scaled).sum()

print("Log-Likelihood - GMM (2 clusters):", total_log_likelihood_gmm_2)
print("Log-Likelihood - GMM (3 clusters):", total_log_likelihood_gmm_3)

# %%
probabilities_2 = gmm2.predict_proba(X_scaled)
probabilities_3 = gmm3.predict_proba(X_scaled)

entropy_2 = -np.sum(probabilities_2 * np.log(probabilities_2 + 1e-10), axis=1)
entropy_3 = -np.sum(probabilities_3 * np.log(probabilities_3 + 1e-10), axis=1)

avg_entropy_2 = np.mean(entropy_2)
avg_entropy_3 = np.mean(entropy_3)

print("Average Entropy - GMM (2 clusters):", avg_entropy_2)
print("Average Entropy - GMM (3 clusters):", avg_entropy_3)

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 6))

titles = ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"]

data = [
    [0.3629, 0.2951, 0.3656, 0.2264],  # Silhouette Score
    [1.1309, 1.2986, 1.1873, 1.4697],  # Davies-Bouldin Index
    [189.3363, 153.5933, 143.4943, 106.7340],  # Calinski-Harabasz Index
]

bar_width = 0.8
x_labels = ["KM (2)", "KM(3)", "GMM (2)", "GMM (3)"]
x = np.arange(len(x_labels))

colors = ["blue", "lightblue", "green", "lightgreen"]

for i, ax in enumerate(axes):
    bars = ax.bar(x, data[i], width=bar_width, color=colors)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}",
                ha='center', va='bottom', fontsize=12)

    ax.set_title(titles[i])
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_ylabel("Score")

plt.tight_layout()
plt.show()


