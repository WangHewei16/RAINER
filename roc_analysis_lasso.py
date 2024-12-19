import pandas as pd
import numpy as np

from warnings import simplefilter
simplefilter(action='ignore')
import category_encoders

from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import copy

weather_data = pd.read_csv("../weatherAUS.csv")
weather_data.isna().sum().sum()
weather_data.isnull().sum().sort_values(ascending=False)
(weather_data.isnull().sum() / weather_data.isnull().count()).sort_values(ascending=False)

weather_data.drop(["Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"], inplace=True, axis=1)

weather_data.RainTomorrow.value_counts()

# search the numerical features
numerical_features = weather_data.columns[weather_data.dtypes != object]

# search the categorical features
categorical_features = weather_data.columns[weather_data.dtypes == object]


# Fill the numerical features NaN values using average values
weather_data["Temp9am"]= weather_data["Temp9am"].fillna(weather_data["Temp9am"].mean())
weather_data["Temp3pm"]= weather_data["Temp3pm"].fillna(weather_data["Temp3pm"].mean())
weather_data["WindSpeed9am"]= weather_data["WindSpeed9am"].fillna(weather_data["WindSpeed9am"].mean())
weather_data["WindSpeed3pm"]= weather_data["WindSpeed3pm"].fillna(weather_data["WindSpeed3pm"].mean())
weather_data["MinTemp"]= weather_data["MinTemp"].fillna(weather_data["MinTemp"].mean())
weather_data["MaxTemp"]= weather_data["MaxTemp"].fillna(weather_data["MaxTemp"].mean())
weather_data["WindGustSpeed"]= weather_data["WindGustSpeed"].fillna(weather_data["WindGustSpeed"].mean())
weather_data["Rainfall"]= weather_data["Rainfall"].fillna(weather_data["Rainfall"].mean())
weather_data["Pressure9am"]= weather_data["Pressure9am"].fillna(weather_data["Pressure9am"].mean())
weather_data["Pressure3pm"]= weather_data["Pressure3pm"].fillna(weather_data["Pressure3pm"].mean())
weather_data["Humidity9am"]= weather_data["Humidity9am"].fillna(weather_data["Humidity9am"].mean())
weather_data["Humidity3pm"]= weather_data["Humidity3pm"].fillna(weather_data["Humidity3pm"].mean())


# Fill the categorical features NaN value using the previous value
weather_data['RainToday'] = weather_data['RainToday'].fillna(weather_data['RainToday'].mode()[0])
weather_data['RainTomorrow']= weather_data['RainTomorrow'].fillna(weather_data['RainTomorrow'].mode()[0])
weather_data['WindGustDir'] = weather_data['WindGustDir'].fillna(weather_data['WindGustDir'].mode()[0])
weather_data['WindDir3pm'] = weather_data['WindDir3pm'].fillna(weather_data['WindDir3pm'].mode()[0])
weather_data['WindDir9am'] = weather_data['WindDir9am'].fillna(weather_data['WindDir9am'].mode()[0])

weather_data.isnull().sum().sort_values()

def find_max(df, feature, max_current):
    return np.where(df[feature] > max_current, max_current, df[feature])

weather_data['Rainfall'] = find_max(weather_data, 'Rainfall', 3.2)
weather_data['WindSpeed9am'] = find_max(weather_data, 'WindSpeed9am', 55)
weather_data['WindSpeed3pm'] = find_max(weather_data, 'WindSpeed3pm', 57)

weather_data['Date']=pd.to_datetime(weather_data['Date'])
weather_data['Month'] = pd.DatetimeIndex(weather_data['Date']).month

weather_data.info()

binary_encoder = category_encoders.BinaryEncoder(cols=['RainToday'])
weather_data = binary_encoder.fit_transform(weather_data)

label_encoder = LabelEncoder()
weather_data['Location'] = label_encoder.fit_transform(weather_data['Location'].values)
weather_data['RainTomorrow'] = label_encoder.fit_transform(weather_data['RainTomorrow'].values)
weather_data['WindDir3pm'] = label_encoder.fit_transform(weather_data['WindDir3pm'].values)
weather_data['WindGustDir'] = label_encoder.fit_transform(weather_data['WindGustDir'].values)
weather_data['WindDir9am'] = label_encoder.fit_transform(weather_data['WindDir9am'].values)

# Copy the original features
original_data = copy.deepcopy(weather_data)


max_temperature_difference = pd.qcut(weather_data.MaxTemp - weather_data.MinTemp, q = 4, labels = [0,1,2,3])
weather_data.insert(2,"MaxDifferenceTemp",max_temperature_difference)
weather_data['MaxDifferenceTemp'] = weather_data['MaxDifferenceTemp'].astype('float64')
weather_data = weather_data.drop(['MinTemp', 'MaxTemp'], axis=1)


max_humidity_difference = pd.qcut(weather_data.Humidity9am - weather_data.Humidity3pm, q = 4, labels = [0,1,2,3])
weather_data.insert(3,"MaxDifferenceHumidity",max_humidity_difference)
weather_data['MaxDifferenceHumidity'] = weather_data['MaxDifferenceHumidity'].astype('float64')
weather_data = weather_data.drop(['Humidity9am', 'Humidity3pm'],axis=1)

weather_data = weather_data.drop(columns=['Date'])
# divide two classes, then get upsample class and concat with normal class
normal_weather_data = weather_data[weather_data.RainTomorrow==0]
abnormal_weather_data = weather_data[weather_data.RainTomorrow==1]
upsampled_data = resample(abnormal_weather_data, replace=True, random_state=123, n_samples=len(normal_weather_data))
weather_data = pd.concat([normal_weather_data, upsampled_data])
# use MinMaxScaler to execute scaling operation
cols = weather_data.columns
min_max_scaler = MinMaxScaler()

weather_data = min_max_scaler.fit_transform(weather_data)
weather_data = pd.DataFrame(weather_data, columns=[cols])
weather_data.describe()

min_max_scaler = MinMaxScaler()
scaled_data = pd.DataFrame(min_max_scaler.fit_transform(weather_data), columns=weather_data.columns, index=weather_data.index)
x = scaled_data.drop(['RainTomorrow'], axis = 1)
y = scaled_data['RainTomorrow']




# 3. LASSO Model Implementation

# Import necessary libraries for LASSO
import pandas as pd
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})
# Initialize lists to record metrics
lasso_accuracy_record = []
lasso_precision_record = []
lasso_recall_record = []
lasso_auc_record = []
lasso_f1_record = []

### 3.1 Split data (8:1:1) and Train LASSO Model

# Split up 0.8 for train and 0.2 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

# Initialize Logistic Regression with L1 penalty (LASSO)
lasso_clf = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)

# Fit the model
lasso_clf.fit(x_train, y_train)

# Predict on validation set
y_pred = lasso_clf.predict(x_validation)
y_pred_proba = lasso_clf.predict_proba(x_validation)[:, 1]

# Calculate metrics
acc = metrics.accuracy_score(y_validation, y_pred)
precision = precision_score(y_validation, y_pred)
recall = recall_score(y_validation, y_pred)
auc_score = metrics.roc_auc_score(y_validation, y_pred_proba)
f1 = f1_score(y_validation, y_pred)

# Record metrics
lasso_accuracy_record.append(acc)
lasso_precision_record.append(precision)
lasso_recall_record.append(recall)
lasso_auc_record.append(auc_score)
lasso_f1_record.append(f1)

# Compute ROC curve
fpr, tpr, _ = metrics.roc_curve(y_validation, y_pred_proba)

### 3.2 Split data (6:2:2) and Train LASSO Model

# Split up 0.6 for train and 0.4 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

# Initialize Logistic Regression with L1 penalty (LASSO)
lasso_clf = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)

# Fit the model
lasso_clf.fit(x_train, y_train)

# Predict on validation set
y_pred1 = lasso_clf.predict(x_validation)
y_pred1_proba = lasso_clf.predict_proba(x_validation)[:, 1]

# Calculate metrics
acc1 = metrics.accuracy_score(y_validation, y_pred1)
precision1 = precision_score(y_validation, y_pred1)
recall1 = recall_score(y_validation, y_pred1)
auc_score1 = metrics.roc_auc_score(y_validation, y_pred1_proba)
f1_1 = f1_score(y_validation, y_pred1)

# Record metrics
lasso_accuracy_record.append(acc1)
lasso_precision_record.append(precision1)
lasso_recall_record.append(recall1)
lasso_auc_record.append(auc_score1)
lasso_f1_record.append(f1_1)

# Compute ROC curve
fpr1, tpr1, _ = metrics.roc_curve(y_validation, y_pred1_proba)

### 3.3 Split data (4:3:3) and Train LASSO Model

# Split up 0.4 for train and 0.6 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=42)
x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

# Initialize Logistic Regression with L1 penalty (LASSO)
lasso_clf = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)

# Fit the model
lasso_clf.fit(x_train, y_train)

# Predict on validation set
y_pred2 = lasso_clf.predict(x_validation)
y_pred2_proba = lasso_clf.predict_proba(x_validation)[:, 1]

# Calculate metrics
acc2 = metrics.accuracy_score(y_validation, y_pred2)
precision2 = precision_score(y_validation, y_pred2)
recall2 = recall_score(y_validation, y_pred2)
auc_score2 = metrics.roc_auc_score(y_validation, y_pred2_proba)
f1_2 = f1_score(y_validation, y_pred2)

# Record metrics
lasso_accuracy_record.append(acc2)
lasso_precision_record.append(precision2)
lasso_recall_record.append(recall2)
lasso_auc_record.append(auc_score2)
lasso_f1_record.append(f1_2)

# Compute ROC curve
fpr2, tpr2, _ = metrics.roc_curve(y_validation, y_pred2_proba)

### 3.4 Create Metrics DataFrame

# Create DataFrame for metrics
data = {
    'Accuracy': [f"{acc:.3f}", f"{acc1:.3f}", f"{acc2:.3f}"],
    'Precision': [f"{precision:.3f}", f"{precision1:.3f}", f"{precision2:.3f}"],
    'Recall': [f"{recall:.3f}", f"{recall1:.3f}", f"{recall2:.3f}"],
    'ROC-score': [f"{auc_score:.3f}", f"{auc_score1:.3f}", f"{auc_score2:.3f}"],
    'F1-score': [f"{f1:.3f}", f"{f1_1:.3f}", f"{f1_2:.3f}"],
}
labels = ['8:1:1', '6:2:2', '4:3:3']
df_lasso = pd.DataFrame(data, index=labels)

## 3.5 Plot ROC Curves

import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')

plt.plot(fpr2, tpr2, label="4:3:3", c='blue', lw=1.5, linestyle='--')
plt.plot(fpr, tpr, label="8:1:1", c='red', lw=1.5, linestyle='-')
plt.plot(fpr1, tpr1, label="6:2:2", c='green', lw=1.5, linestyle='--')

plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7, lw=1.5)

plt.grid(which='major', linestyle='--', linewidth=1.2, alpha=0.8)
plt.grid(which='minor', linestyle='--', linewidth=1.0, alpha=0.6)

ax.set_xlabel('False Positive Rate', fontsize=16)
ax.set_ylabel('True Positive Rate', fontsize=16)

ax.text(0.5, 0.05, "LASSO", fontsize=24, fontweight='bold', ha='center', transform=ax.transAxes)

plt.legend(loc='lower right', fontsize=10, frameon=True, shadow=True, title_fontsize=12)

axins = inset_axes(ax, width="33%", height="30%", loc='lower left',
                   bbox_to_anchor=(0.5, 0.5, 1, 1), bbox_transform=ax.transAxes)

for f, t, c, ls in zip([fpr, fpr1, fpr2],
                       [tpr, tpr1, tpr2],
                       ['red', 'green', 'blue'],
                       ['-', '--', '--']):
    axins.plot(f, t, c=c, lw=2, linestyle=ls)

axins.set_xlim(0.1, 0.2)
axins.set_ylim(0.5, 0.6)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='black', lw=1)

ax.set_facecolor('white')

plt.tight_layout()

save_folder = "figure"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

save_path = os.path.join(save_folder, "ROC_LASSO_with_best_performance_highlighted.pdf")
plt.savefig(save_path, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())

plt.show()

df_lasso

