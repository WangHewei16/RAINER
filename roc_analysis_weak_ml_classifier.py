# data analysis related library
import os

import pandas as pd
import numpy as np

# some visualization related library
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LogNorm
import seaborn as sns

# category_encoders and warning related library
from warnings import simplefilter
simplefilter(action='ignore')
import category_encoders

# sklearn related ML library
from sklearn import metrics
from sklearn.metrics import roc_curve, RocCurveDisplay
from sklearn.metrics import mean_absolute_error
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier

# other useful libray
import missingno as msno
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pylab import subplots_adjust
import copy
import torch
import torch.nn as nn
import torch.optim as optim

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

# 3.1 DT

DT_accuracy_record = list()

# Split up 0.8 for train and 0.2 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_test,x_validation,y_test,y_validation = train_test_split(x_test, y_test, test_size=0.5)
# Gini MaxDepth=5
classifier = DecisionTreeClassifier(criterion='gini',max_depth=5)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_validation)
y_pred_gbc = classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
DT_accuracy_record.append(acc)
fpr, tpr, _= roc_curve(y_validation, y_pred_gbc)
auc = metrics.roc_auc_score(y_validation, y_pred)
a = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred))
b = "{:.3}".format(metrics.precision_score(y_validation, y_pred))
c = "{:.3}".format(metrics.recall_score(y_validation, y_pred))
d = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred))
e = "{:.3}".format(metrics.f1_score(y_validation, y_pred))
# Gini MaxDepth=10
classifier = DecisionTreeClassifier(criterion='gini',max_depth=10)
classifier.fit(x_train, y_train)
y_pred1 = classifier.predict(x_validation)
y_pred1_gbc = classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
DT_accuracy_record.append(acc)
fpr1, tpr1, _= roc_curve(y_validation, y_pred1_gbc)
auc1 = metrics.roc_auc_score(y_validation, y_pred1)
a1 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred1))
b1 = "{:.3}".format(metrics.precision_score(y_validation, y_pred1))
c1 = "{:.3}".format(metrics.recall_score(y_validation, y_pred1))
d1 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred1))
e1 = "{:.3}".format(metrics.f1_score(y_validation, y_pred1))
# Gini MaxDepth=20
classifier = DecisionTreeClassifier(criterion='gini',max_depth=20)
classifier.fit(x_train, y_train)
y_pred2 = classifier.predict(x_validation)
y_pred2_gbc = classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
DT_accuracy_record.append(acc)
fpr2, tpr2, _= roc_curve(y_validation, y_pred2_gbc)
auc2 = metrics.roc_auc_score(y_validation, y_pred2)
a2 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred2))
b2 = "{:.3}".format(metrics.precision_score(y_validation, y_pred2))
c2 = "{:.3}".format(metrics.recall_score(y_validation, y_pred2))
d2 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred2))
e2 = "{:.3}".format(metrics.f1_score(y_validation, y_pred2))
# Entropy MaxDepth=5
classifier = DecisionTreeClassifier(criterion='entropy',max_depth=5)
classifier.fit(x_train, y_train)
y_pred3 = classifier.predict(x_validation)
y_pred3_gbc = classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
DT_accuracy_record.append(acc)
fpr3, tpr3, _= roc_curve(y_validation, y_pred3_gbc)
auc3 = metrics.roc_auc_score(y_validation, y_pred3)
a3 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred3))
b3 = "{:.3}".format(metrics.precision_score(y_validation, y_pred3))
c3 = "{:.3}".format(metrics.recall_score(y_validation, y_pred3))
d3 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred3))
e3 = "{:.3}".format(metrics.f1_score(y_validation, y_pred3))
# Entropy MaxDepth=10
classifier = DecisionTreeClassifier(criterion='entropy',max_depth=10)
classifier.fit(x_train, y_train)
y_pred4 = classifier.predict(x_validation)
y_pred4_gbc = classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
DT_accuracy_record.append(acc)
fpr4, tpr4, _= roc_curve(y_validation, y_pred4_gbc)
auc4 = metrics.roc_auc_score(y_validation, y_pred4)
a4 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred4))
b4 = "{:.3}".format(metrics.precision_score(y_validation, y_pred4))
c4 = "{:.3}".format(metrics.recall_score(y_validation, y_pred4))
d4 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred4))
e4 = "{:.3}".format(metrics.f1_score(y_validation, y_pred4))
# Entropy MaxDepth=20
classifier = DecisionTreeClassifier(criterion='entropy',max_depth=20)
classifier.fit(x_train, y_train)
y_pred5 = classifier.predict(x_validation)
y_pred5_gbc = classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
DT_accuracy_record.append(acc)
fpr5, tpr5, _= roc_curve(y_validation, y_pred5_gbc)
auc5 = metrics.roc_auc_score(y_validation, y_pred5)
a5 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred5))
b5 = "{:.3}".format(metrics.precision_score(y_validation, y_pred5))
c5 = "{:.3}".format(metrics.recall_score(y_validation, y_pred5))
d5 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred5))
e5 = "{:.3}".format(metrics.f1_score(y_validation, y_pred5))


# Split up 0.6 for train and 0.4 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
x_test,x_validation,y_test,y_validation = train_test_split(x_test, y_test, test_size=0.5)
# Gini MaxDepth=5
classifier = DecisionTreeClassifier(criterion='gini',max_depth=5)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_validation)
y_pred_gbc = classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
DT_accuracy_record.append(acc)
fpr6, tpr6, _= roc_curve(y_validation, y_pred_gbc)
auc6 = metrics.roc_auc_score(y_validation, y_pred)
a6 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred))
b6 = "{:.3}".format(metrics.precision_score(y_validation, y_pred))
c6 = "{:.3}".format(metrics.recall_score(y_validation, y_pred))
d6 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred))
e6 = "{:.3}".format(metrics.f1_score(y_validation, y_pred))
# Gini MaxDepth=10
classifier = DecisionTreeClassifier(criterion='gini',max_depth=10)
classifier.fit(x_train, y_train)
y_pred1 = classifier.predict(x_validation)
y_pred1_gbc = classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
DT_accuracy_record.append(acc)
fpr7, tpr7, _= roc_curve(y_validation,  y_pred1_gbc)
auc7 = metrics.roc_auc_score(y_validation, y_pred1)
a7 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred1))
b7 = "{:.3}".format(metrics.precision_score(y_validation, y_pred1))
c7 = "{:.3}".format(metrics.recall_score(y_validation, y_pred1))
d7 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred1))
e7 = "{:.3}".format(metrics.f1_score(y_validation, y_pred1))
# Gini MaxDepth=20
classifier = DecisionTreeClassifier(criterion='gini',max_depth=20)
classifier.fit(x_train, y_train)
y_pred2 = classifier.predict(x_validation)
y_pred2_gbc = classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
DT_accuracy_record.append(acc)
fpr8, tpr8, _= roc_curve(y_validation, y_pred2_gbc)
auc8 = metrics.roc_auc_score(y_validation, y_pred2)
a8 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred2))
b8 = "{:.3}".format(metrics.precision_score(y_validation, y_pred2))
c8 = "{:.3}".format(metrics.recall_score(y_validation, y_pred2))
d8 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred2))
e8 = "{:.3}".format(metrics.f1_score(y_validation, y_pred2))
# Entropy MaxDepth=5
classifier = DecisionTreeClassifier(criterion='entropy',max_depth=5)
classifier.fit(x_train, y_train)
y_pred3 = classifier.predict(x_validation)
y_pred3_gbc = classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
DT_accuracy_record.append(acc)
fpr9, tpr9, _= roc_curve(y_validation, y_pred3_gbc)
auc9 = metrics.roc_auc_score(y_validation, y_pred3)
a9 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred3))
b9 = "{:.3}".format(metrics.precision_score(y_validation, y_pred3))
c9 = "{:.3}".format(metrics.recall_score(y_validation, y_pred3))
d9 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred3))
e9 = "{:.3}".format(metrics.f1_score(y_validation, y_pred3))
# Entropy MaxDepth=10
classifier = DecisionTreeClassifier(criterion='entropy',max_depth=10)
classifier.fit(x_train, y_train)
y_pred4 = classifier.predict(x_validation)
y_pred4_gbc = classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
DT_accuracy_record.append(acc)
fpr10, tpr10, _= roc_curve(y_validation, y_pred4_gbc)
auc10 = metrics.roc_auc_score(y_validation, y_pred4)
a10 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred4))
b10 = "{:.3}".format(metrics.precision_score(y_validation, y_pred4))
c10 = "{:.3}".format(metrics.recall_score(y_validation, y_pred4))
d10 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred4))
e10 = "{:.3}".format(metrics.f1_score(y_validation, y_pred4))
# Entropy MaxDepth=20
classifier = DecisionTreeClassifier(criterion='entropy',max_depth=20)
classifier.fit(x_train, y_train)
y_pred5 = classifier.predict(x_validation)
y_pred5_gbc = classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
DT_accuracy_record.append(acc)
fpr11, tpr11, _= roc_curve(y_validation, y_pred5_gbc)
auc11 = metrics.roc_auc_score(y_validation, y_pred5)
a11 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred5))
b11 = "{:.3}".format(metrics.precision_score(y_validation, y_pred5))
c11 = "{:.3}".format(metrics.recall_score(y_validation, y_pred5))
d11 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred5))
e11 = "{:.3}".format(metrics.f1_score(y_validation, y_pred5))

# df table
data = {'Accuracy': [a,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11],
        'Precision': [b,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11],
        'Recall': [c,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11],
        'ROC-score': [d,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11],
        'F1-score': [e,e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11],
       }
labels = ['8:1:1, Gini MaxDepth=5', '8:1:1, Gini MaxDepth=10', '8:1:1, Gini MaxDepth=20',
          '8:1:1, Entropy MaxDepth=5', '8:1:1, Entropy MaxDepth=10', '8:1:1, Entropy MaxDepth=20',
          '6:2:2, Gini MaxDepth=5', '6:2:2, Gini MaxDepth=10', '6:2:2, Gini MaxDepth=20',
          '6:2:2, Entropy MaxDepth=5', '6:2:2, Entropy MaxDepth=10', '6:2:2, Entropy MaxDepth=20']
df = pd.DataFrame(data,index =labels)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib
import os

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

fig, ax = plt.subplots(figsize=(10, 7))

plt.plot(fpr2, tpr2, label="8:1:1, Gini MaxDepth=20", c='blue', lw=1.5, linestyle='-')

plt.plot(fpr, tpr, label="8:1:1, Gini MaxDepth=5", c='red', lw=1.5, linestyle='--')
plt.plot(fpr1, tpr1, label="8:1:1, Gini MaxDepth=10", c='green', lw=1.5, linestyle='--')
plt.plot(fpr3, tpr3, label="8:1:1, Entropy MaxDepth=5", c='orange', lw=1.5, linestyle='--')
plt.plot(fpr4, tpr4, label="8:1:1, Entropy MaxDepth=10", c='cyan', lw=1.5, linestyle='--')
plt.plot(fpr5, tpr5, label="8:1:1, Entropy MaxDepth=20", c='magenta', lw=1.5, linestyle='--')

plt.plot(fpr6, tpr6, label="6:2:2, Gini MaxDepth=5", c='#FFA07A', lw=1.5, linestyle='--')
plt.plot(fpr7, tpr7, label="6:2:2, Gini MaxDepth=10", c='#20B2AA', lw=1.5, linestyle='--')
plt.plot(fpr8, tpr8, label="6:2:2, Gini MaxDepth=20", c='#DA70D6', lw=1.5, linestyle='--')
plt.plot(fpr9, tpr9, label="6:2:2, Entropy MaxDepth=5", c='#00CED1', lw=1.5, linestyle='--')
plt.plot(fpr10, tpr10, label="6:2:2, Entropy MaxDepth=10", c='#6495ED', lw=1.5, linestyle='--')
plt.plot(fpr11, tpr11, label="6:2:2, Entropy MaxDepth=20", c='#FF6347', lw=1.5, linestyle='--')

plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7, lw=1.5)

plt.grid(which='major', linestyle='--', linewidth=1.2, alpha=0.8)
plt.grid(which='minor', linestyle='--', linewidth=1.0, alpha=0.6)

ax.set_xlabel('False Positive Rate', fontsize=16)
ax.set_ylabel('True Positive Rate', fontsize=16)

ax.text(0.5, 0.05, "Decision Tree", fontsize=24, fontweight='bold', ha='center', transform=ax.transAxes)

plt.legend(loc='lower right', fontsize=10, frameon=True, shadow=True, title_fontsize=12)

axins = inset_axes(ax, width="33%", height="30%", loc='lower left', bbox_to_anchor=(0.5, 0.5, 1, 1), bbox_transform=ax.transAxes)

for f, t, c, ls in zip([fpr, fpr1, fpr2, fpr3, fpr4, fpr6, fpr7, fpr9, fpr10],
                       [tpr, tpr1, tpr2, tpr3, tpr4, tpr6, tpr7, tpr9, tpr10],
                       ['red', 'green', 'blue', 'orange', 'cyan', '#FFA07A', '#20B2AA', '#00CED1', '#6495ED'],
                       ['--', '--', '-', '--', '--', '--', '--', '--', '--']):
    axins.plot(f, t, c=c, lw=2, linestyle=ls)

axins.set_xlim(0.16, 0.25)
axins.set_ylim(0.6, 0.7)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='black', lw=1)

plt.tight_layout()

save_folder = "figure"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

save_path = os.path.join(save_folder, "ROC_DT_with_best_performance_highlighted.pdf")
plt.savefig(save_path, format="pdf", bbox_inches="tight")
plt.show()


# RF
RF_accuracy_record = list()

# Split up 0.8 for train and 0.2 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_test,x_validation,y_test,y_validation = train_test_split(x_test, y_test, test_size=0.5)
# n_estimators=5, max_depth=5
classifier = RandomForestClassifier(n_estimators=5, max_depth=5)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_validation)
y_pred_gbc = classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
RF_accuracy_record.append(acc)
fpr, tpr, _= metrics.roc_curve(y_validation, y_pred_gbc)
auc = metrics.roc_auc_score(y_validation, y_pred)
a = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred))
b = "{:.3}".format(metrics.precision_score(y_validation, y_pred))
c = "{:.3}".format(metrics.recall_score(y_validation, y_pred))
d = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred))
e = "{:.3}".format(metrics.f1_score(y_validation, y_pred))
# n_estimators=5, max_depth=10
classifier = RandomForestClassifier(n_estimators=5, max_depth=10)
classifier.fit(x_train, y_train)
y_pred1 = classifier.predict(x_validation)
y_pred1_gbc = classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
RF_accuracy_record.append(acc)
fpr1, tpr1, _= metrics.roc_curve(y_validation,  y_pred1_gbc)
auc1 = metrics.roc_auc_score(y_validation, y_pred1)
a1 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred1))
b1 = "{:.3}".format(metrics.precision_score(y_validation, y_pred1))
c1 = "{:.3}".format(metrics.recall_score(y_validation, y_pred1))
d1 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred1))
e1 = "{:.3}".format(metrics.f1_score(y_validation, y_pred1))
# n_estimators=5, max_depth=20
classifier = RandomForestClassifier(n_estimators=5, max_depth=20)
classifier.fit(x_train, y_train)
y_pred2 = classifier.predict(x_validation)
y_pred2_gbc = classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
RF_accuracy_record.append(acc)
fpr2, tpr2, _= metrics.roc_curve(y_validation, y_pred2_gbc)
auc2 = metrics.roc_auc_score(y_validation, y_pred2)
a2 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred2))
b2 = "{:.3}".format(metrics.precision_score(y_validation, y_pred2))
c2 = "{:.3}".format(metrics.recall_score(y_validation, y_pred2))
d2 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred2))
e2 = "{:.3}".format(metrics.f1_score(y_validation, y_pred2))
# n_estimators=10, max_depth=5
classifier = RandomForestClassifier(n_estimators=10, max_depth=5)
classifier.fit(x_train, y_train)
y_pred3 = classifier.predict(x_validation)
y_pred3_gbc = classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
RF_accuracy_record.append(acc)
fpr3, tpr3, _= metrics.roc_curve(y_validation, y_pred3_gbc)
auc3 = metrics.roc_auc_score(y_validation, y_pred3)
a3 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred3))
b3 = "{:.3}".format(metrics.precision_score(y_validation, y_pred3))
c3 = "{:.3}".format(metrics.recall_score(y_validation, y_pred3))
d3 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred3))
e3 = "{:.3}".format(metrics.f1_score(y_validation, y_pred3))
# n_estimators=20, max_depth=5
classifier = RandomForestClassifier(n_estimators=20, max_depth=5)
classifier.fit(x_train, y_train)
y_pred4 = classifier.predict(x_validation)
y_pred4_gbc = classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
RF_accuracy_record.append(acc)
fpr4, tpr4, _= metrics.roc_curve(y_validation, y_pred4_gbc)
auc4 = metrics.roc_auc_score(y_validation, y_pred4)
a4 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred4))
b4 = "{:.3}".format(metrics.precision_score(y_validation, y_pred4))
c4 = "{:.3}".format(metrics.recall_score(y_validation, y_pred4))
d4 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred4))
e4 = "{:.3}".format(metrics.f1_score(y_validation, y_pred4))



# Split up 0.6 for train and 0.4 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
x_test,x_validation,y_test,y_validation = train_test_split(x_test, y_test, test_size=0.5)
# n_estimators=5, max_depth=5
classifier = RandomForestClassifier(n_estimators=5, max_depth=5)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_validation)
y_pred_gbc = classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
RF_accuracy_record.append(acc)
fpr5, tpr5, _= metrics.roc_curve(y_validation, y_pred_gbc)
auc5 = metrics.roc_auc_score(y_validation, y_pred)
a5 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred))
b5 = "{:.3}".format(metrics.precision_score(y_validation, y_pred))
c5 = "{:.3}".format(metrics.recall_score(y_validation, y_pred))
d5 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred))
e5 = "{:.3}".format(metrics.f1_score(y_validation, y_pred))
# n_estimators=5, max_depth=10
classifier = RandomForestClassifier(n_estimators=5, max_depth=10)
classifier.fit(x_train, y_train)
y_pred1 = classifier.predict(x_validation)
y_pred1_gbc = classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
RF_accuracy_record.append(acc)
fpr6, tpr6, _= metrics.roc_curve(y_validation, y_pred1_gbc)
auc6 = metrics.roc_auc_score(y_validation, y_pred1)
a6 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred1))
b6 = "{:.3}".format(metrics.precision_score(y_validation, y_pred1))
c6 = "{:.3}".format(metrics.recall_score(y_validation, y_pred1))
d6 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred1))
e6 = "{:.3}".format(metrics.f1_score(y_validation, y_pred1))
# n_estimators=5, max_depth=20
classifier = RandomForestClassifier(n_estimators=5, max_depth=20)
classifier.fit(x_train, y_train)
y_pred2 = classifier.predict(x_validation)
y_pred2_gbc = classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
RF_accuracy_record.append(acc)
fpr7, tpr7, _= metrics.roc_curve(y_validation, y_pred2_gbc)
auc7 = metrics.roc_auc_score(y_validation, y_pred2)
a7 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred2))
b7 = "{:.3}".format(metrics.precision_score(y_validation, y_pred2))
c7 = "{:.3}".format(metrics.recall_score(y_validation, y_pred2))
d7 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred2))
e7 = "{:.3}".format(metrics.f1_score(y_validation, y_pred2))
# n_estimators=10, max_depth=5
classifier = RandomForestClassifier(n_estimators=10, max_depth=5)
classifier.fit(x_train, y_train)
y_pred3 = classifier.predict(x_validation)
y_pred3_gbc = classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
RF_accuracy_record.append(acc)
fpr8, tpr8, _= metrics.roc_curve(y_validation, y_pred3_gbc)
auc8 = metrics.roc_auc_score(y_validation, y_pred3)
a8 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred3))
b8 = "{:.3}".format(metrics.precision_score(y_validation, y_pred3))
c8 = "{:.3}".format(metrics.recall_score(y_validation, y_pred3))
d8 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred3))
e8 = "{:.3}".format(metrics.f1_score(y_validation, y_pred3))
# n_estimators=20, max_depth=5
classifier = RandomForestClassifier(n_estimators=20, max_depth=5)
classifier.fit(x_train, y_train)
y_pred4 = classifier.predict(x_validation)
y_pred4_gbc = classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
RF_accuracy_record.append(acc)
fpr9, tpr9, _= metrics.roc_curve(y_validation, y_pred4_gbc)
auc9 = metrics.roc_auc_score(y_validation, y_pred4)
a9 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred4))
b9 = "{:.3}".format(metrics.precision_score(y_validation, y_pred4))
c9 = "{:.3}".format(metrics.recall_score(y_validation, y_pred4))
d9 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred4))
e9 = "{:.3}".format(metrics.f1_score(y_validation, y_pred4))

# df table
data = {'Accuracy': [a,a1,a2,a3,a4,a5,a6,a7,a8,a9],
        'Precision': [b,b1,b2,b3,b4,b5,b6,b7,b8,b9],
        'Recall': [c,c1,c2,c3,c4,c5,c6,c7,c8,c9],
        'ROC-score': [d,d1,d2,d3,d4,d5,d6,d7,d8,d9],
        'F1-score': [e,e1,e2,e3,e4,e5,e6,e7,e8,e9],
       }
labels = ['8:1:1 n_estimators=5, max_depth=5', '8:1:1 n_estimators=5, max_depth=10', '8:1:1 n_estimators=5, max_depth=20',
          '8:1:1 n_estimators=10, max_depth=5', '8:1:1 n_estimators=20, max_depth=5','6:2:2 n_estimators=5, max_depth=5',
          '6:2:2 n_estimators=5, max_depth=10', '6:2:2 n_estimators=5, max_depth=20',
          '6:2:2 n_estimators=10, max_depth=5', '6:2:2 n_estimators=20, max_depth=5']
df = pd.DataFrame(data,index =labels)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib
import os

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})
fig, ax = plt.subplots(figsize=(10, 7))

plt.plot(fpr2, tpr2, label="8:1:1, n_estimators=5, max_depth=20", c='blue', lw=1.5, linestyle='-')

plt.plot(fpr, tpr, label="8:1:1, n_estimators=5, max_depth=5", c='red', lw=1.5, linestyle='--')
plt.plot(fpr1, tpr1, label="8:1:1, n_estimators=5, max_depth=10", c='green', lw=1.5, linestyle='--')
plt.plot(fpr3, tpr3, label="8:1:1, n_estimators=10, max_depth=5", c='orange', lw=1.5, linestyle='--')
plt.plot(fpr4, tpr4, label="8:1:1, n_estimators=20, max_depth=5", c='cyan', lw=1.5, linestyle='--')

plt.plot(fpr5, tpr5, label="6:2:2, n_estimators=5, max_depth=5", c='#FFA07A', lw=1.5, linestyle='--')
plt.plot(fpr6, tpr6, label="6:2:2, n_estimators=5, max_depth=10", c='#20B2AA', lw=1.5, linestyle='--')
plt.plot(fpr7, tpr7, label="6:2:2, n_estimators=5, max_depth=20", c='#DA70D6', lw=1.5, linestyle='--')
plt.plot(fpr8, tpr8, label="6:2:2, n_estimators=10, max_depth=5", c='#00CED1', lw=1.5, linestyle='--')
plt.plot(fpr9, tpr9, label="6:2:2, n_estimators=20, max_depth=5", c='#6495ED', lw=1.5, linestyle='--')

plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7, lw=1.5)

plt.grid(which='major', linestyle='--', linewidth=1.2, alpha=0.8)
plt.grid(which='minor', linestyle='--', linewidth=1.0, alpha=0.6)

ax.set_xlabel('False Positive Rate', fontsize=16)
ax.set_ylabel('True Positive Rate', fontsize=16)

ax.text(0.5, 0.05, "Random Forest", fontsize=24, fontweight='bold', ha='center', transform=ax.transAxes)

plt.legend(loc='lower right', fontsize=10, frameon=True, shadow=True, title_fontsize=12)

axins = inset_axes(ax, width="33%", height="30%", loc='lower left', bbox_to_anchor=(0.5, 0.5, 1, 1), bbox_transform=ax.transAxes)

for f, t, c, ls in zip([fpr, fpr1, fpr2, fpr3, fpr4, fpr5, fpr6, fpr7, fpr8, fpr9],
                       [tpr, tpr1, tpr2, tpr3, tpr4, tpr5, tpr6, tpr7, tpr8, tpr9],
                       ['red', 'green', 'blue', 'orange', 'cyan', '#FFA07A', '#20B2AA', '#DA70D6', '#00CED1', '#6495ED'],
                       ['--', '--', '-', '--', '--', '--', '--', '--', '--', '--']):
    axins.plot(f, t, c=c, lw=2, linestyle=ls)

axins.set_xlim(0.16, 0.25)
axins.set_ylim(0.6, 0.7)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='black', lw=1)

plt.tight_layout()

save_folder = "figure"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

save_path = os.path.join(save_folder, "ROC_RF_with_best_performance_highlighted.pdf")
plt.savefig(save_path, format="pdf", bbox_inches="tight")
plt.show()

# LR

LR_accuracy_record = list()

# Split up 0.8 for train and 0.2 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_test,x_validation,y_test,y_validation = train_test_split(x_test, y_test, test_size=0.5)

LR_Model = LogisticRegression(max_iter=500)
# n_estimators=5
BLR_Classifier = BaggingClassifier(estimator=LR_Model, n_estimators=5)
BLR_Classifier.fit(x_train, y_train)
y_pred = BLR_Classifier.predict(x_validation)
y_pred_gbc = BLR_Classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
LR_accuracy_record.append(acc)
a = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred))
b = "{:.3}".format(metrics.precision_score(y_validation, y_pred))
c = "{:.3}".format(metrics.recall_score(y_validation, y_pred))
d = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred))
e = "{:.3}".format(metrics.f1_score(y_validation, y_pred))
fpr, tpr, _= metrics.roc_curve(y_validation, y_pred_gbc)
auc = metrics.roc_auc_score(y_validation, y_pred)
# n_estimators=10
BLR_Classifier = BaggingClassifier(estimator=LR_Model, n_estimators=10)
BLR_Classifier.fit(x_train, y_train)
y_pred1 = BLR_Classifier.predict(x_validation)
y_pred1_gbc = BLR_Classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
LR_accuracy_record.append(acc)
fpr1, tpr1, _= metrics.roc_curve(y_validation, y_pred1_gbc)
auc1 = metrics.roc_auc_score(y_validation, y_pred1)
a1 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred1))
b1 = "{:.3}".format(metrics.precision_score(y_validation, y_pred1))
c1 = "{:.3}".format(metrics.recall_score(y_validation, y_pred1))
d1 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred1))
e1 = "{:.3}".format(metrics.f1_score(y_validation, y_pred1))

# Split up 0.6 for train and 0.4 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
x_test,x_validation,y_test,y_validation = train_test_split(x_test, y_test, test_size=0.5)
LR_Model2 = LogisticRegression(max_iter=500)
# n_estimators=5
BLR_Classifier = BaggingClassifier(estimator=LR_Model2, n_estimators=5)
BLR_Classifier.fit(x_train, y_train)
y_pred2 = BLR_Classifier.predict(x_validation)
y_pred2_gbc = BLR_Classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred2)
LR_accuracy_record.append(acc)
fpr2, tpr2, _= metrics.roc_curve(y_validation, y_pred2_gbc)
auc2 = metrics.roc_auc_score(y_validation, y_pred2)
a2 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred2))
b2 = "{:.3}".format(metrics.precision_score(y_validation, y_pred2))
c2 = "{:.3}".format(metrics.recall_score(y_validation, y_pred2))
d2 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred2))
e2 = "{:.3}".format(metrics.f1_score(y_validation, y_pred2))
# n_estimators=10
BLR_Classifier = BaggingClassifier(estimator=LR_Model2, n_estimators=10)
BLR_Classifier.fit(x_train, y_train)
y_pred3 = BLR_Classifier.predict(x_validation)
y_pred3_gbc = BLR_Classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred2)
LR_accuracy_record.append(acc)
fpr3, tpr3, _= metrics.roc_curve(y_validation, y_pred3_gbc)
auc3 = metrics.roc_auc_score(y_validation, y_pred3)
a3 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred3))
b3 = "{:.3}".format(metrics.precision_score(y_validation, y_pred3))
c3 = "{:.3}".format(metrics.recall_score(y_validation, y_pred3))
d3 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred3))
e3 = "{:.3}".format(metrics.f1_score(y_validation, y_pred3))

# Split up 0.4 for train and 0.6 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6)
x_test,x_validation,y_test,y_validation = train_test_split(x_test, y_test, test_size=0.5)
LR_Model3 = LogisticRegression(max_iter=500)
# n_estimators=5
BLR_Classifier = BaggingClassifier(estimator=LR_Model3, n_estimators=5)
BLR_Classifier.fit(x_train, y_train)
y_pred4 = BLR_Classifier.predict(x_validation)
y_pred4_gbc = BLR_Classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred4)
LR_accuracy_record.append(acc)
a4 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred4))
b4 = "{:.3}".format(metrics.precision_score(y_validation, y_pred4))
c4 = "{:.3}".format(metrics.recall_score(y_validation, y_pred4))
d4 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred4))
e4 = "{:.3}".format(metrics.f1_score(y_validation, y_pred4))
fpr4, tpr4, _= metrics.roc_curve(y_validation, y_pred4_gbc)
auc4 = metrics.roc_auc_score(y_validation, y_pred4)
# n_estimators=10
BLR_Classifier = BaggingClassifier(estimator=LR_Model, n_estimators=10)
BLR_Classifier.fit(x_train, y_train)
y_pred5 = BLR_Classifier.predict(x_validation)
y_pred5_gbc = BLR_Classifier.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred4)
LR_accuracy_record.append(acc)
fpr5, tpr5, _= metrics.roc_curve(y_validation, y_pred5_gbc)
auc5 = metrics.roc_auc_score(y_validation, y_pred5)
a5 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred5))
b5 = "{:.3}".format(metrics.precision_score(y_validation, y_pred5))
c5 = "{:.3}".format(metrics.recall_score(y_validation, y_pred5))
d5 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred5))
e5 = "{:.3}".format(metrics.f1_score(y_validation, y_pred5))


# df table
data = {'Accuracy': [a,a1,a2,a3,a4,a5],
        'Precision': [b,b1,b2,b3,b4,b5],
        'Recall': [c,c1,c2,c3,c4,c5],
        'ROC-score': [d,d1,d2,d3,d4,d5],
        'F1-score': [e,e1,e2,e3,e4,e5],
       }
labels = ['8:1:1 n_estimators=5', '8:1:1 n_estimators=10', '6:2:2 n_estimators=5','6:2:2 n_estimators=10',
          '4:3:3 n_estimators=5', '4:3:3 n_estimators=10']
df = pd.DataFrame(data,index =labels)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib
import os

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

fig, ax = plt.subplots(figsize=(10, 7))

plt.plot(fpr1, tpr1, label="8:1:1, n_estimators=10", c='blue', lw=1.5, linestyle='-')

plt.plot(fpr, tpr, label="8:1:1, n_estimators=5", c='red', lw=1.5, linestyle='--')
plt.plot(fpr2, tpr2, label="6:2:2, n_estimators=5", c='green', lw=1.5, linestyle='--')
plt.plot(fpr3, tpr3, label="6:2:2, n_estimators=10", c='orange', lw=1.5, linestyle='--')
plt.plot(fpr4, tpr4, label="4:3:3, n_estimators=5", c='cyan', lw=1.5, linestyle='--')
plt.plot(fpr5, tpr5, label="4:3:3, n_estimators=10", c='magenta', lw=1.5, linestyle='--')

plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7, lw=1.5)

plt.grid(which='major', linestyle='--', linewidth=1.2, alpha=0.8)
plt.grid(which='minor', linestyle='--', linewidth=1.0, alpha=0.6)

ax.set_xlabel('False Positive Rate', fontsize=16)
ax.set_ylabel('True Positive Rate', fontsize=16)

ax.text(0.5, 0.05, "Logistic Regression", fontsize=24, fontweight='bold', ha='center', transform=ax.transAxes)

plt.legend(loc='lower right', fontsize=10, frameon=True, shadow=True, title_fontsize=12)

axins = inset_axes(ax, width="25%", height="25%", loc='lower left', bbox_to_anchor=(0.5, 0.5, 1, 1), bbox_transform=ax.transAxes)

for f, t, c, ls in zip([fpr, fpr1, fpr2, fpr3, fpr4, fpr5],
                       [tpr, tpr1, tpr2, tpr3, tpr4, tpr5],
                       ['red', 'blue', 'green', 'orange', 'cyan', 'magenta'],
                       ['--', '-', '--', '--', '--', '--']):
    axins.plot(f, t, c=c, lw=2, linestyle=ls)

axins.set_xlim(0.11, 0.18)
axins.set_ylim(0.5, 0.6)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='black', lw=1)

plt.tight_layout()

save_folder = "figure"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

save_path = os.path.join(save_folder, "Logistic_Regression.pdf")
plt.savefig(save_path, format="pdf", bbox_inches="tight")
plt.show()

# NB
NB_accuracy_record = list()
# Split up 0.8 for train and 0.2 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_test,x_validation,y_test,y_validation = train_test_split(x_test, y_test, test_size=0.5)
nb = BernoulliNB()
nb.fit(x_train,y_train)
y_pred = nb.predict(x_validation)
y_pred_gbc = nb.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
NB_accuracy_record.append(acc)
fpr, tpr, _= metrics.roc_curve(y_validation, y_pred_gbc)
auc = metrics.roc_auc_score(y_validation, y_pred)
a = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred))
b = "{:.3}".format(metrics.precision_score(y_validation, y_pred))
c = "{:.3}".format(metrics.recall_score(y_validation, y_pred))
d = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred))
e = "{:.3}".format(metrics.f1_score(y_validation, y_pred))



# Split up 0.6 for train and 0.4 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
x_test,x_validation,y_test,y_validation = train_test_split(x_test, y_test, test_size=0.5)
nb = BernoulliNB()
nb.fit(x_train,y_train)
y_pred1 = nb.predict(x_validation)
y_pred1_gbc = nb.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred1)
NB_accuracy_record.append(acc)
fpr1, tpr1, _= metrics.roc_curve(y_validation, y_pred1_gbc)
auc1 = metrics.roc_auc_score(y_validation, y_pred1)
a1 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred1))
b1 = "{:.3}".format(metrics.precision_score(y_validation, y_pred1))
c1 = "{:.3}".format(metrics.recall_score(y_validation, y_pred1))
d1 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred1))
e1 = "{:.3}".format(metrics.f1_score(y_validation, y_pred1))


# Split up 0.4 for train and 0.6 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6)
x_test,x_validation,y_test,y_validation = train_test_split(x_test, y_test, test_size=0.5)
nb = BernoulliNB()
nb.fit(x_train,y_train)
y_pred2 = nb.predict(x_validation)
y_pred2_gbc = nb.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred2)
NB_accuracy_record.append(acc)
fpr2, tpr2, _= metrics.roc_curve(y_validation, y_pred2_gbc)
auc2 = metrics.roc_auc_score(y_validation, y_pred2)
a2 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred2))
b2 = "{:.3}".format(metrics.precision_score(y_validation, y_pred2))
c2 = "{:.3}".format(metrics.recall_score(y_validation, y_pred2))
d2 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred2))
e2 = "{:.3}".format(metrics.f1_score(y_validation, y_pred2))


# df table
data = {'Accuracy': [a,a1,a2],
        'Precision': [b,b1,b2],
        'Recall': [c,c1,c2],
        'ROC-score': [d,d1,d2],
        'F1-score': [e,e1,e2],
       }
labels = ['8:1:1', '6:2:2', '4:3:3']
print('NB')
df = pd.DataFrame(data,index =labels)




import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib
import os

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

fig, ax = plt.subplots(figsize=(10, 7))

plt.plot(fpr1, tpr1, label="6:2:2", c='blue', lw=1.5, linestyle='--')

plt.plot(fpr, tpr, label="8:1:1", c='red', lw=1.5, linestyle='-')
plt.plot(fpr2, tpr2, label="4:3:3", c='green', lw=1.5, linestyle='--')

plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7, lw=1.5)

plt.grid(which='major', linestyle='--', linewidth=1.2, alpha=0.8)
plt.grid(which='minor', linestyle='--', linewidth=1.0, alpha=0.6)

ax.set_xlabel('False Positive Rate', fontsize=16)
ax.set_ylabel('True Positive Rate', fontsize=16)

ax.text(0.5, 0.05, "Naive Bayes", fontsize=24, fontweight='bold', ha='center', transform=ax.transAxes)

plt.legend(loc='lower right', fontsize=10, frameon=True, shadow=True, title_fontsize=12)

axins = inset_axes(ax, width="25%", height="25%", loc='lower left', bbox_to_anchor=(0.5, 0.5, 1, 1), bbox_transform=ax.transAxes)

for f, t, c, ls in zip([fpr, fpr1, fpr2],
                       [tpr, tpr1, tpr2],
                       ['red', 'blue', 'green'],
                       ['-', '--', '--']):
    axins.plot(f, t, c=c, lw=2, linestyle=ls)

axins.set_xlim(0.22, 0.35)
axins.set_ylim(0.65, 0.78)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='black', lw=1)

plt.tight_layout()

save_folder = "figure"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

save_path = os.path.join(save_folder, "ROC_BernoulliNB.pdf")
plt.savefig(save_path, format="pdf", bbox_inches="tight")
plt.show()

print(df)

# KNN
KNN_accuracy_record = list()

# Split up 0.8 for train and 0.2 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_test,x_validation,y_test,y_validation = train_test_split(x_test, y_test, test_size=0.5)
# n_neighbors=3
KNN = KNeighborsClassifier(n_neighbors=3,p=2)
KNN.fit(x_train,y_train)
y_pred = KNN.predict(x_validation)
y_pred_gbc = KNN.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
KNN_accuracy_record.append(acc)
fpr, tpr, _= metrics.roc_curve(y_validation, y_pred_gbc)
auc = metrics.roc_auc_score(y_validation, y_pred)
a = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred))
b = "{:.3}".format(metrics.precision_score(y_validation, y_pred))
c = "{:.3}".format(metrics.recall_score(y_validation, y_pred))
d = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred))
e = "{:.3}".format(metrics.f1_score(y_validation, y_pred))
# n_neighbors=5
KNN = KNeighborsClassifier(n_neighbors=5,p=2)
KNN.fit(x_train,y_train)
y_pred1 = KNN.predict(x_validation)
y_pred1_gbc = KNN.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
KNN_accuracy_record.append(acc)
fpr1, tpr1, _= metrics.roc_curve(y_validation,  y_pred1_gbc)
auc1 = metrics.roc_auc_score(y_validation, y_pred1)
a1 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred1))
b1 = "{:.3}".format(metrics.precision_score(y_validation, y_pred1))
c1 = "{:.3}".format(metrics.recall_score(y_validation, y_pred1))
d1 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred1))
e1 = "{:.3}".format(metrics.f1_score(y_validation, y_pred1))
# n_neighbors=10
KNN = KNeighborsClassifier(n_neighbors=10,p=2)
KNN.fit(x_train,y_train)
y_pred2 = KNN.predict(x_validation)
y_pred2_gbc = KNN.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
KNN_accuracy_record.append(acc)
fpr2, tpr2, _= metrics.roc_curve(y_validation, y_pred2_gbc)
auc2 = metrics.roc_auc_score(y_validation, y_pred2)
a2 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred2))
b2 = "{:.3}".format(metrics.precision_score(y_validation, y_pred2))
c2 = "{:.3}".format(metrics.recall_score(y_validation, y_pred2))
d2 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred2))
e2 = "{:.3}".format(metrics.f1_score(y_validation, y_pred2))
# n_neighbors=15
KNN = KNeighborsClassifier(n_neighbors=15,p=2)
KNN.fit(x_train,y_train)
y_pred3 = KNN.predict(x_validation)
y_pred3_gbc = KNN.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
KNN_accuracy_record.append(acc)
fpr3, tpr3, _= metrics.roc_curve(y_validation, y_pred3_gbc)
auc3 = metrics.roc_auc_score(y_validation, y_pred3)
a3 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred3))
b3 = "{:.3}".format(metrics.precision_score(y_validation, y_pred3))
c3 = "{:.3}".format(metrics.recall_score(y_validation, y_pred3))
d3 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred3))
e3 = "{:.3}".format(metrics.f1_score(y_validation, y_pred3))
# n_neighbors=20
KNN = KNeighborsClassifier(n_neighbors=20,p=2)
KNN.fit(x_train,y_train)
y_pred4 = KNN.predict(x_validation)
y_pred4_gbc = KNN.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
KNN_accuracy_record.append(acc)
fpr4, tpr4, _= metrics.roc_curve(y_validation, y_pred4_gbc)
auc4 = metrics.roc_auc_score(y_validation, y_pred4)
a4 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred4))
b4 = "{:.3}".format(metrics.precision_score(y_validation, y_pred4))
c4 = "{:.3}".format(metrics.recall_score(y_validation, y_pred4))
d4 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred4))
e4 = "{:.3}".format(metrics.f1_score(y_validation, y_pred4))



# Split up 0.6 for train and 0.4 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
x_test,x_validation,y_test,y_validation = train_test_split(x_test, y_test, test_size=0.5)
# n_neighbors=3
KNN = KNeighborsClassifier(n_neighbors=3,p=2)
KNN.fit(x_train,y_train)
y_pred = KNN.predict(x_validation)
y_pred_gbc = KNN.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
KNN_accuracy_record.append(acc)
fpr5, tpr5, _= metrics.roc_curve(y_validation, y_pred_gbc)
auc5 = metrics.roc_auc_score(y_validation, y_pred)
a5 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred))
b5 = "{:.3}".format(metrics.precision_score(y_validation, y_pred))
c5 = "{:.3}".format(metrics.recall_score(y_validation, y_pred))
d5 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred))
e5 = "{:.3}".format(metrics.f1_score(y_validation, y_pred))
# n_neighbors=5
KNN = KNeighborsClassifier(n_neighbors=5,p=2)
KNN.fit(x_train,y_train)
y_pred1 = KNN.predict(x_validation)
y_pred1_gbc = KNN.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
KNN_accuracy_record.append(acc)
fpr6, tpr6, _= metrics.roc_curve(y_validation, y_pred1_gbc)
auc6 = metrics.roc_auc_score(y_validation, y_pred1)
a6 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred1))
b6 = "{:.3}".format(metrics.precision_score(y_validation, y_pred1))
c6 = "{:.3}".format(metrics.recall_score(y_validation, y_pred1))
d6 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred1))
e6 = "{:.3}".format(metrics.f1_score(y_validation, y_pred1))
# n_neighbors=10
KNN = KNeighborsClassifier(n_neighbors=10,p=2)
KNN.fit(x_train,y_train)
y_pred2 = KNN.predict(x_validation)
y_pred2_gbc = KNN.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
KNN_accuracy_record.append(acc)
fpr7, tpr7, _= metrics.roc_curve(y_validation, y_pred2_gbc)
auc7 = metrics.roc_auc_score(y_validation, y_pred2)
a7 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred2))
b7 = "{:.3}".format(metrics.precision_score(y_validation, y_pred2))
c7 = "{:.3}".format(metrics.recall_score(y_validation, y_pred2))
d7 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred2))
e7 = "{:.3}".format(metrics.f1_score(y_validation, y_pred2))
# n_neighbors=15
KNN = KNeighborsClassifier(n_neighbors=15,p=2)
KNN.fit(x_train,y_train)
y_pred3 = KNN.predict(x_validation)
y_pred3_gbc = KNN.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
KNN_accuracy_record.append(acc)
fpr8, tpr8, _= metrics.roc_curve(y_validation, y_pred3_gbc)
auc8 = metrics.roc_auc_score(y_validation, y_pred3)
a8 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred3))
b8 = "{:.3}".format(metrics.precision_score(y_validation, y_pred3))
c8 = "{:.3}".format(metrics.recall_score(y_validation, y_pred3))
d8 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred3))
e8 = "{:.3}".format(metrics.f1_score(y_validation, y_pred3))
# n_neighbors=20
KNN = KNeighborsClassifier(n_neighbors=20,p=2)
KNN.fit(x_train,y_train)
y_pred4 = KNN.predict(x_validation)
y_pred4_gbc = KNN.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
KNN_accuracy_record.append(acc)
fpr9, tpr9, _= metrics.roc_curve(y_validation, y_pred4_gbc)
auc9 = metrics.roc_auc_score(y_validation, y_pred4)
a9 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred4))
b9 = "{:.3}".format(metrics.precision_score(y_validation, y_pred4))
c9 = "{:.3}".format(metrics.recall_score(y_validation, y_pred4))
d9 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred4))
e9 = "{:.3}".format(metrics.f1_score(y_validation, y_pred4))

# df table
data = {'Accuracy': [a,a1,a2,a3,a4,a5,a6,a7,a8,a9],
        'Precision': [b,b1,b2,b3,b4,b5,b6,b7,b8,b9],
        'Recall': [c,c1,c2,c3,c4,c5,c6,c7,c8,c9],
        'ROC-score': [d,d1,d2,d3,d4,d5,d6,d7,d8,d9],
        'F1-score': [e,e1,e2,e3,e4,e5,e6,e7,e8,e9],
       }
labels = ['8:1:1 n_neighbors=3', '8:1:1 n_neighbors=5', '8:1:1 n_neighbors=10',
          '8:1:1 n_neighbors=15', '8:1:1 n_neighbors=20','6:2:2 n_neighbors=3',
          '6:2:2 n_neighbors=5', '6:2:2 n_neighbors=10',
          '6:2:2 n_neighbors=15', '6:2:2 n_neighbors=20']
df = pd.DataFrame(data,index =labels)
# show the metrics table
df.head(10)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib
import os

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

fig, ax = plt.subplots(figsize=(10, 7))

plt.plot(fpr2, tpr2, label="8:1:1, n_neighbors=10", c='blue', lw=1.5, linestyle='--')

plt.plot(fpr, tpr, label="8:1:1, n_neighbors=3", c='red', lw=1.5, linestyle='-')
plt.plot(fpr1, tpr1, label="8:1:1, n_neighbors=5", c='green', lw=1.5, linestyle='--')
plt.plot(fpr3, tpr3, label="8:1:1, n_neighbors=15", c='orange', lw=1.5, linestyle='--')
plt.plot(fpr4, tpr4, label="8:1:1, n_neighbors=20", c='cyan', lw=1.5, linestyle='--')

plt.plot(fpr5, tpr5, label="6:2:2, n_neighbors=3", c='#FFA07A', lw=1.5, linestyle='--')
plt.plot(fpr6, tpr6, label="6:2:2, n_neighbors=5", c='#20B2AA', lw=1.5, linestyle='--')
plt.plot(fpr7, tpr7, label="6:2:2, n_neighbors=10", c='#DA70D6', lw=1.5, linestyle='--')
plt.plot(fpr8, tpr8, label="6:2:2, n_neighbors=15", c='#00CED1', lw=1.5, linestyle='--')
plt.plot(fpr9, tpr9, label="6:2:2, n_neighbors=20", c='#6495ED', lw=1.5, linestyle='--')

plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7, lw=1.5)

plt.grid(which='major', linestyle='--', linewidth=1.2, alpha=0.8)
plt.grid(which='minor', linestyle='--', linewidth=1.0, alpha=0.6)

ax.set_xlabel('False Positive Rate', fontsize=16)
ax.set_ylabel('True Positive Rate', fontsize=16)

ax.text(0.5, 0.05, "K-Nearest Neighbors", fontsize=24, fontweight='bold', ha='center', transform=ax.transAxes)

plt.legend(loc='lower right', fontsize=10, frameon=True, shadow=True, title_fontsize=12)

axins = inset_axes(ax, width="33%", height="30%", loc='lower left', bbox_to_anchor=(0.5, 0.5, 1, 1), bbox_transform=ax.transAxes)

for f, t, c, ls in zip([fpr, fpr1, fpr2, fpr3, fpr4, fpr5, fpr6, fpr7, fpr8, fpr9],
                       [tpr, tpr1, tpr2, tpr3, tpr4, tpr5, tpr6, tpr7, tpr8, tpr9],
                       ['red', 'green', 'blue', 'orange', 'cyan', '#FFA07A', '#20B2AA', '#DA70D6', '#00CED1', '#6495ED'],
                       ['-', '--', '--', '--', '--', '--', '--', '--', '--', '--']):
    axins.plot(f, t, c=c, lw=2, linestyle=ls)

axins.set_xlim(0.01, 0.20)
axins.set_ylim(0.6, 0.7)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='black', lw=1)

plt.tight_layout()

save_folder = "figure"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

save_path = os.path.join(save_folder, "ROC_KNN_with_best_performance_highlighted.pdf")
plt.savefig(save_path, format="pdf", bbox_inches="tight")
plt.show()

print(df)

# GB
GB_accuracy_record = list()

# Split up 0.8 for train and 0.2 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_test,x_validation,y_test,y_validation = train_test_split(x_test, y_test, test_size=0.5)
# n_estimators=200
GB = GradientBoostingClassifier(n_estimators=200)
GB.fit(x_train, y_train)
y_pred = GB.predict(x_validation)
y_pred_gbc = GB.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
GB_accuracy_record.append(acc)
a = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred))
b = "{:.3}".format(metrics.precision_score(y_validation, y_pred))
c = "{:.3}".format(metrics.recall_score(y_validation, y_pred))
d = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred))
e = "{:.3}".format(metrics.f1_score(y_validation, y_pred))
fpr, tpr, _= metrics.roc_curve(y_validation, y_pred_gbc)
auc = metrics.roc_auc_score(y_validation, y_pred)
# n_estimators=500
GB = GradientBoostingClassifier(n_estimators=500)
GB.fit(x_train, y_train)
y_pred1 = GB.predict(x_validation)
y_pred1_gbc = GB.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
GB_accuracy_record.append(acc)
fpr1, tpr1, _= metrics.roc_curve(y_validation, y_pred1_gbc)
auc1 = metrics.roc_auc_score(y_validation, y_pred1)
a1 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred1))
b1 = "{:.3}".format(metrics.precision_score(y_validation, y_pred1))
c1 = "{:.3}".format(metrics.recall_score(y_validation, y_pred1))
d1 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred1))
e1 = "{:.3}".format(metrics.f1_score(y_validation, y_pred1))

# Split up 0.6 for train and 0.4 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
x_test,x_validation,y_test,y_validation = train_test_split(x_test, y_test, test_size=0.5)
# n_estimators=200
GB = GradientBoostingClassifier(n_estimators=200)
GB.fit(x_train, y_train)
y_pred2 = GB.predict(x_validation)
y_pred2_gbc = GB.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred2)
GB_accuracy_record.append(acc)
fpr2, tpr2, _= metrics.roc_curve(y_validation, y_pred2_gbc)
auc2 = metrics.roc_auc_score(y_validation, y_pred2)
a2 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred2))
b2 = "{:.3}".format(metrics.precision_score(y_validation, y_pred2))
c2 = "{:.3}".format(metrics.recall_score(y_validation, y_pred2))
d2 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred2))
e2 = "{:.3}".format(metrics.f1_score(y_validation, y_pred2))
# n_estimators=500
GB = GradientBoostingClassifier(n_estimators=500)
GB.fit(x_train, y_train)
y_pred3 = GB.predict(x_validation)
y_pred3_gbc = GB.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred2)
GB_accuracy_record.append(acc)
fpr3, tpr3, _= metrics.roc_curve(y_validation, y_pred3_gbc)
auc3 = metrics.roc_auc_score(y_validation, y_pred3)
a3 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred3))
b3 = "{:.3}".format(metrics.precision_score(y_validation, y_pred3))
c3 = "{:.3}".format(metrics.recall_score(y_validation, y_pred3))
d3 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred3))
e3 = "{:.3}".format(metrics.f1_score(y_validation, y_pred3))

# Split up 0.4 for train and 0.6 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6)
x_test,x_validation,y_test,y_validation = train_test_split(x_test, y_test, test_size=0.5)
LR_Model3 = LogisticRegression(max_iter=500)
# n_estimators=200
GB = GradientBoostingClassifier(n_estimators=200)
GB.fit(x_train, y_train)
y_pred4 = GB.predict(x_validation)
y_pred4_gbc = GB.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred4)
GB_accuracy_record.append(acc)
a4 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred4))
b4 = "{:.3}".format(metrics.precision_score(y_validation, y_pred4))
c4 = "{:.3}".format(metrics.recall_score(y_validation, y_pred4))
d4 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred4))
e4 = "{:.3}".format(metrics.f1_score(y_validation, y_pred4))
fpr4, tpr4, _= metrics.roc_curve(y_validation, y_pred4_gbc)
auc4 = metrics.roc_auc_score(y_validation, y_pred4)
# n_estimators=500
GB = GradientBoostingClassifier(n_estimators=500)
GB.fit(x_train, y_train)
y_pred5 = GB.predict(x_validation)
y_pred5_gbc = GB.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred4)
GB_accuracy_record.append(acc)
fpr5, tpr5, _= metrics.roc_curve(y_validation, y_pred5_gbc)
auc5 = metrics.roc_auc_score(y_validation, y_pred5)
a5 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred5))
b5 = "{:.3}".format(metrics.precision_score(y_validation, y_pred5))
c5 = "{:.3}".format(metrics.recall_score(y_validation, y_pred5))
d5 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred5))
e5 = "{:.3}".format(metrics.f1_score(y_validation, y_pred5))


# df table
data = {'Accuracy': [a,a1,a2,a3,a4,a5],
        'Precision': [b,b1,b2,b3,b4,b5],
        'Recall': [c,c1,c2,c3,c4,c5],
        'ROC-score': [d,d1,d2,d3,d4,d5],
        'F1-score': [e,e1,e2,e3,e4,e5],
       }
labels = ['8:1:1 n_estimators=5', '8:1:1 n_estimators=10', '6:2:2 n_estimators=5','6:2:2 n_estimators=10',
          '4:3:3 n_estimators=5', '4:3:3 n_estimators=10']
df = pd.DataFrame(data,index =labels)

# show the metrics table
df.head(6)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib
import os

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

fig, ax = plt.subplots(figsize=(10, 7))

plt.plot(fpr3, tpr3, label="6:2:2, n_estimators=500", c='blue', lw=1.5, linestyle='--')
plt.plot(fpr, tpr, label="8:1:1, n_estimators=200", c='red', lw=1.5, linestyle='--')
plt.plot(fpr1, tpr1, label="8:1:1, n_estimators=500", c='green', lw=1.5, linestyle='-')
plt.plot(fpr2, tpr2, label="6:2:2, n_estimators=200", c='orange', lw=1.5, linestyle='--')
plt.plot(fpr4, tpr4, label="4:3:3, n_estimators=200", c='cyan', lw=1.5, linestyle='--')
plt.plot(fpr5, tpr5, label="4:3:3, n_estimators=500", c='magenta', lw=1.5, linestyle='--')

plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7, lw=1.5)

plt.grid(which='major', linestyle='--', linewidth=1.2, alpha=0.8)
plt.grid(which='minor', linestyle='--', linewidth=1.0, alpha=0.6)

ax.set_xlabel('False Positive Rate', fontsize=16)
ax.set_ylabel('True Positive Rate', fontsize=16)

ax.text(0.5, 0.05, "Gradient Boosting", fontsize=24, fontweight='bold', ha='center', transform=ax.transAxes)

plt.legend(loc='lower right', fontsize=10, frameon=True, shadow=True, title_fontsize=12)

axins = inset_axes(ax, width="25%", height="25%", loc='lower left', bbox_to_anchor=(0.5, 0.5, 1, 1), bbox_transform=ax.transAxes)

for f, t, c, ls in zip([fpr, fpr1, fpr2, fpr3, fpr4, fpr5],
                       [tpr, tpr1, tpr2, tpr3, tpr4, tpr5],
                       ['red', 'green', 'orange', 'blue', 'cyan', 'magenta'],
                       ['--', '-', '--', '--', '--', '--']):
    axins.plot(f, t, c=c, lw=2, linestyle=ls)

axins.set_xlim(0.01, 0.13)
axins.set_ylim(0.5, 0.6)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='black', lw=1)

plt.tight_layout()

save_folder = "figure"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

save_path = os.path.join(save_folder, "ROC_GradientBoosting.pdf")
plt.savefig(save_path, format="pdf", bbox_inches="tight")
plt.show()

print(df)


#DT+LR+RF

voting_accuracy_record = list()

# Split up 0.8 for train and 0.2 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_test,x_validation,y_test,y_validation = train_test_split(x_test, y_test, test_size=0.5)

tree_clf = DecisionTreeClassifier()
log_reg_clf = LogisticRegression(max_iter=500)
randf_clf = RandomForestClassifier()
voting_clf = VotingClassifier([('DT', tree_clf), ('LR', log_reg_clf), ('RF', randf_clf)], voting='soft', weights=None)
voting_clf.fit(x_train, y_train)
y_pred = voting_clf.predict(x_validation)
y_pred_gbc = voting_clf.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
voting_accuracy_record.append(acc)
fpr, tpr, _= metrics.roc_curve(y_validation, y_pred_gbc)
auc = metrics.roc_auc_score(y_validation, y_pred)
a = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred))
b = "{:.3}".format(metrics.precision_score(y_validation, y_pred))
c = "{:.3}".format(metrics.recall_score(y_validation, y_pred))
d = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred))
e = "{:.3}".format(metrics.f1_score(y_validation, y_pred))



# Split up 0.6 for train and 0.4 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
x_test,x_validation,y_test,y_validation = train_test_split(x_test, y_test, test_size=0.5)
tree_clf = DecisionTreeClassifier()
log_reg_clf = LogisticRegression(max_iter=500)
randf_clf = RandomForestClassifier()
voting_clf = VotingClassifier([('DT', tree_clf), ('LR', log_reg_clf), ('RF', randf_clf)], voting='soft', weights=None)
voting_clf.fit(x_train, y_train)
y_pred1 = voting_clf.predict(x_validation)
y_pred1_gbc = voting_clf.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred1)
voting_accuracy_record.append(acc)
fpr1, tpr1, _= metrics.roc_curve(y_validation, y_pred1_gbc)
auc1 = metrics.roc_auc_score(y_validation, y_pred1)
a1 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred1))
b1 = "{:.3}".format(metrics.precision_score(y_validation, y_pred1))
c1 = "{:.3}".format(metrics.recall_score(y_validation, y_pred1))
d1 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred1))
e1 = "{:.3}".format(metrics.f1_score(y_validation, y_pred1))


# Split up 0.4 for train and 0.6 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6)
x_test,x_validation,y_test,y_validation = train_test_split(x_test, y_test, test_size=0.5)
tree_clf = DecisionTreeClassifier()
log_reg_clf = LogisticRegression(max_iter=500)
randf_clf = RandomForestClassifier()
voting_clf = VotingClassifier([('DT', tree_clf), ('LR', log_reg_clf), ('RF', randf_clf)], voting='soft', weights=None)
voting_clf.fit(x_train, y_train)
y_pred2 = voting_clf.predict(x_validation)
y_pred2_gbc = voting_clf.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred2)
voting_accuracy_record.append(acc)
fpr2, tpr2, _= metrics.roc_curve(y_validation, y_pred2_gbc)
auc2 = metrics.roc_auc_score(y_validation, y_pred2)
a2 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred2))
b2 = "{:.3}".format(metrics.precision_score(y_validation, y_pred2))
c2 = "{:.3}".format(metrics.recall_score(y_validation, y_pred2))
d2 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred2))
e2 = "{:.3}".format(metrics.f1_score(y_validation, y_pred2))


# df table
data = {'Accuracy': [a,a1,a2],
        'Precision': [b,b1,b2],
        'Recall': [c,c1,c2],
        'ROC-score': [d,d1,d2],
        'F1-score': [e,e1,e2],
       }
labels = ['8:1:1', '6:2:2', '4:3:3']
df = pd.DataFrame(data,index =labels)


# show the metrics table
df.head(6)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib
import os

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

fig, ax = plt.subplots(figsize=(10, 7))

plt.plot(fpr1, tpr1, label="6:2:2", c='blue', lw=1.5, linestyle='--')

plt.plot(fpr, tpr, label="8:1:1", c='red', lw=1.5, linestyle='-')
plt.plot(fpr2, tpr2, label="4:3:3", c='green', lw=1.5, linestyle='--')

plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7, lw=1.5)

plt.grid(which='major', linestyle='--', linewidth=1.2, alpha=0.8)
plt.grid(which='minor', linestyle='--', linewidth=1.0, alpha=0.6)

ax.set_xlabel('False Positive Rate', fontsize=16)
ax.set_ylabel('True Positive Rate', fontsize=16)

ax.text(0.5, 0.05, "Voting Classifier (DT + LR + RF)", fontsize=24, fontweight='bold', ha='center', transform=ax.transAxes)

plt.legend(loc='lower right', fontsize=10, frameon=True, shadow=True, title_fontsize=12)

axins = inset_axes(ax, width="25%", height="25%", loc='lower left', bbox_to_anchor=(0.5, 0.5, 1, 1), bbox_transform=ax.transAxes)

for f, t, c, ls in zip([fpr, fpr1, fpr2],
                       [tpr, tpr1, tpr2],
                       ['red', 'blue', 'green'],
                       ['-', '--', '--']):
    axins.plot(f, t, c=c, lw=2, linestyle=ls)

axins.set_xlim(0.01, 0.10)
axins.set_ylim(0.7, 0.8)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='black', lw=1)

plt.tight_layout()

save_folder = "figure"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

save_path = os.path.join(save_folder, "ROC_VotingClassifier_DT_LR_RF.pdf")
plt.savefig(save_path, format="pdf", bbox_inches="tight")
plt.show()

print(df)


#KNN+LR+RF

voting1_accuracy_record = list()

# Split up 0.8 for train and 0.2 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
x_test,x_validation,y_test,y_validation = train_test_split(x_test, y_test, test_size=0.5)

KNN_clf = KNeighborsClassifier(n_neighbors=3,p=2)
log_reg_clf = LogisticRegression(max_iter=500)
randf_clf = RandomForestClassifier()
voting_clf = VotingClassifier([('KNN', KNN_clf), ('LR', log_reg_clf), ('RF', randf_clf)], voting='soft',weights=None)
voting_clf.fit(x_train, y_train)
y_pred = voting_clf.predict(x_validation)
y_pred_gbc = voting_clf.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred)
voting1_accuracy_record.append(acc)
fpr, tpr, _= metrics.roc_curve(y_validation, y_pred_gbc)
auc = metrics.roc_auc_score(y_validation, y_pred)
a = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred))
b = "{:.3}".format(metrics.precision_score(y_validation, y_pred))
c = "{:.3}".format(metrics.recall_score(y_validation, y_pred))
d = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred))
e = "{:.3}".format(metrics.f1_score(y_validation, y_pred))



# Split up 0.6 for train and 0.4 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
x_test,x_validation,y_test,y_validation = train_test_split(x_test, y_test, test_size=0.5)
KNN_clf = KNeighborsClassifier(n_neighbors=3,p=2)
log_reg_clf = LogisticRegression(max_iter=500)
randf_clf = RandomForestClassifier()
voting_clf = VotingClassifier([('KNN', KNN_clf), ('LR', log_reg_clf), ('RF', randf_clf)], voting='soft',weights=None)
voting_clf.fit(x_train, y_train)
y_pred1 = voting_clf.predict(x_validation)
y_pred1_gbc = voting_clf.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred1)
voting1_accuracy_record.append(acc)
fpr1, tpr1, _= metrics.roc_curve(y_validation, y_pred1_gbc)
auc1 = metrics.roc_auc_score(y_validation, y_pred1)
a1 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred1))
b1 = "{:.3}".format(metrics.precision_score(y_validation, y_pred1))
c1 = "{:.3}".format(metrics.recall_score(y_validation, y_pred1))
d1 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred1))
e1 = "{:.3}".format(metrics.f1_score(y_validation, y_pred1))


# Split up 0.4 for train and 0.6 for test.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6)
x_test,x_validation,y_test,y_validation = train_test_split(x_test, y_test, test_size=0.5)
KNN_clf = KNeighborsClassifier(n_neighbors=3,p=2)
log_reg_clf = LogisticRegression(max_iter=500)
randf_clf = RandomForestClassifier()
voting_clf = VotingClassifier([('KNN', KNN_clf), ('LR', log_reg_clf), ('RF', randf_clf)], voting='soft',weights=None)
voting_clf.fit(x_train, y_train)
y_pred2 = voting_clf.predict(x_validation)
y_pred2_gbc = voting_clf.predict_proba(x_validation)[:,1]
acc=metrics.accuracy_score(y_validation, y_pred2)
voting1_accuracy_record.append(acc)
fpr2, tpr2, _= metrics.roc_curve(y_validation, y_pred2_gbc)
auc2 = metrics.roc_auc_score(y_validation, y_pred2)
a2 = "{:.3}".format(metrics.accuracy_score(y_validation, y_pred2))
b2 = "{:.3}".format(metrics.precision_score(y_validation, y_pred2))
c2 = "{:.3}".format(metrics.recall_score(y_validation, y_pred2))
d2 = "{:.3}".format(metrics.roc_auc_score(y_validation, y_pred2))
e2 = "{:.3}".format(metrics.f1_score(y_validation, y_pred2))

# df table
data = {'Accuracy': [a,a1,a2],
        'Precision': [b,b1,b2],
        'Recall': [c,c1,c2],
        'ROC-score': [d,d1,d2],
        'F1-score': [e,e1,e2],
       }
labels = ['8:1:1', '6:2:2', '4:3:3']
df = pd.DataFrame(data,index =labels)


# show the metrics table
df.head(6)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib
import os

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

fig, ax = plt.subplots(figsize=(10, 7))

plt.plot(fpr1, tpr1, label="6:2:2", c='blue', lw=1.5, linestyle='--')

plt.plot(fpr, tpr, label="8:1:1", c='red', lw=1.5, linestyle='-')
plt.plot(fpr2, tpr2, label="4:3:3", c='green', lw=1.5, linestyle='--')

plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7, lw=1.5)

plt.grid(which='major', linestyle='--', linewidth=1.2, alpha=0.8)
plt.grid(which='minor', linestyle='--', linewidth=1.0, alpha=0.6)

ax.set_xlabel('False Positive Rate', fontsize=16)
ax.set_ylabel('True Positive Rate', fontsize=16)

ax.text(0.5, 0.05, "Voting Classifier (KNN + LR + RF)", fontsize=24, fontweight='bold', ha='center', transform=ax.transAxes)

plt.legend(loc='lower right', fontsize=10, frameon=True, shadow=True, title_fontsize=12)

axins = inset_axes(ax, width="25%", height="25%", loc='lower left', bbox_to_anchor=(0.5, 0.5, 1, 1), bbox_transform=ax.transAxes)

for f, t, c, ls in zip([fpr, fpr1, fpr2],
                       [tpr, tpr1, tpr2],
                       ['red', 'blue', 'green'],
                       ['-', '--', '--']):
    axins.plot(f, t, c=c, lw=2, linestyle=ls)

axins.set_xlim(0.01, 0.12)
axins.set_ylim(0.7, 0.8)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='black', lw=1)

plt.tight_layout()

save_folder = "figure"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

save_path = os.path.join(save_folder, "ROC_VotingClassifier_KNN_LR_RF.pdf")
plt.savefig(save_path, format="pdf", bbox_inches="tight")
plt.show()

print(df)



