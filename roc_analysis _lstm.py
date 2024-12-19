import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes

from warnings import simplefilter
simplefilter(action='ignore')
import category_encoders

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

import copy
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

weather_data = pd.read_csv("../weatherAUS.csv")

weather_data.drop(["Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"], inplace=True, axis=1)


numerical_features = weather_data.columns[weather_data.dtypes != object]
categorical_features = weather_data.columns[weather_data.dtypes == object]

numerical_fill_cols = ["Temp9am", "Temp3pm", "WindSpeed9am", "WindSpeed3pm",
                       "MinTemp", "MaxTemp", "WindGustSpeed", "Rainfall",
                       "Pressure9am", "Pressure3pm", "Humidity9am", "Humidity3pm"]
for col in numerical_fill_cols:
    weather_data[col] = weather_data[col].fillna(weather_data[col].mean())

categorical_fill_cols = ['RainToday', 'RainTomorrow', 'WindGustDir', 'WindDir3pm', 'WindDir9am']
for col in categorical_fill_cols:
    weather_data[col] = weather_data[col].fillna(weather_data[col].mode()[0])


def find_max(df, feature, max_current):
    return np.where(df[feature] > max_current, max_current, df[feature])

weather_data['Rainfall'] = find_max(weather_data, 'Rainfall', 3.2)
weather_data['WindSpeed9am'] = find_max(weather_data, 'WindSpeed9am', 55)
weather_data['WindSpeed3pm'] = find_max(weather_data, 'WindSpeed3pm', 57)

weather_data['Date'] = pd.to_datetime(weather_data['Date'])
weather_data['Month'] = weather_data['Date'].dt.month

binary_encoder = category_encoders.BinaryEncoder(cols=['RainToday'])
weather_data = binary_encoder.fit_transform(weather_data)

label_encoder = LabelEncoder()
weather_data['Location'] = label_encoder.fit_transform(weather_data['Location'].values)
weather_data['RainTomorrow'] = label_encoder.fit_transform(weather_data['RainTomorrow'].values)
weather_data['WindDir3pm'] = label_encoder.fit_transform(weather_data['WindDir3pm'].values)
weather_data['WindGustDir'] = label_encoder.fit_transform(weather_data['WindGustDir'].values)
weather_data['WindDir9am'] = label_encoder.fit_transform(weather_data['WindDir9am'].values)

original_data = copy.deepcopy(weather_data)

weather_data['MaxDifferenceTemp'] = pd.qcut(weather_data.MaxTemp - weather_data.MinTemp, q=4, labels=[0,1,2,3]).astype('float64')
weather_data['MaxDifferenceHumidity'] = pd.qcut(weather_data.Humidity9am - weather_data.Humidity3pm, q=4, labels=[0,1,2,3]).astype('float64')

weather_data = weather_data.drop(['MinTemp', 'MaxTemp', 'Humidity9am', 'Humidity3pm'], axis=1)

weather_data = weather_data.drop(columns=['Date'])

normal_weather_data = weather_data[weather_data.RainTomorrow == 0]
abnormal_weather_data = weather_data[weather_data.RainTomorrow == 1]
upsampled_data = resample(abnormal_weather_data, replace=True, random_state=123, n_samples=len(normal_weather_data))
weather_data = pd.concat([normal_weather_data, upsampled_data])

cols = weather_data.columns
min_max_scaler = MinMaxScaler()
weather_data_scaled = min_max_scaler.fit_transform(weather_data)
weather_data = pd.DataFrame(weather_data_scaled, columns=cols)

scaled_data = pd.DataFrame(min_max_scaler.fit_transform(weather_data), columns=weather_data.columns, index=weather_data.index)

X = scaled_data.drop(['RainTomorrow'], axis=1).values
y = scaled_data['RainTomorrow'].values

X_scaled = X.reshape((X.shape[0], 1, X.shape[1]))

def build_train_evaluate_lstm(X_train, y_train, X_val, y_val):
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(32))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_val, y_val),
                        callbacks=[early_stop], verbose=0)
    y_pred_prob = model.predict(X_val).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    auc_score = roc_auc_score(y_val, y_pred_prob)
    f1 = f1_score(y_val, y_pred)

    return y_pred, y_pred_prob, acc, prec, rec, auc_score, f1

split_ratios = [(0.8, 0.2), (0.6, 0.4), (0.4, 0.6)]
labels = ['8:1:1', '6:2:2', '4:3:3']
metrics_record = {'Accuracy': [], 'Precision': [], 'Recall': [], 'ROC-score': [], 'F1-score': []}
fpr_list = []
tpr_list = []
auc_list = []

for ratio, label in zip(split_ratios, labels):
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=1 - ratio[0], random_state=42,
                                                        stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=ratio[1], random_state=42,
                                                    stratify=y_temp)

    y_pred, y_pred_prob, acc, prec, rec, auc_score, f1 = build_train_evaluate_lstm(X_train, y_train, X_val, y_val)

    metrics_record['Accuracy'].append(f"{acc:.3f}")
    metrics_record['Precision'].append(f"{prec:.3f}")
    metrics_record['Recall'].append(f"{rec:.3f}")
    metrics_record['ROC-score'].append(f"{auc_score:.3f}")
    metrics_record['F1-score'].append(f"{f1:.3f}")

    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    auc_list.append(auc_score)

data_metrics = {
    'Accuracy': metrics_record['Accuracy'],
    'Precision': metrics_record['Precision'],
    'Recall': metrics_record['Recall'],
    'ROC-score': metrics_record['ROC-score'],
    'F1-score': metrics_record['F1-score'],
}
df_metrics = pd.DataFrame(data_metrics, index=labels)
print(df_metrics)

if len(fpr_list) == 3 and len(tpr_list) == 3:
    fpr, fpr1, fpr2 = fpr_list
    tpr, tpr1, tpr2 = tpr_list
else:
    raise ValueError("1")

fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')

plt.plot(fpr2, tpr2, label="4:3:3", color='blue', lw=1.5, linestyle='--')
plt.plot(fpr, tpr, label="8:1:1", color='red', lw=1.5, linestyle='-')
plt.plot(fpr1, tpr1, label="6:2:2", color='green', lw=1.5, linestyle='--')

plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.7, lw=1.5)

plt.grid(which='major', linestyle='--', linewidth=1.2, alpha=0.8)
plt.grid(which='minor', linestyle='--', linewidth=1.0, alpha=0.6)

ax.set_xlabel('False Positive Rate', fontsize=16)
ax.set_ylabel('True Positive Rate', fontsize=16)

ax.text(0.5, 0.05, "LSTM", fontsize=24, fontweight='bold', ha='center', transform=ax.transAxes)

plt.legend(loc='lower right', fontsize=10, frameon=True, shadow=True, title_fontsize=12)

axins = inset_axes(ax, width="33%", height="30%", loc='lower left',
                   bbox_to_anchor=(0.5, 0.5, 1, 1), bbox_transform=ax.transAxes)

for f, t, c, ls in zip([fpr, fpr1, fpr2],
                       [tpr, tpr1, tpr2],
                       ['red', 'green', 'blue'],
                       ['-', '--', '--']):
    axins.plot(f, t, color=c, lw=2, linestyle=ls)

axins.set_xlim(0.08, 0.18)
axins.set_ylim(0.6, 0.7)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec='black', lw=1)

ax.set_facecolor('white')

plt.tight_layout()

save_folder = "figure"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

save_path = os.path.join(save_folder, "ROC_LSTM_with_best_performance_highlighted.pdf")
plt.savefig(save_path, format="pdf", bbox_inches="tight", facecolor=fig.get_facecolor())

plt.show()

print(df_metrics)
