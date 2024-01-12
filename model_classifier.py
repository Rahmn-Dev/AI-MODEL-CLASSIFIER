import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Import library joblib

# Memuat dataset stroke
print("Memuat dataset stroke...")
data = pd.read_csv("dataset/stroke_data.csv")

# Pisahkan atribut dan label
print("Pisahkan atribut dan label...")
X = data.drop("stroke", axis=1)
y = data["stroke"]

# Menggunakan SimpleImputer untuk menggantikan nilai NaN dengan rata-rata
print("Data Prepocessing Menggunakan SimpleImputer untuk menggantikan nilai NaN dengan rata-rata...")
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X) # data preprocessing

# Konversi hasil imputasi kembali ke DataFrame
print("Konversi hasil imputasi kembali ke DataFrame...")
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Pembagian data train dan data test
print("Pembagian data train dan data test...")
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42) # perbandingan epoch 80:20 pada test_size

# Menyetel hyperparameter KNN menggunakan validasi silang
print("Menyetel hyperparameter KNN menggunakan validasi silang...")
neighbors = list(range(1, 21))  # Coba nilai K dari 1 hingga 20
cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Menemukan nilai K terbaik
optimal_k = neighbors[cv_scores.index(max(cv_scores))]
print(f"Nilai K terbaik: {optimal_k}")

# Pelatihan model (K-Nearest Neighbors) dengan nilai K terbaik
print("Pelatihan model (K-Nearest Neighbors) dengan nilai K terbaik...")
model = KNeighborsClassifier(n_neighbors=optimal_k)
model.fit(X_train, y_train)

# Menyimpan model ke dalam file menggunakan joblib
print("Menyimpan model ke dalam file menggunakan joblib...")
joblib.dump(model, 'model_results/model_klasifikasi_kneighbors.joblib')

# Evaluasi model pada data training
print("Evaluasi model pada data training...")
train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
train_report = classification_report(y_train, train_predictions)
train_report_dict = classification_report(y_train, train_predictions, output_dict=True)
# Simpan metrik evaluasi pada data training ke dalam file CSV
# train_metrics = {
#     "Akurasi": [train_accuracy],
#     "Precision (0)": [train_report_dict["0"]["precision"]],
#     "Precision (1)": [train_report_dict["1"]["precision"]],
#     "Recall (0)": [train_report_dict["0"]["recall"]],
#     "Recall (1)": [train_report_dict["1"]["recall"]],
#     "F1-score (0)": [train_report_dict["0"]["f1-score"]],
#     "F1-score (1)": [train_report_dict["1"]["f1-score"]]
# }

train_metrics_df = pd.DataFrame(train_report_dict).transpose()
train_metrics_df.to_csv("report_data/train_metrics_report.csv", index=True)


# Evaluasi model pada data testing
print("Evaluasi model pada data testing...")
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
test_report = classification_report(y_test, test_predictions)
test_report_dict = classification_report(y_test, test_predictions,  output_dict=True)
# Simpan metrik evaluasi pada data testing ke dalam file CSV
# test_metrics = {
#     "Akurasi": [test_accuracy],
#     "Precision (0)": [test_report_dict["0"]["precision"]],
#     "Precision (1)": [test_report_dict["1"]["precision"]],
#     "Recall (0)": [test_report_dict["0"]["recall"]],
#     "Recall (1)": [test_report_dict["1"]["recall"]],
#     "F1-score (0)": [test_report_dict["0"]["f1-score"]],
#     "F1-score (1)": [test_report_dict["1"]["f1-score"]]
# }

test_metrics_df = pd.DataFrame(test_report_dict).transpose()
test_metrics_df.to_csv("report_data/test_metrics_report.csv", index=True)

print("=== Data Training ===")
print(f"Akurasi model: {train_accuracy}")
print(f"Classification Report:\n{train_report}")


print("\n=== Data Testing ===")
print(f"Akurasi model: {test_accuracy}")
print(f"Classification Report:\n{test_report}")

# Menyimpan hasil prediksi pada data train ke dalam file CSV
train_result = pd.concat([X_train, pd.Series(train_predictions, name="predicted_stroke"), y_train], axis=1)
train_result.to_csv("data-train/train_data.csv", index=False)

# Menyimpan hasil prediksi pada data test ke dalam file CSV
test_result = pd.concat([X_test, pd.Series(test_predictions, name="predicted_stroke"), y_test], axis=1)
test_result.to_csv("data-test/test_data.csv", index=False)



# Visualisasi
# Plot distribusi kelas pada data train, distribusi kelas pada data test, dan confusion matrix
fig, axes = plt.subplots(1, 3, figsize=(18, 6))


# Plot distribusi kelas pada data train
sns.countplot(x='stroke', data=train_result, ax=axes[0])
axes[0].set_title('Distribusi Kelas pada Data Latih')

# Plot distribusi kelas pada data test
sns.countplot(x='stroke', data=test_result, ax=axes[1])
axes[1].set_title('Distribusi Kelas pada Data Uji')

# Confusion Matrix untuk data test
cm = confusion_matrix(y_test, test_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Tanpa Stroke", "Dengan Stroke"], yticklabels=["Tanpa Stroke", "Dengan Stroke"], ax=axes[2])
axes[2].set_title('Confusion Matrix untuk Data Uji')

#simpan gambar visualisasi
fig.savefig("report_data/classification_visualization.png")

# Tampilkan gambar
plt.show()
