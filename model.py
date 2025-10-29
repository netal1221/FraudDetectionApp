# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import pickle

# 1️⃣ Load dataset
data = pd.read_csv('creditcard.csv')

# 2️⃣ Feature and Target
X = data.drop('Class', axis=1)
y = data['Class']

# 3️⃣ Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4️⃣ Balance classes (Fraud vs Non-Fraud)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y)

# 5️⃣ Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 6️⃣ Train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 7️⃣ Evaluate
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 8️⃣ Save model
with open('fraud_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

# 9️⃣ Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# ✅ Run check
if __name__ == "__main__":
    print("✅ model.py is running successfully!")


