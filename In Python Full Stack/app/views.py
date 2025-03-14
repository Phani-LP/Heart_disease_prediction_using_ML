from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# Load and preprocess data
data = pd.read_csv(r"C:\Users\naga_\Documents\Projects\Prediction System\HeartAttack.csv", na_values='?')
data = data.drop(columns=["slope", "ca", "thal"])
data = data.dropna()
data = data.rename(columns={"num       ": "target"})
data = pd.get_dummies(data, columns=["cp", "restecg"])
numerical_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
cat_cols = list(set(data.columns) - set(numerical_cols) - {"target"})

scaler = StandardScaler()

def my_fun(data, numerical_cols, cat_cols, scaler):
    x_scaled = scaler.fit_transform(data[numerical_cols])
    x_cat = data[cat_cols].astype(float).to_numpy()
    x = np.hstack((x_cat, x_scaled))
    y = data["target"]
    return x, y

data_x, data_y = my_fun(data, numerical_cols, cat_cols, scaler)

# Train models
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, random_state=54)
svm_clas = SVC(kernel='rbf', C=2)
svm_clas.fit(X_train, y_train)
rf_clf = RandomForestClassifier(random_state=96, n_estimators=19, criterion="entropy")
rf_clf.fit(X_train, y_train)
tn_clf = TabNetClassifier(optimizer_fn=torch.optim.Adam, scheduler_params={"step_size": 20, "gamma": 1.2}, scheduler_fn=torch.optim.lr_scheduler.StepLR)
tn_clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_name=['train', 'test'], eval_metric=['auc', 'balanced_accuracy'], max_epochs=53, patience=40, batch_size=203, virtual_batch_size=203, num_workers=0, weights=1, drop_last=False)

def predict(request):
    if request.method == 'POST':
        # Get form data
        p_age = int(request.POST['age'])
        p_trestbps = int(request.POST['trestbps'])
        p_chol = int(request.POST['chol'])
        p_thalach = int(request.POST['thalach'])
        p_oldpeak = float(request.POST['oldpeak'])
        p_fbs = int(request.POST['fbs'])
        p_cp = int(request.POST['cp'])
        p_exang = int(request.POST['exang'])
        p_sex = int(request.POST['sex'])
        p_restecg = int(request.POST['restecg'])

        # Preprocess input
        p_numerical_cols = [[p_age, p_trestbps, p_chol, p_thalach, p_oldpeak]]
        p_cp_1 = p_cp == 1
        p_cp_2 = p_cp == 2
        p_cp_3 = p_cp == 3
        p_cp_4 = p_cp == 4
        p_restecg_0 = p_restecg == 0
        p_restecg_1 = p_restecg == 1
        p_restecg_2 = p_restecg == 2
        p_cat_cols = [[p_fbs, p_cp_2, p_cp_3, p_exang, p_cp_1, p_sex, p_cp_4, p_restecg_0, p_restecg_1, p_restecg_2]]

        dummy_cat1 = [[0, False, False, 0, True, 0, False, False, True, False]]
        dummy_cat2 = [[0, True, False, 0, False, 0, False, True, False, False]]
        dummy_num1 = [[30, 170, 237, 170, 0]]
        dummy_num2 = [[32, 105, 198, 165, 0]]
        p_cate_cols = p_cat_cols + dummy_cat1 + dummy_cat2
        p_numeri_cols = p_numerical_cols + dummy_num1 + dummy_num2

        p_numeri_cols = np.array(p_numeri_cols)
        p_cate_cols = np.array(p_cate_cols)
        p_cate_cols.reshape(3, 10)
        p_numeri_cols.reshape(3, 5)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        def my_function(p_numeri_cols,p_cate_cols,scaler):
            p_x_scaled = scaler.fit_transform(p_numeri_cols)
            p_x_cat = p_cate_cols
            p_x = np.hstack((p_x_cat,p_x_scaled))
            return p_x
        p_data_x = my_function(p_numeri_cols,p_cate_cols,scaler)

        # Predict
        svm_pred = svm_clas.predict(p_data_x)[0]
        rf_pred = rf_clf.predict(p_data_x)[0]
        tabnet_pred = tn_clf.predict(p_data_x)[0]

        # Determine conclusion
        count1 = 0
        if svm_pred == 1:
            count1 += 1
        if rf_pred == 1:
            count1 += 1
        if tabnet_pred == 1:
            count1 += 1
        conclusion = "The patient has a RISK of heart disease." if count1 >= 2 else "The patient seems to be NORMAL."

        result = {
            'svm': "Risk" if svm_pred == 1 else "Normal",
            'rf': "Risk" if rf_pred == 1 else "Normal",
            'tabnet': "Risk" if tabnet_pred == 1 else "Normal",
            'conclusion': conclusion,
        }

        return render(request, 'index.html', {'result': result})

    # Handle GET requests
    return render(request, 'index.html')
