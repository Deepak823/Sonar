import pandas as pd
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import time

sns.set(color_codes=True)


def train_SVC(X_train, X_test, y_train, y_test, scorer):
    SVC_classification = SVC()

    grid_parameters_SVC_classification = {'C': [50,52,55,57,60,65],
                                          'kernel': ['rbf', 'linear'],
                                          'shrinking': [False, True],
                                          'tol': [0.001, 0.0001, 0.00001],
                                          'gamma': ['auto','scale']}
    start_time = time.time()
    grid_obj = GridSearchCV(SVC_classification, param_grid=grid_parameters_SVC_classification, cv=4, n_jobs=-1,
                            scoring=scorer)
    grid_fit = grid_obj.fit(X_train, y_train)
    print(grid_fit.best_params_)
    training_time = time.time() - start_time
    best_SVC_classification = grid_fit.best_estimator_
    prediction = best_SVC_classification.predict(X_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=prediction)
    classification_rep = classification_report(y_true=y_test, y_pred=prediction)
    conf_matrix = confusion_matrix(y_test, prediction)

    return { 'model': grid_fit, 'Predictions': prediction,
            'Accuracy': accuracy, 'Classification Report': classification_rep,
            'confusion matrix': conf_matrix}



df = pd.read_csv('sonar.csv', header=None)


# Testing features correlation
df.rename(columns={60:'Label'}, inplace=True)
for i in range(0,60):
    df.rename(columns={i: 'A'+str(i)}, inplace=True)



print(df.describe())



y = df['Label'].copy()
y = LabelEncoder().fit_transform(y)

# Testing model on all features
X_df = df.copy()
X_df.drop(['Label'], inplace=True, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_df.values, y, test_size=0.2, shuffle=True, random_state=42)

res = train_SVC(X_train,X_test,y_train,y_test,'accuracy')
print("accuracy : "+ str(res['Accuracy']))
print(res['Classification Report'])
print(res['confusion matrix'])

# Looking at the correlation matrix to see if some features are more or less connected
plt.figure(figsize=(20,10))
c= df.corr()
sns.heatmap(c, vmin=0, vmax=1)
plt.show()
y_corr =  df['Label'].copy()
print("Close the correlation plot to continue the script")




# Trying to remove some features to have better results on recall
X_df2 = df.copy()
X_df2.drop(['Label','A20','A21','A22','A23','A24','A25','A26','A27','A28','A29'], inplace=True, axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_df2.values, y, test_size=0.2, shuffle=True, random_state=41)

res2 = train_SVC(X_train,X_test,y_train,y_test,'accuracy')
print("accuracy : "+ str(res2['Accuracy']))
print(res2['Classification Report'])
print(res2['confusion matrix'])
