import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score,mean_squared_error, classification_report, confusion_matrix,r2_score

data ={
"hours studied" : [3,4,6,4,3,7,8,10,9,8],
"Attendance": [60,70,80,50,85,90,90,95,88,89],
"previous score": [34,40,50,60,90,90,80,88,60,50],
"Assignment_score": [2,3,2,4,5,5,4,4,3,2],
"pass/fail": [0,0,0,1,1,1,1,1,1,0]}

X=df=pd.DataFrame(data)
y=df["pass/fail"]

X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)

pipeline=Pipeline([
("scalar", StandardScaler()),
("model", LogisticRegression())
])


param_grid={
"model__C": [0.01,0.1,1,10],
 "model__solver":["lbfgs"]
 }

grid=GridSearchCV(pipeline, param_grid,cv=3)
grid.fit(X_train,y_train)

best_model=grid.best_estimator_

y_pred=best_model.predict(X_test)

print("Best parameters:", grid.best_params_)
print("Accuracy:", accuracy_score (y_test,y_pred))
print("Mean squared error", mean_squared_error(y_test,y_pred))
print("classification_report:", classification_report(y_test,y_pred))
print("confusion_matrix:", confusion_matrix(y_test,y_pred))
print("r2score:", r2_score(y_test,y_test))
