import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,mean_squared_error,r2_score



def load_data():
    df = pd.read_csv("ScikitLearn/AppleStore.csv")
    return df


df=pd.read_csv("ScikitLearn/AppleStore.csv")
df=df.drop(["id","track_name","currency","ver","cont_rating","prime_genre"], axis=1)
y=df["user_rating"]

y = (y >= 4).astype(int)
X = df.drop("user_rating", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline_concept=Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
        ])

random_forest=RandomForestClassifier()
random_forest.fit(X_train,y_train)
C=random_forest.score (X_test,y_test)
print("RandomForestClassifier",C)



model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
D = model.predict(X_test)
print("LogisticRegression",D)



pipeline_concept.fit(X_train,y_train)
E=pipeline_concept.score(X_test,y_test)
print("Pipeline",E)

param_grid={
    "n_estimators": [50,100,200],
    "max_depth": [None, 5,10]}

grid=GridSearchCV(RandomForestClassifier(), param_grid,cv=5)
grid.fit(X_train,y_train)


print("Best parameters")
print (grid.best_params_)

print("Best score:")
print( grid.best_score_)

param_grid={
    "model__C": [0.01,0.1,1,10],
    "model__solver":["lbfgs"]
    }

grid=GridSearchCV(pipeline_concept, param_grid,cv=3)
grid.fit(X_train,y_train)
best_model=grid.best_estimator_

y_pred=best_model.predict(X_test)

print("Best parameters - model__c | model__solver")
print(grid.best_params_)

print("Standard Scaler")
scaler = StandardScaler()
X= scaler.fit_transform(X)
print(X)

print("Simple Imputer")
imputer = SimpleImputer()
X= imputer.fit_transform(X)
print(X)


print("Accuracy:",accuracy_score (y_test,y_pred))
print("Mean Squared Error", mean_squared_error(y_test,y_pred))
print("Classification Report", classification_report(y_test,y_pred))
print("Confusion Matrix",confusion_matrix(y_test,y_pred))
print("r_Square Score",r2_score(y_test,y_test))

