import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,mean_squared_error,r2_score

st.title("PREPROCESSING")

st.write("Read File - AppleStore[ About rating]")

df=pd.read_csv("ScikitLearn/AppleStore.csv")
df=df.drop(["id","track_name","currency","ver","cont_rating","prime_genre"], axis=1)
y=df["user_rating"]

y = (y >= 4).astype(int)
X = df.drop("user_rating", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

menu=st.sidebar.radio("Select option",["Main","StandardScaler","SimpleImputer","Metrics"])

if menu == "Main":
    col1,col2,col3=st.columns(3)
    pipeline_concept=Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression())
                ])
    with col1:
        st.write("Random Forest Classifier")
        random_forest=RandomForestClassifier()
        random_forest.fit(X_train,y_train)
        C=random_forest.score (X_test,y_test)
        st.info(C)

    with col2: 
        st.write("Logistic Regression")
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train,y_train)
        D = model.predict(X_test)
        st.info(D)

    with col3:
        st.write("Pipeline")
        pipeline_concept.fit(X_train,y_train)
        E=pipeline_concept.score(X_test,y_test)
        st.info(E)

    st.subheader("Grid Search")
    co1,co2=st.columns(2)
    param_grid={
    "n_estimators": [50,100,200],
    "max_depth": [None, 5,10]}

    grid=GridSearchCV(RandomForestClassifier(), param_grid,cv=5)
    grid.fit(X_train,y_train)

    with co1:
        st.write("Best parameters")
        st.info( grid.best_params_)
    with co2:
        st.write("Best score:")
        st.info( grid.best_score_)

    param_grid={
    "model__C": [0.01,0.1,1,10],
    "model__solver":["lbfgs"]
    }

    grid=GridSearchCV(pipeline_concept, param_grid,cv=3)
    grid.fit(X_train,y_train)
    best_model=grid.best_estimator_

    y_pred=best_model.predict(X_test)

    st.subheader("Best parameters - model__c | model__solver")
    st.info(grid.best_params_)

if menu == "StandardScaler":
    st.header("Standard Scaler")
    scaler = StandardScaler()
    X= scaler.fit_transform(X)
    st.text(X)

if menu == "SimpleImputer":
    st.header("Simple Imputer")
    imputer = SimpleImputer()
    X= imputer.fit_transform(X)
    st.text(X)


if menu == "Metrics":
    st.header("Metrics")

    pipeline_concept=Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression())
                ])

    param_grid={
    "model__C": [0.01,0.1,1,10],
    "model__solver":["lbfgs"]
    }

    grid=GridSearchCV(pipeline_concept, param_grid,cv=3)
    grid.fit(X_train,y_train)
    best_model=grid.best_estimator_

    y_pred=best_model.predict(X_test)
    st.write("Accuracy:")
    st.info(accuracy_score (y_test,y_pred))
    st.write("Mean Squared Error")
    st.info( mean_squared_error(y_test,y_pred))
    st.write("Classification Report")
    st.info( classification_report(y_test,y_pred))
    st.write("Confusion Matrix")
    st.info(confusion_matrix(y_test,y_pred))
    st.write("r_Square Score")
    st.info(r2_score(y_test,y_test))

