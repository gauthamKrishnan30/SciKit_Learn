import streamlit as st
from sklearn.datasets import load_iris
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import pandas as pd

st.title("SciKit Learn")

menu=st.sidebar.radio("Select option",["Load_Iris","Load_Digits"])

if menu=="Load_Iris":

    st.header("Load_Iris")

    iris=load_iris()
    X=iris.data
    y=iris.target

    X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    model = LinearRegression()
    model.fit(X_train,y_train)

    y_predict = model.predict(X_test)

    st.subheader("Predictions")
    A=pd.DataFrame(y_predict,y_test,columns=["Values"])
    st.dataframe(A)


    mse=mean_squared_error(y_test,y_predict)
    col1,col2,col3=st.columns(3)
    with col1:
        mse=mean_squared_error(y_test,y_predict)
        st.subheader("Mean Sqaure")
        st.info(mse)
    with col2:
        r2=r2_score(y_test,y_predict)
        st.subheader("Rsquare_Score:")
        st.info(r2)
    with col3:
        shape=iris.data.shape
        st.subheader("Data Shape")
        st.info(shape)

    st.header("Prediction Plot")
    plt.scatter(y_predict,y_test)
    st.pyplot(plt)
    
    if "show_data" not in st.session_state:
        st.session_state.show_data=False
    
    st.write("Show/Hide")
    if st.button("Data"):
        st.session_state.show_data = not st.session_state.show_data
    if st.session_state.show_data:
        co1,co2=st.columns(2)
        with co1:
            st.subheader("X-Iris.Data")
            st.info(X)

        with co2:
            st.subheader("y-Iris.predict")
            st.info(y)
        

elif menu=="Load_Digits":

    st.header("Load_Digits")

    digits=load_digits()
    X=digits.data
    y=digits.target

    X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    model = LinearRegression()
    model.fit(X_train,y_train)

    y_predict = model.predict(X_test)

    st.subheader("Predictions")
    A=pd.DataFrame(y_predict,y_test,columns=["Values"])
    st.dataframe(A)


    mse=mean_squared_error(y_test,y_predict)
    col1,col2,col3=st.columns(3)
    with col1:
        mse=mean_squared_error(y_test,y_predict)
        st.subheader("Mean Sqaure")
        st.info(mse)
    with col2:
        r2=r2_score(y_test,y_predict)
        st.subheader("Rsquare_Score:")
        st.info(r2)
    with col3:
        shape=digits.data.shape
        st.subheader("Data Shape")
        st.info(shape)

    st.header("Prediction Plot")
    plt.scatter(y_predict,y_test)
    st.pyplot(plt)
    
    if "show_data" not in st.session_state:
        st.session_state.show_data=False
    
    st.write("Show/Hide")
    if st.button("Data"):
        st.session_state.show_data = not st.session_state.show_data
    if st.session_state.show_data:

        co1,co2=st.columns(2)
        with co1:
            st.subheader("X-Digits.Data")
            st.info(X)

        with co2:
            st.subheader("y-Digits.predict")
            st.info(y)