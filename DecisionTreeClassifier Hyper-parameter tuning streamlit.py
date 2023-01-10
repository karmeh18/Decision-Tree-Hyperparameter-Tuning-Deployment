import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
scaler=StandardScaler()
norm=MinMaxScaler()
le=LabelEncoder()
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


st.title("Decision Tree Hyper Parameter Tuning")

st.header("In this website, you just have change the parameters to get different accuracies on training data and test data to compare. Just cahnge the parameters as per your choice and observe the accuracies.")

data=pd.read_excel("HR_data.xlsx")
data["Sales_Occured"]=le.fit_transform(data["Sales_Occured"])
data["salary"]=le.fit_transform(data["salary"])

X=data.drop(columns=["left"])
y=data["left"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train['average_montly_hours']=scaler.fit_transform(X_train[["average_montly_hours"]])
X_test['average_montly_hours']=scaler.fit_transform(X_test[["average_montly_hours"]])

criteria=st.sidebar.selectbox("Select any option",["gini","entropy"])
depth=st.sidebar.slider("Max_Depth",min_value=0,max_value=20,step=1)
if depth==0:
    depth=None
split=st.sidebar.selectbox("Select any option for best splitting options",["best","random"])
sample=st.sidebar.slider("Select the Minimum Splits",min_value=2,max_value=20,step=1)
features=st.sidebar.slider("Select the number of features",min_value=1,value=9,max_value=9,step=1)
max_leaf=st.sidebar.slider("Select the Max Number of Leaf Nodes", min_value=0,max_value=10,step=1)
if max_leaf==0:
    max_leaf=None
        
if st.sidebar.button("Predict"):
    dtree=DecisionTreeClassifier(criterion=criteria,
                             max_depth=depth,
                             splitter=split,
                             min_samples_split=sample,
                             max_features=features,
                             max_leaf_nodes=max_leaf)

    #Test accuracy
    dtree_model=dtree.fit(X_train,y_train)
    dtree_pred=dtree_model.predict(X_test)
    acc=accuracy_score(y_test,dtree_pred)
    st.subheader("The accuracy on Test Data is: ");st.subheader(np.round(acc,3))
    
    #Training Accuracy
    dtree_model_train=dtree.fit(X_test,y_test)
    dtree_pred_train=dtree_model_train.predict(X_train)
    acc_train=accuracy_score(y_train,dtree_pred_train)
    st.subheader("The accuracy on Training Data is: ");st.subheader(np.round(acc_train,3))
else:
    st.header("Please press the Predict button to get the accuracies.")


