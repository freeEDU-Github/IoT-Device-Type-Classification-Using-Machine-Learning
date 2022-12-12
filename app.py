import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report

#Load dataset for the modeling
data_set = pd.read_csv("final.csv")

#Load sample data for UI
data_sample = pd.read_csv("sample_data.csv")

#Text
st.markdown("<h1 style='text-align: center;'>IoT Device Type Identification Using Machine Learning</h1>", unsafe_allow_html=True)

image = Image.open('IoT.png')
st.image(image)

st.markdown("IoT Device Type Identification Using Machine Learning aims to test Extreme Gradient Boosting Classifier (XGBoost Classifier) to determine the model in terms of accuracy on testing and validation data.")

st.markdown("***")

st.subheader("Implementation")
st.markdown("**Dataset**")
st.write("The dataset for this project is available at shorturl.at/nqY08. It consists 298 columns and 1000 rows.")

st.markdown("**Data Preprocessing**")
st.markdown('<div style="text-align: justify;">The dataset will undergo to data preprocessing. Since the dataset has 298 features, XGBoost feature importance has been used to know which features have a larger effect on the model.</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,6,1])

with col2:
    image1 = Image.open('feature.png')
    st.image(image1, caption='Top 5 most and least important features')

st.markdown('<div style="text-align: justify;">The top 5 most important features are http_time_avg, ttl_avg, packet_inter_arrivel_B_firstQ, packet_inter_arrivel_A_sum, and packet_inter_arrivel_B_min. These features will be used for our data modeling.</div>', unsafe_allow_html=True)

st.markdown("***")

st.markdown("**Extreme Gradient Boosting Classifier Model (XGBoost Classifier)**")
st.markdown('<div style="text-align: justify;">The experimental results demonstrate that XGBoost Classifier model has an accuracy of 98.00% on training data and 77.00% on validation data. The model performs well in terms of accuracy. It has been found that the functionalities of the model are working as intended.</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,6,1])

with col1:
    st.write("")

with col2:
    image1 = Image.open('accuracy.png')
    st.image(image1, caption='Confusion Matrix of XGBoost Classifier Model')

with col3:
    st.write("")


st.markdown("***")

st.subheader("Prediction")
st.write("You can test the sample data below")
st.markdown(" **Features of our data:** ")
st.text(
    "HTTP Time Average: The average http response time\n"
    "TTL Average: The average time-to-live value for the period of time that a packet or data should exist on a network before being discarded\n"
    "Packet Arrival Time B First: The amount of time that elapses after the receipt of packet A until the next packet arrive\n"
    "Summation of Packet Arrival Time A: The sum value of the amount of time that elapses after the receipt of Packet B until the next packet arrive\n"
    "Minimum Packet Arrival Time B: The least amount of time that elapses after the receipt of Packet B until the next packet arrive\n"
)
st.dataframe(data_sample)

#Create a user input for each of our data set's five features. Set the decimal places into 10
http_time_avg = st.number_input("HTTP Time Average", step=1e-10, format="%.9f")
ttl_avg = st.number_input("TTL Average", step=1e-10, format="%.9f")
packet_inter_arrivel_B_firstQ = st.number_input("Packet Arrival Time B First", step=1e-10, format="%.9f")
packet_inter_arrivel_A_sum = st.number_input("Summation of Packet Arrival Time A", step=1e-10, format="%.9f")
packet_inter_arrivel_B_min = st.number_input("Minimum Packet Arrival Time B", step=1e-10, format="%.9f")

#Assign our 5 features to variable 'features'
features = ['http_time_avg', 'ttl_avg', 'packet_inter_arrivel_B_firstQ', 'packet_inter_arrivel_A_sum',
            'packet_inter_arrivel_B_min']

#Assign variable 'features' to X and 'device_category' to y from our dataset
X = data_set[features]
y = data_set['device_category']

#Training and testing of our model. The dataset will be divided into eighty percent (80%) training sets and twenty percent (20%) validation sets.
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.20)

#Extreme Gradient Boosting Classifier was used to determine the model in terms of accuracy with a parameter 'n_estimators=100'
model = XGBClassifier(n_estimators=100)
model.fit(X_train, y_train)

model_train = model.predict(X_train)
model_test = model.predict(X_test)

#XGB Classifier Confusion Matrix to know the accuracy
xgb_predicted = model.predict(X_test)
xgb_conf_matrix = confusion_matrix(y_test, xgb_predicted)
xgb_acc_score = accuracy_score(y_test, xgb_predicted)
print("confusion matrix")
print(xgb_conf_matrix)
print("\n")
print("Accuracy of XGB:",xgb_acc_score*100,'\n')
print(classification_report(y_test, xgb_predicted))

xgb_preds_train = model.predict(X_train)
xgb_preds_test = model.predict(X_test)

print('XGB:\n> Accuracy on training data = {:.4f}\n> Accuracy on validation data = {:.4f}'.format(
    accuracy_score(y_true=y_train, y_pred=xgb_preds_train),
    accuracy_score(y_true=y_test, y_pred=xgb_preds_test)
))

#Prediction
predict_val = model.predict([[http_time_avg, ttl_avg, packet_inter_arrivel_B_firstQ, packet_inter_arrivel_A_sum, packet_inter_arrivel_B_min]])
predict_val = float(predict_val)

#IoT Device Types:
if predict_val == 1:
    st.info("Device Category: Lights")

elif predict_val == 2:
    st.info("Device Category: Motion Sensor")

elif predict_val == 3:
    st.info("Device Category: Security Camera")

elif predict_val == 4:
    st.info("Device Category: Smoke Detector")

elif predict_val == 5:
    st.info("Device Category: Socket")

elif predict_val == 6:
    st.info("Device Category: Thermostat")

elif predict_val == 7:
    st.info("Device Category: Television")

elif predict_val == 8:
    st.info("Device Category: Watch")

elif predict_val == 9:
    st.info("Device Category: Water Sensor")

else:
    st.info("Device Category: Baby Monitor")