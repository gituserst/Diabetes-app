import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import cv2

#Create a title and sub-title
st.write("""
# Diabetes detection
Detect if someone has diabetes using machine learning and python!
""")
#Open and display an image
image = cv2.imread('C:/Users/Stefan/PycharmProjects/pythonProject1/Diabetes detection.png')
st.image(image, caption="Machine Learning", use_column_width=True)
#Get the data
df = pd.read_csv('C:/Users/Stefan/PycharmProjects/pythonProject1/Diabetes.csv')
#Set a subheader
st.subheader('Data information: ')
#Show the data as a table
st.dataframe(df)
#Show statistics of data
st.write(df.describe())
#Show the data as a chart
chart = st.bar_chart(df)

#Split the data into X and Y variables
x = df.iloc[:, 0:8].values
y = df.iloc[:,-1].values

#Split the data set into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

#Get the users input
def get_user_input():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 4)
    glucose = st.sidebar.slider('Glucose', 0, 199, 110)
    BloodPressure = st.sidebar.slider('BloodPressure', 0, 122, 70)
    SkinThickness = st.sidebar.slider('SkinThickness', 0, 99, 25)
    insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 40.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 28.0)
    DPF = st.sidebar.slider('DiabetesPedigreeFunction', 0.072, 2.32, 0.3275)
    age = st.sidebar.slider('Age', 18, 81, 25)

    #Store a dictionary into a variable
    user_data = {'pregnancies':pregnancies,
                 'glucose':glucose,
                 'BloodPressure':BloodPressure,
                 'SkinThickness':SkinThickness,
                 'insulin':insulin,
                 'BMI':BMI,
                 'DPF':DPF,
                 'age':age
                 }
    #Transform the data into a data frame
    features = pd.DataFrame(user_data, index=[0])
    return features

#Store the user input into a variable
user_input = get_user_input()

#Displey the users input
st.subheader('User input: ')
st.write(user_input)

#Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(x_train, y_train)

#Show the models metrics
st.subheader('Model test Accuracy Score: ')
st.write(str(accuracy_score(y_test, RandomForestClassifier.predict(x_test)) * 100) + '%')

#Store the models predictions
prediction = RandomForestClassifier.predict(user_input)

#Displey the classification
st.subheader('Classification: ')
st.write(prediction)










