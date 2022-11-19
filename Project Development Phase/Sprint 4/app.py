import streamlit as st
import pandas as pd
import pickle

st.write("""
# Heart disease Prediction App
This app predicts If a patient has a heart disease.
Data obtained from Kaggle:https://www.kaggle.com/datasets/rishidamarla/heart-disease-prediction
""")

st.sidebar.header('User Input Features')



# Collects user input features into dataframe

def user_input_features():
    age = st.sidebar.number_input('Enter your age: ')

    sex  = st.sidebar.selectbox('Sex',(0,1))
    cp = st.sidebar.selectbox('Chest pain type',(1,2,3,4))
    tres = st.sidebar.number_input('BP: ')
    chol = st.sidebar.number_input('Serum cholestoral in mg/dl: ')
    fbs = st.sidebar.selectbox('Fasting blood sugar over 120',(0,1))
    res = st.sidebar.selectbox('EKG Results:',(0,1,2))
    tha = st.sidebar.number_input('Maximum HR:')
    exa = st.sidebar.selectbox('Exercise angina: ',(0,1))
    old = st.sidebar.number_input('ST Depression ')
    slope = st.sidebar.number_input('ST Slope:')
    ca = st.sidebar.selectbox('Number of major vessels fluori',(0,1,2,3))
    thal = st.sidebar.selectbox('Thallium',(0,1,2))

    data = {'age': age,
            'sex': sex, 
            'cp': cp,
            'trestbps':tres,
            'chol': chol,
            'fbs': fbs,
            'restecg': res,
            'thalach':tha,
            'exang':exa,
            'oldpeak':old,
            'slope':slope,
            'ca':ca,
            'thal':thal
                }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

heart_dataset = pd.read_csv('heart.csv')
heart_dataset = heart_dataset.drop(columns=['target'])

df = pd.concat([input_df,heart_dataset],axis=0)

df = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

df = df[:1] # Selects only the first row (the user input data)

st.write(input_df)
# Reads in saved classification model
load_clf = pickle.load(open('model.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


#st.subheader('Prediction')
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)