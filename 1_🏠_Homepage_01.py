import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import xgboost as xgb

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Homepage",page_icon="👋")

st.sidebar.success("Select a page above.")

st.write('# Generator Health Index Prediction')  #st.title('')
st.image('picture/8716725.png', width=45)
st.markdown(
    """
    **This is a dashboard showing the *health index prediction* of generator "GT3301 at 15.15MW"**
""") 
st.markdown(
    """
    **👈 Select a demo dataset from the sidebar**
"""
)

st.image('picture/GT3301.png')


st.sidebar.header("Input features for simulation 👨🏽‍🔬")


# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

#select = st.sidebar.radio("Select Model",('XGBoost Regression','Voting Regressor'))
select = st.sidebar.radio("Select Model",('XGBoost Regression','Random Forest Regressor'))

if uploaded_file is not None:
    input_df2 = pd.read_csv(uploaded_file)
    input_df = input_df2.drop(columns=['Start_time','End_time','Severity'])
else:
    def user_input_features():
        Gen_DE = st.sidebar.slider('Gen_DE',  0.00, 120.00, 69.71)
        Gen_DE_1 = st.sidebar.slider('Gen_DE_1',  0.00, 120.00, 64.33)
        Cooling_ColdSide = st.sidebar.slider('Cooling_ColdSide',  0.00, 100.00, 50.05)
        Cooling_ColdSide_1 = st.sidebar.slider('Cooling_ColdSide_1',  0.00, 100.00, 50.33)
        Cooling_WarmSide = st.sidebar.slider('Cooling_WarmSide',  0.00, 100.00, 71.07)
        Cooling_WarmSide_1 = st.sidebar.slider('Cooling_WarmSide_1',  0.00, 100.00, 68.55)
        Cooling_ColdSide_2 = st.sidebar.slider('Cooling_ColdSide_2',  0.00, 100.00, 49.99)
        Cooling_ColdSide_3 = st.sidebar.slider('Cooling_ColdSide_3',  0.00, 100.00, 49.81)
        Gen_NDE = st.sidebar.slider('Gen_NDE',  0.00, 120.00, 70.89)
        Gen_NDE_1 = st.sidebar.slider('Gen_NDE_1',  0.00, 120.00, 69.54)
        Cooling_WarmSide_Exc = st.sidebar.slider('Cooling_WarmSide_Exc',  0.00, 100.00, 64.47)
        Gen_stator1 = st.sidebar.slider('Gen_stator1', 0.00, 150.00, 76.81)
        Gen_stator1_1 = st.sidebar.slider('Gen_stator1_1', 0.00, 150.00, 77.73)
        Gen_stator2 = st.sidebar.slider('Gen_stator2', 0.00, 150.00, 77.42)
        Gen_stator2_1 = st.sidebar.slider('Gen_stator2_1', 0.00, 150.00, 77.83)
        Gen_stator3 = st.sidebar.slider('Gen_stator3', 0.00, 150.00, 78.01)
        Gen_stator3_1 = st.sidebar.slider('Gen_stator3_1', 0.00, 150.00, 76.47)
        Gen_Ambiant = st.sidebar.slider('Gen_Ambiant', 0.00, 100.00, 32.12)
        Gen_LubeOil = st.sidebar.slider('Gen_LubeOil', 0.00, 100.00, 44.01)
        Vi_XDE = st.sidebar.slider('Vi_XDE', 0.00, 150.00,36.78)
        Vi_YDE = st.sidebar.slider('Vi_YDE', 0.00, 150.00,25.31)
        Vi_XNDE = st.sidebar.slider('Vi_XNDE', 0.00, 150.00,47.60)
        Vi_YNDE = st.sidebar.slider('Vi_YNDE', 0.00, 150.00,30.45)
        Voltage = st.sidebar.slider('Voltage', 0.00,20.00,11.13)
        Frequency = st.sidebar.slider('Frequency', 0.00,60.00,49.94)
        PD_Gen_PhaseU = st.sidebar.slider('PD_Gen_PhaseU', 0.00,180.00, 14.00)
        PD_Gen_PhaseV = st.sidebar.slider('PD_Gen_PhaseV', 0.00,180.00, 12.00)
        PD_Gen_PhaseW = st.sidebar.slider('PD_Gen_PhaseW', 0.00,180.00, 41.00)
        PD_Incoming = st.sidebar.slider('PD_Incoming', 0.00,5000.00, 90.00)
        data = {
                'Gen_DE':Gen_DE,
                'Gen_DE_1':Gen_DE_1,
                'Cooling_ColdSide':Cooling_ColdSide,
                'Cooling_ColdSide_1':Cooling_ColdSide_1,
                'Cooling_WarmSide':Cooling_WarmSide,
                'Cooling_WarmSide_1':Cooling_WarmSide_1,
                'Cooling_ColdSide_2':Cooling_ColdSide_2,
                'Cooling_ColdSide_3':Cooling_ColdSide_3,
                'Gen_NDE':Gen_NDE,
                'Gen_NDE_1':Gen_NDE_1,
                'Cooling_WarmSide_Exc':Cooling_WarmSide_Exc,
                'Gen_stator1':Gen_stator1,
                'Gen_stator1_1':Gen_stator1_1,
                'Gen_stator2':Gen_stator2,
                'Gen_stator2_1':Gen_stator2_1,
                'Gen_stator3':Gen_stator3,
                'Gen_stator3_1':Gen_stator3_1,
                'Gen_Ambiant':Gen_Ambiant,
                'Gen_LubeOil':Gen_LubeOil,
                'Vi_XDE':Vi_XDE,
                'Vi_YDE':Vi_YDE,
                'Vi_XNDE':Vi_XNDE,
                'Vi_YNDE':Vi_YNDE,
                'Voltage':Voltage,
                'Frequency':Frequency,
                'PD_Gen_PhaseU':PD_Gen_PhaseU,
                'PD_Gen_PhaseV':PD_Gen_PhaseV,
                'PD_Gen_PhaseW':PD_Gen_PhaseW,
                'PD_Incoming':PD_Incoming
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()


#-----------------------------------------------------------------
lastrow = len(input_df.index)

data_train = pd.read_csv('20221208_DataForStreamlit_1.csv')
raw_data = data_train.drop(columns=['Start_time','End_time','Severity'])


#st.table(input_df)
df = pd.concat([input_df,raw_data],axis=0,ignore_index=True)


# Selects only the first row (the user input data)
df = df[:] 
#st.table(df)


# Displays the user input features
st.subheader('1. Features for Simulation')
st.image('picture/7068006.png', width=45)
if uploaded_file is not None:
   st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using input parameters (shown below).')
    st.write(input_df)

# Scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df = scaler.fit_transform(df)
#st.write(df)

# Reads in saved regression model
load_clf1 = pickle.load(open('20221221_XGBModel.pkl', 'rb'))
#load_clf2 = pickle.load(open('./20221221_VotingModel.pkl', 'rb'))
load_clf2 = pickle.load(open('20221201_RFModel.pkl', 'rb'))


# Apply model to make predictions
predict = pd.DataFrame(df).iloc[:lastrow]
#prediction = load_clf2.predict(predict)
if select == 'XGBoost Regression':
    prediction = load_clf1.predict(predict)
elif select =='Random Forest Regressor':
    prediction = load_clf2.predict(predict)

#----------------------------------------------------------

st.subheader('2. Simulation and Prediction')
st.image('picture/2382533.png', width=45)
#st.write([prediction])
st.write('Severity')
st.write(prediction)
#-----------------------------------------------------------



st.subheader('3. Chart of severity')
st.image('picture/1807350.png', width=45)
st.write("3.1 Line chart of eeverity")
#line_fig = px.line(uploaded_file,x='Start_time', y='prediction', title='Line chart of eeverity')
#st.plotly_chart(line_fig)
st.line_chart(prediction)

st.write("3.2 Bar chart of severity")
st.bar_chart(prediction)


#----------------------------------------------------------------
#Save prediction file to csv

predic = pd.Series(prediction, name='Severity')

df_final = pd.concat([input_df.iloc[:,:31], pd.Series(predic)], axis=1)

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

csv = convert_df(df_final)


st.download_button(
    label="Download prediction as CSV",
    data=csv,
    file_name='prediction_file.csv',
    mime='text/csv',
)

#----------------------------------------------------------
