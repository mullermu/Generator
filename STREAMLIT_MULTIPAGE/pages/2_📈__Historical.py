import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.express as px
from progress_bar import InitBar
from tqdm import tqdm
from time import sleep

st.set_page_config(page_title="Historical", page_icon="📈")

st.markdown("# Historical Plotting")
st.image('../picture./893216.png', width=45)

st.write("""This page illustrates a historical dataset from sensor.""")

#read csv file
input = pd.read_csv('../20221208_DataForStreamlit_1.csv')
input['Start_time'] = pd.to_datetime(input['Start_time'])
input['End_time'] = pd.to_datetime(input['End_time'])
input = input.sort_values(by = 'Start_time')

#Severity plot
st.subheader('Line chart of severity')
severity = px.line(input,x='Start_time', y=['Severity'],title='Date vs. severity')
st.plotly_chart(severity)

#----------------------------------------------------------------
st.sidebar.header("Loading...")
progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
for i in range(1, 101):
        #status_text.text("%i%% Complete" % i)
        progress_bar.progress(i)
        time.sleep(0.05)

#Temperature plot
st.subheader('Line chart of temperature')
temp = px.line(input,x='Start_time', y=['Gen_DE', 'Gen_DE_1', 'Cooling_ColdSide',
       'Cooling_ColdSide_1', 'Cooling_WarmSide', 'Cooling_WarmSide_1',
       'Cooling_ColdSide_2', 'Cooling_ColdSide_3', 'Gen_NDE', 'Gen_NDE_1',
       'Cooling_WarmSide_Exc', 'Gen_stator1', 'Gen_stator1_1', 'Gen_stator2',
       'Gen_stator2_1', 'Gen_stator3', 'Gen_stator3_1', 'Gen_Ambiant',
       'Gen_LubeOil'],title='Date vs. Temperature')
st.plotly_chart(temp)

#Vibration plot
st.subheader('Line chart of vibration')
vibration = px.line(input,x='Start_time', y=['Vi_XDE', 'Vi_YDE','Vi_XNDE',
        'Vi_YNDE'],title='Date vs. Vibration')
st.plotly_chart(vibration)

#Electrical plot
st.subheader('Line chart of voltage and frequency')
EE = px.line(input,x='Start_time', y=['Voltage', 'Frequency'],
        title='Date vs. voltage and frequency')
st.plotly_chart(EE)

#Partial Discharge plot
st.subheader('Line chart of partial discharge')
PD = px.line(input,x='Start_time', y=['PD_Gen_PhaseU','PD_Gen_PhaseV','PD_Gen_PhaseW',
        'PD_Incoming'],title='Date vs. partial discharge')
st.plotly_chart(PD)


# rerun.
st.button("Re-run")

for i in range(1, 101):
        status_text.text("%i%% Complete" % i)
        time.sleep(0.05)
