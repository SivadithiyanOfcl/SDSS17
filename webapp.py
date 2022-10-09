import streamlit as st
import pickle
#import joblib
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import random
import pandas as pd

df= pd.read_csv("scaler_data.csv")
sample = [[135.6891066,32.49463184,23.87882,22.2753,20.39501,19.16573,18.79371,0.6347936,5812],
            [340.9951205,20.58947628,23.48827,23.33776,21.32195,20.25615,19.545441,0.424659,5026],
            [39.1496906,28.10284161,21.74669,20.03493,19.17553,18.81823,18.65422,-7.90E-06,2444],
            [344.9847703,-0.352615781,23.20911,22.79291,22.08589,21.86282,21.85120,0.8181597,9215],
            [353.2015224,3.080795936,24.5489,21.44267,20.95315,20.7936,20.48442,-0.000428576,4283],
            [20.05255573,11.49788077,21.89214,21.35124,21.18755,20.843,20.76581,0.528308,11074],
            [181.6453305,42.27399522,21.20149,19.77107,19.27176,19.04226,18.94414,0.83E-05,8382],
            [168.7266013,27.6809253,21.65936,21.73216,21.61713,21.60229,21.241921,0.007728,11357],
            [11.17152657,12.00731689,20.22819,20.18983,19.53547,19.50299,19.352941,0.319689,11042],
            [171.7118263,20.49908446,18.92981,17.81624,17.39396,17.10856,16.933580,0.04427182,2498]]

#scaler_mms = joblib.load('scaler.gz')
knn_model = pickle.load(open('knn.pkl','rb'))
svc_model = pickle.load(open('svc.pkl','rb'))
naive_model = pickle.load(open('nb.pkl','rb'))
gbc_model = pickle.load(open('gbc.pkl','rb'))
tree_model = pickle.load(open('tree.pkl','rb'))
forest_model = pickle.load(open('forest.pkl','rb'))

mms = MinMaxScaler()
df = mms.fit_transform(df)

def classify(num):
    if num==0:
        return 'Galaxy'
    elif num==1:
        return 'Quasar'
    else:
        return 'Star'

def sample_input():
    
    n = random.randint(1,9)
    st.session_state.field1 = str(sample[n][0])
    st.session_state.field2 = str(sample[n][1]) 
    st.session_state.field3 = str(sample[n][2]) 
    st.session_state.field4 = str(sample[n][3]) 
    st.session_state.field5 = str(sample[n][4]) 
    st.session_state.field6 = str(sample[n][5]) 
    st.session_state.field7 = str(sample[n][6]) 
    st.session_state.field8 = str(sample[n][7])
    st.session_state.field9 = str(sample[n][8])
    
def main():
    
    st.title("SDSS17 Stellar object Classifier")
    
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Stellar Object Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['Gradient Boosting Classifier','Decision Tree','Support Vector Classifier','Random Forest Classifier',
               'Naive Bayes Classifier','K Nearest Neighbors Classifier']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    
    col1, col2,col3 = st.columns(3)

    with col1:
        alpha=st.text_input("Enter Alpha filter value 👇",placeholder="eg. 23",key="field1")
        delta=st.text_input("Enter Delta value 👇",placeholder="eg. 22",key="field2")
        redshift=st.text_input("Enter redshift value 👇",placeholder="eg. 0.6452",key="field8")
        
    with col2:
        u=st.text_input("Enter UV filter value 👇",placeholder="eg. 21",key="field3")
        g=st.text_input("Enter Green filter value 👇",placeholder="eg. 20",key="field4")
        r=st.text_input("Enter Red filter value 👇",placeholder="eg. 19",key="field5")
    
    with col3:
        
        i=st.text_input("Enter Near Infrared filter value 👇",placeholder="eg. 18",key="field6")
        z=st.text_input("Enter Far Infrared filter value 👇",placeholder="eg. 17",key="field7")
        plate=st.text_input("Enter Plate ID 👇",placeholder="eg. 5124",key="field9")
    
    c1, c2 = st.columns(2)
    with c1:
        add = st.button(label="Generate Sample Values",on_click = sample_input)
    with c2:
        classifier = st.button('Classify')

    if classifier:
        lst = [float(alpha), float(delta), float(u), float(g), float(r), float(i),
                                   float(z),float(redshift),float(plate)]
        inputs=mms.transform([lst])
        if option=='Gradient Boosting Classifier':
            st.success(classify(gbc_model.predict(inputs)))
        elif option=='Decision Tree':
            st.success(classify(tree_model.predict(inputs)))
        elif option=='Naive Bayes Classifier':
            st.success(classify(nb_model.predict(inputs)))
        elif option=='Support Vector Classifier':
            st.success(classify(svc_model.predict(inputs)))
        elif option=='Random Forest Classifier':
            st.success(classify(forest_model.predict(inputs)))
        elif option=='K Nearest Neighbors Classifier':
            st.success(classify(knn_model.predict(inputs)))
        else:
            st.success(classify(forest_model.predict(inputs)))

if __name__=='__main__':
    main()