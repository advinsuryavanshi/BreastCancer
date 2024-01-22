import pickle
import numpy as np
import pandas as pd
import streamlit as st

pickled_model = pickle.load(open('model.pkl', 'rb'))



title = st.text_input('Enter Your Data', '')

phrase_to_list = title.split(",")

test_list = [float(i) for i in phrase_to_list]

def convert(list):
    return tuple(i for i in list)

ans = convert(test_list)


# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(ans)


# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

z = pickled_model.predict(input_data_reshaped)
if (z[0] == 0):
  st.write('The Breast cancer is Malignant')

else:
  st.write('The Breast Cancer is Benign')
