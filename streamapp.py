import streamlit as st
import pandas as pd
import numpy as np
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load the pre-trained model and scaler
model = joblib.load('grid_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to add a background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string});
            background-size: cover;
        }}
        .content {{
            background-color: white;
            padding: 20px;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add background image
# Uncomment the following line and specify the path to your image file
# add_bg_from_local('path/to/your/background/image.png')

# Custom CSS for styling the Streamlit app
st.markdown("""
    <style>
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #1f77b4;
        }
        .input-label {
            font-size: 18px;
            font-weight: bold;
        }
        .button {
            background-color: #1f77b4;
            color: white;
            font-size: 16px;
            font-weight: bold;
        }
        .result {
            font-size: 24px;
            font-weight: bold;
            color: #d62728;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app title
st.markdown('<p class="title">Student Dropout Prediction</p>', unsafe_allow_html=True)

# Collecting features from the user
feature_1 = st.number_input('Age', min_value=0, max_value=100, value=18, key='feature_1')
feature_2 = st.selectbox('Gender', ['Male', 'Female'], key='feature_2')
feature_3 = st.number_input('Attendance Rate', min_value=0.0, max_value=100.0, value=75.0, key='feature_3')
feature_4 = st.number_input('Parent Education Level (1-5)', min_value=1, max_value=5, value=3, key='feature_4')
feature_5 = st.number_input('Income Level (1-100)', min_value=1, max_value=100, value=50, key='feature_5')
feature_6 = st.number_input('Study Time (hours)', min_value=0, max_value=50, value=20, key='feature_6')
feature_7 = st.number_input('GPA', min_value=0.0, max_value=10.0, value=2.5, key='feature_7')
feature_8 = st.number_input('Number of Failed Subjects', min_value=0, max_value=15, value=1, key='feature_8')
feature_9 = st.number_input('Extracurricular Participation (1-5)', min_value=1, max_value=5, value=3, key='feature_9')
feature_10 = st.number_input('Living Conditions (1-5)', min_value=1, max_value=5, value=3, key='feature_10')

# Gender needs to be encoded to 0 and 1
gender_encoded = 1 if feature_2 == 'Male' else 0

# Creating a DataFrame for the input values
input_data = pd.DataFrame({
    'Age': [feature_1],
    'Gender': [gender_encoded],
    'Attendance Rate': [feature_3],
    'Parent Education Level': [feature_4],
    'Income Level': [feature_5],
    'Study Time': [feature_6],
    'GPA': [feature_7],
    'Number of Failed Subjects': [feature_8],
    'Extracurricular Participation': [feature_9],
    'Living Conditions': [feature_10]
})

# Define the expected features (make sure this list matches the features used in the scaler)
expected_features = ['Age', 'Gender', 'Attendance Rate', 'Parent Education Level', 'Income Level',
                      'Study Time', 'GPA', 'Number of Failed Subjects', 'Extracurricular Participation',
                      'Living Conditions', 'Feature11', 'Feature12', 'Feature13', 'Feature14', 'Feature15',
                      'Feature16', 'Feature17']  # Update this list with all features the scaler expects

# Add missing features with default values
for feature in expected_features:
    if feature not in input_data.columns:
        input_data[feature] = 0  # or np.nan, depending on your data

# Ensure the order of features matches the scaler's expectations
input_data = input_data[expected_features]

# Display the input data
st.write("### Input Data:")
st.write(input_data)

# Normalizing input data using the scaler
input_data_scaled = scaler.transform(input_data)

# Prediction button
if st.button('Predict Dropout', key='predict_button'):
    # Predict using the pre-trained model
    prediction = model.predict(input_data_scaled)

    # Display the prediction result
    if prediction == 1:
        result = "Dropout"
    elif prediction == 0:
        result = "Enrolled"  # Assuming 0 is "Enrolled" or "Graduated"
    else:
        result = "Graduated"  # Adjust as needed based on your model

    st.markdown(f'<p class="result"> Prediction: The student is likely to {result}.</p>', unsafe_allow_html=True)



# import streamlit as st
# import pandas as pd
# import numpy as np
# import base64
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# import joblib

# # Load the pre-trained model and scaler
# model = joblib.load('grid_rf_model.pkl')
# scaler = joblib.load('scaler.pkl')

# # Function to add a background image
# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as img_file:
#         encoded_string = base64.b64encode(img_file.read()).decode()
#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background-image: url(data:image/png;base64,{encoded_string});
#             background-size: cover;
#             background-attachment: fixed;
#         }}
#         .content {{
#             background-color: rgba(255, 255, 255, 0.9);
#             padding: 20px;
#             border-radius: 10px;
#             box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
#             max-width: 800px;
#             margin: auto;
#         }}
#         .title {{
#             font-size: 36px;
#             font-weight: bold;
#             color: #1f77b4;
#             text-align: center;
#             margin-bottom: 20px;
#         }}
#         .input-label {{
#             font-size: 18px;
#             font-weight: bold;
#             margin-bottom: 5px;
#         }}
#         .button {{
#             background-color: #1f77b4;
#             color: white;
#             font-size: 16px;
#             font-weight: bold;
#             padding: 10px 20px;
#             border: none;
#             border-radius: 5px;
#             cursor: pointer;
#             transition: background-color 0.3s ease;
#         }}
#         .button:hover {{
#             background-color: #1565c0;
#         }}
#         .result {{
#             font-size: 24px;
#             font-weight: bold;
#             color: #d62728;
#             text-align: center;
#             margin-top: 20px;
#         }}
#         .stSelectbox, .stNumberInput {{
#             margin-bottom: 15px;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# # Add background image
# # Uncomment the following line and specify the path to your image file
# add_bg_from_local(r"C:\Users\Virat Dwivedi\Pictures\peakpx.jpg")

# # Streamlit app title
# st.markdown('<p class="title">Student Dropout Prediction</p>', unsafe_allow_html=True)

# # Container for content
# with st.container():
#     st.markdown('<div class="content">', unsafe_allow_html=True)

#     # Collecting features from the user
#     feature_1 = st.number_input('Age', min_value=0, max_value=100, value=18, key='feature_1')
#     feature_2 = st.selectbox('Gender', ['Male', 'Female'], key='feature_2')
#     feature_3 = st.number_input('Attendance Rate', min_value=0.0, max_value=100.0, value=75.0, key='feature_3')
#     feature_4 = st.number_input('Parent Education Level (1-5)', min_value=1, max_value=5, value=3, key='feature_4')
#     feature_5 = st.number_input('Income Level (1-100)', min_value=1, max_value=100, value=50, key='feature_5')
#     feature_6 = st.number_input('Study Time (hours)', min_value=0, max_value=50, value=20, key='feature_6')
#     feature_7 = st.number_input('GPA', min_value=0.0, max_value=10.0, value=2.5, key='feature_7')
#     feature_8 = st.number_input('Number of Failed Subjects', min_value=0, max_value=15, value=1, key='feature_8')
#     feature_9 = st.number_input('Extracurricular Participation (1-5)', min_value=1, max_value=5, value=3, key='feature_9')
#     feature_10 = st.number_input('Living Conditions (1-5)', min_value=1, max_value=5, value=3, key='feature_10')

#     # Gender needs to be encoded to 0 and 1
#     gender_encoded = 1 if feature_2 == 'Male' else 0

#     # Creating a DataFrame for the input values
#     input_data = pd.DataFrame({
#         'Age': [feature_1],
#         'Gender': [gender_encoded],
#         'Attendance Rate': [feature_3],
#         'Parent Education Level': [feature_4],
#         'Income Level': [feature_5],
#         'Study Time': [feature_6],
#         'GPA': [feature_7],
#         'Number of Failed Subjects': [feature_8],
#         'Extracurricular Participation': [feature_9],
#         'Living Conditions': [feature_10]
#     })

#     # Define the expected features (make sure this list matches the features used in the scaler)
#     expected_features = ['Age', 'Gender', 'Attendance Rate', 'Parent Education Level', 'Income Level',
#                           'Study Time', 'GPA', 'Number of Failed Subjects', 'Extracurricular Participation',
#                           'Living Conditions', 'Feature11', 'Feature12', 'Feature13', 'Feature14', 'Feature15',
#                           'Feature16', 'Feature17']  # Update this list with all features the scaler expects

#     # Add missing features with default values
#     for feature in expected_features:
#         if feature not in input_data.columns:
#             input_data[feature] = 0  # or np.nan, depending on your data

#     # Ensure the order of features matches the scaler's expectations
#     input_data = input_data[expected_features]

#     # Display the input data
#     st.write("### Input Data:")
#     st.write(input_data)

#     # Normalizing input data using the scaler
#     input_data_scaled = scaler.transform(input_data)

#     # Prediction button
#     if st.button('Predict Dropout', key='predict_button', help='Click to predict dropout status'):
#         # Predict using the pre-trained model
#         prediction = model.predict(input_data_scaled)

#         # Display the prediction result
#         if prediction == 1:
#             result = "Dropout"
#         elif prediction == 0:
#             result = "Enrolled"  # Assuming 0 is "Enrolled" or "Graduated"
#         else:
#             result = "Graduated"  # Adjust as needed based on your model

#         st.markdown(f'<p class="result">### Prediction: The student is likely to {result}.</p>', unsafe_allow_html=True)

#     st.markdown('</div>', unsafe_allow_html=True)
