import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import hashlib
import joblib
import re

# Load the saved model
model_path = 'D:\\BrainTumor_Survival\\ML_MODEL\\xgboost_best_modelseconddd.pkl'
model = joblib.load(model_path)

# Manually specify the expected feature names based on model training
expected_columns = [
    'Age', 'Gender', 'Tumor Type', 'Tumor Grade', 'Tumor Location',
    'Treatment', 'Treatment Outcome', 'Time to Recurrence (months)', 'Recurrence Site', 'Extra Feature'
]

def preprocess_input(data):
    """Ensure input data has all required columns with correct types"""
    # Add missing columns with default values if they are not in the input data
    for col in expected_columns:
        if col not in data.columns:
            data[col] = 0
        else:
            if data[col].dtype.name == 'object':
                data[col] = data[col].astype('category')
    
    # Reindex to ensure columns are in the expected order
    data = data.reindex(columns=expected_columns, fill_value=0)
    return data

# Function to validate username and password
def is_valid_username(username):
    """Check if the username is valid"""
    return len(username) > 3 and username.isalnum()

def is_valid_password(password):
    """Check if the password is valid"""
    return (
        len(password) >= 8 and 
        any(char.isdigit() for char in password) and 
        any(char.isalpha() for char in password) and
        any(char in '!@#$%^&*()_+-=' for char in password)
    )

# Mock database with validation
users_db = {}

def add_mock_user(username, password):
    """Add a user to the mock database with validation"""
    if not is_valid_username(username):
        return "Invalid username. It must be at least 4 characters long and contain only alphanumeric characters."
    if not is_valid_password(password):
        return "Invalid password. It must be at least 8 characters long and include a mix of letters, numbers, and special characters."
    
    if username in users_db:
        return "Username already exists."
    
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    users_db[username] = {'password': hashed_password}
    return "Signup successful!"

# Pre-add mock users with validation
add_mock_user('ruby', 'ruby123!')
add_mock_user('suravi', 'quack123!')  # Updated to meet password criteria

def login(username, password):
    """Check if the username and password are in the mock database"""
    if username in users_db:
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        return users_db[username]['password'] == hashed_password
    return False

def signup(username, password):
    """Add new user to the mock database"""
    return add_mock_user(username, password)

# Set page configuration
st.set_page_config(page_title="Brain Tumor Survival Prediction", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f5f5f0;  /* Light beige background */
        color: #333;
    }
    .main {
        background-color: #ffffff;  /* White card background */
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        margin: 20px auto;
        max-width: 1200px;
    }
    .title {
        font-size: 2.8em;
        color: #b08d57;  /* Beige color for title */
        text-align: center;
        padding-bottom: 20px;
        border-bottom: 2px solid #b08d57;  /* Beige border */
    }
    .footer {
        font-size: 1.2em;
        text-align: center;
        color: #666;
        padding: 20px 0;
        margin-top: 20px;
        border-top: 2px solid #e1e5e8;  /* Light gray border */
    }
    .stSidebar {
        background-color: #ffffff;  /* White sidebar background */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #b08d57;  /* Beige button color */
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        cursor: pointer;
        font-size: 1.1em;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #a07a49;  /* Darker beige on hover */
    }
    .stTextInput>div>div>input,
    .stSelectbox>div>div>select,
    .stNumberInput>div>div>input,
    .stSlider>div>div>input {
        border: 1px solid #d1d3d4;
        border-radius: 8px;
        padding: 12px;
        font-size: 1em;
        transition: border-color 0.3s;
    }
    .stTextInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus,
    .stNumberInput>div>div>input:focus,
    .stSlider>div>div>input:focus {
        border-color: #b08d57;  /* Beige focus border */
    }
    .stMarkdown {
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

def login_page():
    st.markdown("<div class='main'><div class='title'>Login</div>", unsafe_allow_html=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(username, password):
            st.session_state['logged_in'] = True
            st.success("Login successful!")
        else:
            st.error("Invalid username or password")
    st.markdown("</div>", unsafe_allow_html=True)

def signup_page():
    st.markdown("<div class='main'><div class='title'>Signup</div>", unsafe_allow_html=True)
    username = st.text_input("New Username")
    password = st.text_input("New Password", type="password")
    if st.button("Signup"):
        result = signup(username, password)
        if result == "Signup successful!":
            st.success(result + " Please login.")
            login_page()
        else:
            st.error(result)
    st.markdown("</div>", unsafe_allow_html=True)

def main_app():
    st.markdown("<div class='main'><div class='title'>Brain Tumor Survival Time Prediction</div>", unsafe_allow_html=True)

    # Input form
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader('Patient Information')

        patient_id = st.text_input('Patient ID')
        age = st.slider("Patient's Age", 0, 120, 50)
        gender = st.selectbox("Patient's Gender", ['Male', 'Female'])
        tumor_type = st.selectbox("Tumor Type", ['Type 1', 'Type 2', 'Type 3', 'Type 4'])
        tumor_grade = st.selectbox("Tumor Grade", ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'])
        tumor_location = st.selectbox("Tumor Location", ['Location 1', 'Location 2', 'Location 3', 'Location 4'])
        treatment = st.selectbox("Treatment", ['Surgery', 'Radiation', 'Chemotherapy', 'Immunotherapy'])
        treatment_outcome = st.selectbox("Treatment Outcome", ['Success', 'Failure'])
        time_to_recurrence = st.number_input('Time to Recurrence (months)', min_value=0.0, value=0.0)
        recurrence_site = st.text_input('Recurrence Site')

    gender = 1 if gender == 'Female' else 0
    tumor_type_code = int(tumor_type.split(' ')[-1])
    tumor_grade_code = int(tumor_grade.split(' ')[-1])
    tumor_location_code = int(tumor_location.split(' ')[-1])
    treatment_code = treatment.replace(' ', '_')
    treatment_outcome_code = 1 if treatment_outcome == 'Success' else 0

    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Tumor Type': [tumor_type_code],
        'Tumor Grade': [tumor_grade_code],
        'Tumor Location': [tumor_location_code],
        'Treatment': [treatment_code],
        'Treatment Outcome': [treatment_outcome_code],
        'Time to Recurrence (months)': [time_to_recurrence],
        'Recurrence Site': [recurrence_site],
        'Extra Feature': [0]  # Add this feature with a default value
    })

    # Ensure input data matches the expected columns
    input_data = preprocess_input(input_data)

    st.write("Reindexed input data for prediction:")
    st.write(input_data)

    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            try:
                # Convert DataFrame to appropriate format
                prediction = model.predict(input_data)
                
                st.write(f'Predicted Survival Time for Patient ID {patient_id}: {prediction[0]:.2f} months')

                # Plotting
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=['Predicted Survival Time'], y=[prediction[0]], ax=ax, palette=['#b08d57'])  # Beige color
                ax.set_title('Predicted Survival Time', fontsize=16, color='#333')
                ax.set_ylabel('Months', fontsize=14, color='#333')
                ax.set_ylim(0, max(prediction[0] * 1.1, 12))  # Add some padding to the y-axis limit
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error during prediction: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# Display login/signup page or main app based on login status
if st.session_state['logged_in']:
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
    main_app()
else:
    page = st.sidebar.selectbox("Choose Page", ["Login", "Signup"])
    if page == "Login":
        login_page()
    else:
        signup_page()



st.markdown("<div class='footer'>NEUROSURVIVE</div>", unsafe_allow_html=True)