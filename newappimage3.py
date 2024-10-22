import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import hashlib
import joblib
import sqlite3

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
    for col in expected_columns:
        if col not in data.columns:
            data[col] = 0
        else:
            if data[col].dtype.name == 'object':
                data[col] = data[col].astype('category')
    data = data.reindex(columns=expected_columns, fill_value=0)
    return data

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

def get_db_connection():
    """Create a database connection"""
    conn = sqlite3.connect('users.db')
    return conn

def add_user(username, password):
    """Add a user to the SQLite database with validation"""
    conn = get_db_connection()
    c = conn.cursor()
    
    if not is_valid_username(username):
        return "Invalid username. It must be at least 4 characters long and contain only alphanumeric characters."
    if not is_valid_password(password):
        return "Invalid password. It must be at least 8 characters long and include a mix of letters, numbers, and special characters."
    
    try:
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
        conn.commit()
        return "Signup successful!"
    except sqlite3.IntegrityError:
        return "Username already exists."
    finally:
        conn.close()

def authenticate_user(username, password):
    """Authenticate user with the SQLite database"""
    conn = get_db_connection()
    c = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, hashed_password))
    user = c.fetchone()
    conn.close()
    return user is not None

# Set page configuration
st.set_page_config(page_title="NeuroSurvive", layout="wide")

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
        color: #BACD92;  /* Beige color for title #b08d57;*/
        text-align: center;
        padding-bottom: 20px;
        border-bottom: 2px solid #BACD92;  /* Beige border b08d57*/
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
        background-color: #BACD92;  /* Beige button color b08d57 */
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
    .image-column {
        display: flex;
        justify-content: center;
        align-items: flex-start;
        height: 100%;
        padding-top: 60px;  /* Adjust the top padding to lower the image */
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
        if authenticate_user(username, password):
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
        result = add_user(username, password)
        if result == "Signup successful!":
            st.success(result + " Please login.")
            login_page()
        else:
            st.error(result)
    st.markdown("</div>", unsafe_allow_html=True)

def main_app():
    st.markdown("<div class='main'><div class='title'>NEUROSURVIVE</div>", unsafe_allow_html=True)

    # Create columns for layout
    col1, col2 = st.columns([1, 1])

    # Display the image in the left column
    with col1:
        st.markdown("<div class='image-column'>", unsafe_allow_html=True)
        try:
            st.image("brain.webp", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Input form and prediction in the right column
    with col2:
        st.subheader('Patient Information')

        patient_id = st.text_input('Patient ID')
        age = st.slider("Patient's Age", 0, 120, 50)
        gender = st.selectbox("Patient's Gender", ['Male', 'Female'])
        tumor_type = st.selectbox("Tumor Type", ['Type 1', 'Type 2', 'Type 3', 'Type 4'])
        tumor_grade = st.selectbox("Tumor Grade", ['Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'])
        tumor_location = st.selectbox("Tumor Location", ['Location 1', 'Location 2', 'Location 3', 'Location 4'])
        treatment = st.selectbox("Treatment", ['Surgery', 'Radiation', 'Chemotherapy', 'Immunotherapy'])
        treatment_outcome = st.selectbox("Treatment Outcome", ['Progressive disease', 'Partial Disease', 'Complete Response', 'Stable Disease'])
        time_to_recurrence = st.number_input('Time to Recurrence (months)', min_value=0.0, value=0.0)
        recurrence_site = st.text_input('Recurrence Site')

    if patient_id and recurrence_site:
        gender = 1 if gender == 'Female' else 0
        tumor_type_code = int(tumor_type.split(' ')[-1])
        tumor_grade_code = int(tumor_grade.split(' ')[-1])
        tumor_location_code = int(tumor_location.split(' ')[-1])
        treatment_code = ['Surgery', 'Radiation', 'Chemotherapy', 'Immunotherapy'].index(treatment)
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

        # Right-aligned Predict button
        st.markdown(
            """
            <div style="display: flex; justify-content: flex-end;">
            """,
            unsafe_allow_html=True,
        )
        if st.button('Predict'):
            try:
                # Convert DataFrame to appropriate format
                prediction = model.predict(input_data)

                st.write(f'Predicted Survival Time for Patient ID {patient_id}: {prediction[0]:.2f} months')

                # Plotting
                # Example data for the distribution of survival times (this should come from your actual data)
                survival_times = np.random.normal(loc=40, scale=10, size=1000)  # Replace with actual data

                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(survival_times, kde=True, color='skyblue', stat="count", bins=20, ax=ax)
                ax.axvline(prediction[0], color='r', linestyle='--', label=f'Predicted: {prediction[0]:.2f} months')
                ax.set_title('Distribution of Survival Time', fontsize=16, color='#333')
                ax.set_xlabel('Survival Time (months)', fontsize=14, color='#333')
                ax.set_ylabel('Count', fontsize=14, color='#333')
                ax.legend()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        if st.button('Predict'):
            st.error('Please fill all the fields.')

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