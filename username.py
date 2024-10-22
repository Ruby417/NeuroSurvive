import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import hashlib

# Load the saved model
model_path = 'D:\\BrainTumor_Survival\\ML_MODEL\\xgboost_best_modelseconddd.pkl'
model = joblib.load(model_path)

# Manually specify the expected feature names based on model training
expected_columns = [
    'Age', 'Gender', 'Tumor Type', 'Tumor Grade', 'Tumor Location',
    'Treatment', 'Treatment Outcome', 'Time to Recurrence (months)', 'Recurrence Site', 'Extra Feature'
]

# Mock database for demo purposes
users_db = {
    'user123': {
        'password': hashlib.sha256('password123'.encode()).hexdigest()
    }
}

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

def login(username, password):
    """Check if the username and password are in the mock database"""
    if username in users_db:
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        return users_db[username]['password'] == hashed_password
    return False

def signup(username, password):
    """Add new user to the mock database"""
    if username in users_db:
        return False
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    users_db[username] = {'password': hashed_password}
    return True

def main():
    st.title('Brain Tumor Survival Time Prediction')
    
    menu = ["Login", "Signup", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if login(username, password):
                st.success("Logged in successfully!")
                st.session_state.logged_in = True
            else:
                st.error("Invalid username or password")

    elif choice == "Signup":
        st.subheader("Signup")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Signup"):
            if signup(username, password):
                st.success("Signup successful! Please log in.")
            else:
                st.error("Username already registered")

    elif choice == "Prediction":
        if 'logged_in' not in st.session_state or not st.session_state.logged_in:
            st.warning("Please log in to make a prediction")
        else:
            st.subheader('Patient Information')
            col1, col2 = st.columns([2, 1])
            
            with col1:
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
                        sns.barplot(x=['Predicted Survival Time'], y=[prediction[0]], ax=ax, palette=['blue'])
                        ax.set_title('Predicted Survival Time')
                        ax.set_ylabel('Months')
                        ax.set_ylim(0, max(prediction[0] * 1.1, 12))  # Add some padding to the y-axis limit
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()
