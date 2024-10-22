import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model_path = 'D:\\BrainTumor_Survival\\ML_MODEL\\xgboost_best_modelseconddd.pkl'
model = joblib.load(model_path)

# Extract feature names from the model if they are available
try:
    expected_columns = model.feature_names_in_
except AttributeError:
    st.error("The model does not contain feature names. Please ensure the model is trained with feature names.")

def main():
    # Set the title of the web app
    st.title('NeuroSurvive')

    # Add a description
    st.write('Enter patient information to predict survival time.')

    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader('Patient Information')

        # Add input fields for features
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
        
    # Convert categorical inputs to numerical
    gender = 1 if gender == 'Female' else 0

    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Tumor Type': [tumor_type],
        'Tumor Grade': [tumor_grade],
        'Tumor Location': [tumor_location],
        'Treatment': [treatment],
        'Treatment Outcome': [treatment_outcome],
        'Time to Recurrence (months)': [time_to_recurrence],
        'Recurrence Site': [recurrence_site]
    })

    # Ensure columns are in the same order as during model training
    input_data = input_data[expected_columns]

    # Prediction and results section
    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            prediction = model.predict(input_data)
            
            st.write(f'Predicted Survival Time for Patient ID {patient_id}: {prediction[0]:.2f} months')

            # Plotting
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=[prediction[0]], y=['Predicted Survival Time (months)'], ax=ax, palette=['blue'])
            ax.set_title('Predicted Survival Time')
            ax.set_xlim(0, max(prediction[0], 12))
            st.pyplot(fig)

if __name__ == '__main__':
    main()
