from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import joblib
import hashlib
import sqlite3
import logging
import os

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_default_secret_key')  # Use environment variables for sensitive data

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the model
model_path = 'D:\\BrainTumor_Survival\\ML_MODEL\\xgboost_best_modelseconddd.pkl'
model = joblib.load(model_path)

expected_columns = [
    'Age', 'Gender', 'Tumor Type', 'Tumor Grade', 'Tumor Location',
    'Treatment', 'Treatment Outcome', 'Time to Recurrence (months)', 'Recurrence Site', 'Extra Feature'
]

def preprocess_input(data):
    for col in expected_columns:
        if col not in data.columns:
            data[col] = 0
        else:
            if data[col].dtype.name == 'object':
                data[col] = data[col].astype('category')
    data = data.reindex(columns=expected_columns, fill_value=0)
    return data

def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

# Create the users table if it does not exist
def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
            conn.commit()
            flash("Signup successful! Please login.")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists.")
            return redirect(url_for('signup'))
        finally:
            conn.close()

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, hashed_password)).fetchone()
        conn.close()

        if user:
            session['logged_in'] = True
            session['username'] = username
            flash("Login successful!")
            return redirect(url_for('about'))
        else:
            flash("Invalid username or password.")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.")
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'logged_in' not in session:
        flash("Please login first.", "danger")
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            patient_id = request.form.get('patient_id')
            age = int(request.form.get('age'))
            gender = 1 if request.form.get('gender') == 'Female' else 0
            tumor_type = int(request.form.get('tumor_type').split(' ')[-1])
            tumor_grade = int(request.form.get('tumor_grade').split(' ')[-1])
            tumor_location = int(request.form.get('tumor_location').split(' ')[-1])
            treatment = ['Surgery', 'Radiation', 'Chemotherapy', 'Immunotherapy'].index(request.form.get('treatment'))
            treatment_outcome = {
                'Progressive Disease': 0,  # Corrected to match the form input
                'Partial Response': 1,
                'Complete Response': 2,
                'Stable Disease': 3
            }[request.form.get('treatment_outcome')]
            time_to_recurrence = float(request.form.get('time_to_recurrence'))
            recurrence_site = request.form.get('recurrence_site')

            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender],
                'Tumor Type': [tumor_type],
                'Tumor Grade': [tumor_grade],
                'Tumor Location': [tumor_location],
                'Treatment': [treatment],
                'Treatment Outcome': [treatment_outcome],
                'Time to Recurrence (months)': [time_to_recurrence],
                'Recurrence Site': [recurrence_site],
                'Extra Feature': [0]
            })

            input_data = preprocess_input(input_data)

            # Predict survival time
            prediction = model.predict(input_data)

            #return render_template('index.html', prediction=f'Predicted Survival Time for Patient ID {patient_id}: {prediction[0]:.2f} months')
            return render_template(
                    'index.html', 
                    prediction=f'Predicted Survival Time for Patient ID {patient_id}:\n\n            {prediction[0]:.2f} months'
                )

        except Exception as e:
            logging.error(f'Error during prediction: {e}')
            return render_template('index.html', error=f'Error during prediction: {e}')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
