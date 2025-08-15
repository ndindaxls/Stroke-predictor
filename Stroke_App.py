
import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

from streamlit_option_menu import option_menu

# Page configuration
st.set_page_config(
    page_title="Health Assistant - Stroke Prediction",
    layout="wide",
    page_icon="🧑‍⚕️",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         color: #2c3e50;
#         text-align: center;
#         margin-bottom: 2rem;
#         font-weight: bold;
#     }
#     .prediction-result {
#         padding: 1rem;
#         border-radius: 10px;
#         text-align: center;
#         font-size: 1.2rem;
#         font-weight: bold;
#         margin: 1rem 0;
#     }
#     .high-risk {
#         background-color: #ffebee;
#         color: #c62828;
#         border: 2px solid #ef5350;
#     }
#     .low-risk {
#         background-color: #e8f5e8;
#         color: #2e7d32;
#         border: 2px solid #66bb6a;
#     }
#     .info-box {
#         background-color: #e3f2fd;
#         padding: 1rem;
#         border-radius: 8px;
#         border-left: 4px solid #1976d2;
#         margin: 1rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model_path ='models\model.pkl'

        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            return model
        else:
            st.error(f"Model file not found at {model_path}")
            return None

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


@st.cache_resource
def create_dummy_training_data():
    """Create dummy training data to fit the preprocessors"""
    # This creates synthetic data with the same structure as your training data
    # You should replace this with your actual training data statistics
    np.random.seed(42)  # For reproducibility

    # Create dummy data with realistic ranges for each feature
    n_samples = 1000
    dummy_data = np.random.rand(n_samples, 10)

    # Scale features to realistic ranges based on your data
    dummy_data[:, 0] = np.random.choice([0, 1], n_samples)  # gender
    dummy_data[:, 1] = np.random.uniform(18, 82, n_samples)  # age
    dummy_data[:, 2] = np.random.choice([0, 1], n_samples)  # hypertension
    dummy_data[:, 3] = np.random.choice([0, 1], n_samples)  # heart_disease
    dummy_data[:, 4] = np.random.choice([0, 1], n_samples)  # marital_status
    dummy_data[:, 5] = np.random.choice([0, 1, 2, 3, 4], n_samples)  # work_type
    dummy_data[:, 6] = np.random.choice([0, 1], n_samples)  # residence_type
    dummy_data[:, 7] = np.random.uniform(55, 272, n_samples)  # avg_glucose_level
    dummy_data[:, 8] = np.random.uniform(10.3, 97.6, n_samples)  # bmi
    dummy_data[:, 9] = np.random.choice([0, 1, 2, 3], n_samples)  # smoking_status

    return dummy_data


def preprocess_categorical_data(data_dict):
    gender_map = {'Male': 1, 'Female': 0}
    hypertension_map = {'Yes': 1, 'No': 0}
    heart_disease_map = {'Yes': 1, 'No': 0}
    marital_status_map = {'Married': 1, 'Single': 0}
    work_type_map = {
        'Private': 2,
        'Self-employed': 3,
        'Govt_job': 0,
        'Children': 4,
        'Never_worked': 1
    }
    residence_type_map = {'Urban': 1, 'Rural': 0}
    smoking_status_map = {
        'formerly smoked': 1,
        'never smoked': 2,
        'smokes': 3,
        'Unknown': 0
    }

    encoded_data = [
        float(gender_map[data_dict['gender']]),
        float(data_dict['age']),
        float(hypertension_map[data_dict['hypertension']]),
        float(heart_disease_map[data_dict['heart_disease']]),
        float(marital_status_map[data_dict['marital_status']]),
        float(work_type_map[data_dict['work_type']]),
        float(residence_type_map[data_dict['residence_type']]),
        float(data_dict['avg_glucose_level']),
        float(data_dict['bmi']),
        float(smoking_status_map[data_dict['smoking_status']])
    ]
    return np.array(encoded_data).reshape(1, -1)


def validate_inputs(age, avg_glucose_level, bmi):
    errors = []
    if age < 0 or age > 120:
        errors.append("Age must be between 0 and 120 years")
    if avg_glucose_level < 50 or avg_glucose_level > 300:
        errors.append("Average glucose level seems unusual (normal range: 70-140 mg/dL)")
    if bmi < 10 or bmi > 50:
        errors.append("BMI seems unusual (normal range: 18.5-25 kg/m²)")
    return errors


def display_risk_factors():
    with st.expander("ℹ️ Learn About Stroke Risk Factors"):
        st.markdown("""
        **Major Risk Factors for Stroke:**
        - **Age**: Risk increases with age, especially after 55
        - **High Blood Pressure**: Leading cause of stroke
        - **Heart Disease**: Increases stroke risk
        - **Diabetes**: High glucose levels damage blood vessels
        - **High Cholesterol**: Can block arteries
        - **Smoking**: Damages blood vessels and increases clot risk
        - **Obesity**: BMI > 25 increases stroke risk
        """)


def main():
    stroke_model = load_model()
    if stroke_model is None:
        st.error("Unable to load prediction model.")
        return

    # Create and fit preprocessors using dummy data
    # In a production app, you should load pre-fitted preprocessors
    dummy_data = create_dummy_training_data()
    scaler = StandardScaler()
    pca = PCA(n_components=2)

    # Fit preprocessors on dummy data
    scaled_dummy = scaler.fit_transform(dummy_data)
    pca.fit(scaled_dummy)

    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/health-checkup.png", width=80)
        selected = option_menu(
            "Health Assistant",
            ["Stroke Prediction"],
            icons=['activity'],
            menu_icon="hospital",
            default_index=0,
            styles={
                "container": {"padding": "5!important", "background-color": "#fafafa"},
                "icon": {"color": "#1976d2", "font-size": "25px"},
                "nav-link-selected": {"background-color": "#1976d2", "color": "white"}
            }
        )
        st.markdown("---")
        display_risk_factors()

    if selected == "Stroke Prediction":
        st.markdown('<h1 class="main-header">🧠 Stroke Risk Prediction</h1>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Fill in the information below to assess your stroke risk.</div>',
                    unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox('Gender:', ['Male', 'Female'], index=0)
            age = st.number_input('Age:', min_value=0, max_value=120, value=30, step=1)
            marital_status = st.selectbox('Marital Status:', ['Married', 'Single'], index=0)

        with col2:
            hypertension = st.selectbox('Hypertension:', ['No', 'Yes'], index=0)
            heart_disease = st.selectbox('Heart Disease:', ['No', 'Yes'], index=0)
            avg_glucose_level = st.number_input('Average Glucose Level (mg/dL):', min_value=50.0, max_value=300.0,
                                                value=100.0, step=0.1)
            bmi = st.number_input('BMI (kg/m²):', min_value=10.0, max_value=50.0, value=25.0, step=0.1)

        with col3:
            work_type = st.selectbox('Work Type:', ['Private', 'Self-employed', 'Govt_job', 'Children', 'Never_worked'],
                                     index=0)
            residence_type = st.selectbox('Residence Type:', ['Urban', 'Rural'], index=0)
            smoking_status = st.selectbox('Smoking Status:', ['never smoked', 'formerly smoked', 'smokes', 'Unknown'],
                                          index=0)

        if st.button('🔍 Predict Stroke Risk'):
            validation_errors = validate_inputs(age, avg_glucose_level, bmi)
            if validation_errors:
                for error in validation_errors:
                    st.error(f"• {error}")
                return

            input_data_dict = {
                'gender': gender,
                'age': age,
                'hypertension': hypertension,
                'heart_disease': heart_disease,
                'marital_status': marital_status,
                'work_type': work_type,
                'residence_type': residence_type,
                'avg_glucose_level': avg_glucose_level,
                'bmi': bmi,
                'smoking_status': smoking_status
            }

            # Preprocess the input data
            processed_data = preprocess_categorical_data(input_data_dict)

            try:
                # Apply the same preprocessing pipeline as training:
                # 1. Scale the features
                processed_scaled = scaler.transform(processed_data)

                # 2. Apply PCA transformation
                processed_final = pca.transform(processed_scaled)

                # Make prediction
                prediction = stroke_model.predict(processed_final)

                # Get probabilities if the model supports it
                try:
                    prediction_proba = stroke_model.predict_proba(processed_final)
                    if prediction[0] == 1:
                        risk_prob = prediction_proba[0][1] * 100
                        st.markdown(
                            f'<div class="prediction-result high-risk">⚠️ HIGH RISK - {risk_prob:.1f}% probability</div>',
                            unsafe_allow_html=True)
                        st.warning(
                            "⚠️ **Recommendation:** Please consult with a healthcare professional for proper evaluation.")
                    else:
                        low_risk_prob = prediction_proba[0][0] * 100
                        st.markdown(
                            f'<div class="prediction-result low-risk">✅ LOW RISK - {low_risk_prob:.1f}% probability</div>',
                            unsafe_allow_html=True)
                        st.success(
                            "✅ **Good news:** Your current risk appears low, but continue healthy lifestyle practices.")

                except AttributeError:
                    # Model doesn't support predict_proba
                    if prediction[0] == 1:
                        st.markdown('<div class="prediction-result high-risk">⚠️ HIGH RISK detected</div>',
                                    unsafe_allow_html=True)
                        st.warning(
                            "⚠️ **Recommendation:** Please consult with a healthcare professional for proper evaluation.")
                    else:
                        st.markdown('<div class="prediction-result low-risk">✅ LOW RISK detected</div>',
                                    unsafe_allow_html=True)
                        st.success(
                            "✅ **Good news:** Your current risk appears low, but continue healthy lifestyle practices.")

                # Display additional recommendations
                st.markdown("### 💡 General Health Recommendations:")
                recommendations = []

                if age > 55:
                    recommendations.append("Regular health check-ups are especially important at your age")
                if hypertension == 'Yes':
                    recommendations.append("Monitor and manage blood pressure regularly")
                if heart_disease == 'Yes':
                    recommendations.append("Follow your cardiologist's recommendations closely")
                if avg_glucose_level > 140:
                    recommendations.append("Consider monitoring blood glucose levels")
                if bmi > 25:
                    recommendations.append("Maintain a healthy weight through diet and exercise")
                if smoking_status == 'smokes':
                    recommendations.append("Consider smoking cessation programs")

                recommendations.extend([
                    "Maintain a healthy, balanced diet",
                    "Exercise regularly (as approved by your doctor)",
                    "Manage stress effectively",
                    "Get adequate sleep (7-9 hours nightly)"
                ])

                for rec in recommendations:
                    st.write(f"• {rec}")

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.info("Please check that your model was trained and saved with the same preprocessing steps.")


if __name__ == "__main__":
    main()
