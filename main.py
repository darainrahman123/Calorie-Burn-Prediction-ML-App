import streamlit as st
import joblib
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
MODEL_FILES = {
    'cols': 'cols.pkl',
    'model': 'LR.pkl',
    'scaler': 'scaler.pkl'
}

# Input constraints
INPUT_RANGES = {
    'age': (10, 80),
    'height': (120, 220),
    'weight': (30, 150),
    'duration': (1, 180),
    'heart_rate': (60, 200),
    'body_temp': (35.0, 42.0)
}

CALORIE_THRESHOLDS = {
    'high': 300,
    'moderate': 150
}

# ==================== UTILITY FUNCTIONS ====================

@st.cache_resource
def load_ml_resources() -> Tuple[list, Any, Any]:
    """
    Load ML model resources with error handling.
    
    Returns:
        Tuple of (expected_cols, model, scaler)
    
    Raises:
        Exception: If any model file is missing or corrupted
    """
    try:
        resources = {}
        for key, filename in MODEL_FILES.items():
            filepath = Path(filename)
            if not filepath.exists():
                raise FileNotFoundError(f"Model file not found: {filename}")
            resources[key] = joblib.load(filepath)
            logger.info(f"Successfully loaded {filename}")
        
        return resources['cols'], resources['model'], resources['scaler']
    
    except FileNotFoundError as e:
        logger.error(f"File loading error: {e}")
        st.error(f"❌ Error: {e}\n\nPlease ensure all model files (cols.pkl, LR.pkl, scaler.pkl) are in the application directory.")
        st.stop()
    except Exception as e:
        logger.error(f"Unexpected error loading resources: {e}")
        st.error(f"❌ Failed to load ML resources: {e}")
        st.stop()


def apply_custom_styling() -> None:
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    
    .stButton>button {
        width: 100%;
        height: 55px;
        font-size: 18px;
        border-radius: 10px;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #ff3333;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }
    
    .info-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 10px 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 10px 0;
    }
    
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


def create_input_dataframe(user_inputs: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a DataFrame from user inputs.
    
    Args:
        user_inputs: Dictionary containing user input values
    
    Returns:
        DataFrame with user inputs
    """
    return pd.DataFrame({
        'Age': [user_inputs['age']],
        'Height': [user_inputs['height']],
        'Weight': [user_inputs['weight']],
        'Duration': [user_inputs['duration']],
        'Heart_Rate': [user_inputs['heart_rate']],
        'Body_Temp': [user_inputs['body_temp']],
        'Sex_male': [user_inputs['sex_male']]
    })


def calculate_bmi(weight: float, height: float) -> float:
    """
    Calculate BMI from weight and height.
    
    Args:
        weight: Weight in kilograms
        height: Height in centimeters
    
    Returns:
        BMI value rounded to 2 decimal places
    """
    height_m = height / 100
    return round(weight / (height_m ** 2), 2)


def get_bmi_category(bmi: float) -> str:
    """
    Get BMI category based on BMI value.
    
    Args:
        bmi: BMI value
    
    Returns:
        BMI category string
    """
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def get_workout_intensity_feedback(prediction: float, duration: int, heart_rate: int) -> Dict[str, str]:
    """
    Provide detailed feedback on workout intensity.
    
    Args:
        prediction: Predicted calories burned
        duration: Workout duration in minutes
        heart_rate: Heart rate during workout
    
    Returns:
        Dictionary with intensity level and feedback message
    """
    avg_cal_per_min = prediction / duration if duration > 0 else 0
    
    if prediction > CALORIE_THRESHOLDS['high']:
        intensity = "High"
        message = f"🔥 Excellent! You're burning {avg_cal_per_min:.2f} calories per minute."
    elif prediction > CALORIE_THRESHOLDS['moderate']:
        intensity = "Moderate"
        message = f"💪 Good pace! You're burning {avg_cal_per_min:.2f} calories per minute."
    else:
        intensity = "Low"
        message = f"🚶 Light workout! You're burning {avg_cal_per_min:.2f} calories per minute."
    
    return {
        'intensity': intensity,
        'message': message,
        'cal_per_min': avg_cal_per_min
    }


def make_prediction(input_data: pd.DataFrame, expected_cols: list, model: Any, scaler: Any) -> float:
    """
    Make prediction using the ML model.
    
    Args:
        input_data: Input DataFrame
        expected_cols: Expected column names for the model
        model: Trained ML model
        scaler: Scaler object for feature scaling
    
    Returns:
        Predicted calorie value
    
    Raises:
        Exception: If prediction fails
    """
    try:
        # Apply one-hot encoding if needed
        input_processed = pd.get_dummies(input_data)
        
        # Align columns with expected columns
        input_processed = input_processed.reindex(
            columns=expected_cols,
            fill_value=0
        )
        
        # Scale the input
        scaled_input = scaler.transform(input_processed)
        
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        logger.info(f"Prediction made: {prediction} calories")
        
        return prediction
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise


# ==================== PAGE SETUP ====================

st.set_page_config(
    page_title="Calorie Burn Predictor",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

apply_custom_styling()

# ==================== MAIN CONTENT ====================

st.title("🏃 Calorie Burn Prediction System")

st.markdown("""
Predict calories burned based on your workout metrics using a trained machine learning model. 
Enter your personal data and workout information below to get started.
""")

st.divider()

# Load ML resources
expected_cols, model, scaler = load_ml_resources()

# ==================== USER INPUT SECTION ====================

st.subheader("📋 Enter Your Workout Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider(
        "Age (years)",
        min_value=INPUT_RANGES['age'][0],
        max_value=INPUT_RANGES['age'][1],
        value=25,
        help="Your age in years"
    )
    
    height = st.number_input(
        "Height (cm)",
        min_value=INPUT_RANGES['height'][0],
        max_value=INPUT_RANGES['height'][1],
        value=170,
        help="Your height in centimeters"
    )
    
    weight = st.number_input(
        "Weight (kg)",
        min_value=INPUT_RANGES['weight'][0],
        max_value=INPUT_RANGES['weight'][1],
        value=70,
        help="Your weight in kilograms"
    )

with col2:
    duration = st.slider(
        "Workout Duration (minutes)",
        min_value=INPUT_RANGES['duration'][0],
        max_value=INPUT_RANGES['duration'][1],
        value=30,
        help="Total duration of your workout session"
    )
    
    heart_rate = st.slider(
        "Average Heart Rate (bpm)",
        min_value=INPUT_RANGES['heart_rate'][0],
        max_value=INPUT_RANGES['heart_rate'][1],
        value=100,
        help="Your average heart rate during the workout"
    )
    
    body_temp = st.slider(
        "Body Temperature (°C)",
        min_value=INPUT_RANGES['body_temp'][0],
        max_value=INPUT_RANGES['body_temp'][1],
        value=37.0,
        step=0.1,
        help="Your body temperature during the workout"
    )

sex = st.radio(
    "Sex",
    ["Female", "Male"],
    horizontal=True,
    help="Your biological sex"
)

sex_male = 1 if sex == "Male" else 0

st.divider()

# ==================== PREDICTION SECTION ====================

if st.button("🔥 Predict Calories Burned", use_container_width=True):
    
    try:
        # Create input data
        user_inputs = {
            'age': age,
            'height': height,
            'weight': weight,
            'duration': duration,
            'heart_rate': heart_rate,
            'body_temp': body_temp,
            'sex_male': sex_male
        }
        
        input_data = create_input_dataframe(user_inputs)
        
        # Make prediction
        prediction = make_prediction(input_data, expected_cols, model, scaler)
        
        st.divider()
        st.markdown("### 📊 Prediction Results")
        
        # Display main metrics
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.metric(
                label="Calories Burned",
                value=f"{round(prediction, 2)} kcal",
                delta=None
            )
        
        with col4:
            bmi = calculate_bmi(weight, height)
            st.metric(
                label="BMI",
                value=f"{bmi}",
                delta=get_bmi_category(bmi)
            )
        
        with col5:
            cal_per_min = prediction / duration if duration > 0 else 0
            st.metric(
                label="Cal/Min",
                value=f"{cal_per_min:.2f}",
                delta=None
            )
        
        # Get detailed feedback
        feedback = get_workout_intensity_feedback(prediction, duration, heart_rate)
        
        st.markdown(f"""
        <div class="info-box">
        <strong>Workout Intensity: {feedback['intensity']}</strong><br>
        {feedback['message']}
        </div>
        """, unsafe_allow_html=True)
        
        # Conditional warnings
        if prediction > CALORIE_THRESHOLDS['high']:
            st.warning("⚠️ High calorie burn! Make sure to stay hydrated and take adequate rest.")
        elif prediction < CALORIE_THRESHOLDS['moderate']:
            st.info("💡 Light intensity workout. Consider increasing duration or intensity for better results.")
        
        # Display input details
        with st.expander("📋 View Prediction Inputs"):
            st.write("**Your Input Data:**")
            display_data = pd.DataFrame({
                'Metric': ['Age', 'Height (cm)', 'Weight (kg)', 'Duration (min)', 'Heart Rate (bpm)', 'Body Temp (°C)', 'Sex'],
                'Value': [age, height, weight, duration, heart_rate, body_temp, sex]
            })
            st.dataframe(display_data, use_container_width=True, hide_index=True)
        
        st.success("✅ Prediction completed successfully!")
        logger.info(f"Prediction completed for user: Age={age}, Weight={weight}, Duration={duration}")
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        st.error(f"❌ Prediction failed: {e}\n\nPlease check your inputs and try again.")

st.divider()

# Footer with additional information
st.markdown("""
---
### 📖 How This Works
- **Input Data**: Enter your personal metrics and workout details
- **Model**: Uses a trained Linear Regression model on historical workout data
- **Processing**: Features are scaled and processed to match training data
- **Output**: Predicts total calories burned during your workout

### ⚠️ Disclaimer
This prediction is based on a machine learning model trained on historical data. Actual calorie burn may vary based on individual metabolism, fitness level, and other factors.

---
**ML Model | Streamlit Deployment | AI Engineer Portfolio Project**
""")
   
