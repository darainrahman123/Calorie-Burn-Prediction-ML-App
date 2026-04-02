import streamlit as st 
import joblib 
import pandas as pd

expected_cols=joblib.load('cols.pkl')
model=joblib.load('LR.pkl')
scaler=joblib.load('scaler.pkl')

st.set_page_config(
    page_title="Calorie Burn Predictor",
    page_icon="🔥",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>

.main {
    padding: 2rem;
}

.stButton>button {
    width:100%;
    height:55px;
    font-size:18px;
    border-radius:10px;
    background-color:#ff4b4b;
    color:white;
}

.metric-card {
    background-color:#f0f2f6;
    padding:20px;
    border-radius:12px;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)


st.title("🏃 Calorie Burn Prediction System")

st.markdown(
"""
Predict calories burned based on workout metrics using a trained ML model.
"""
)

st.divider()

# Layout
col1, col2 = st.columns(2)

with col1:

    age = st.slider("Age",10,80,25)

    height = st.number_input(
        "Height (cm)",
        min_value=120,
        max_value=220,
        value=170
    )

    weight = st.number_input(
        "Weight (kg)",
        min_value=30,
        max_value=150,
        value=70
    )

with col2:

    duration = st.slider(
        "Workout Duration (min)",
        1,180,30
    )

    heart_rate = st.slider(
        "Heart Rate",
        60,200,100
    )

    body_temp = st.slider(
        "Body Temperature (°C)",
        35.0,42.0,37.0
    )

sex = st.radio(
    "Sex",
    ["Female","Male"],
    horizontal=True
)

sex_male = 1 if sex=="Male" else 0

st.divider()

# Create dataframe
input_data = pd.DataFrame({

    'Age':[age],
    'Height':[height],
    'Weight':[weight],
    'Duration':[duration],
    'Heart_Rate':[heart_rate],
    'Body_Temp':[body_temp],
    'Sex_male':[sex_male]

})

# Prediction button
if st.button("Predict Calories Burned 🔥"):

    input_data=pd.get_dummies(input_data)

    input_data=input_data.reindex(
        columns=expected_cols,
        fill_value=0
    )

    scaled_input=scaler.transform(input_data)

    prediction=model.predict(scaled_input)[0]

    st.divider()

    st.markdown("### Prediction Result")

    col3, col4 = st.columns(2)
    with col3:
        st.metric(
            label="Estimated Calories Burned",
            value=f"{round(prediction,2)} kcal"
        )

    with col4:
        bmi = weight/((height/100)**2)
        st.metric("BMI📊",round(bmi,2))
    
    if prediction > 300:

        st.warning("⚠️ High calorie burn workout")

    else:

        st.success("✅ Moderate workout intensity")


    with st.expander("View Model Input"):

        st.dataframe(input_data)


st.divider()

st.caption("ML Model | Streamlit Deployment | AI Engineer Portfolio Project")