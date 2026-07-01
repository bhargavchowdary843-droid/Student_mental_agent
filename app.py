import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# Load model files
model = joblib.load("student_wellness_model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.set_page_config(
    page_title="Student Wellness Agent",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Intelligent Student Wellness Agent")
st.write("Mental Health Assessment System")

# Input fields
gender = st.selectbox(
    "Choose your gender",
    encoders["Choose your gender"].classes_
)

age = st.number_input(
    "Age",
    min_value=15,
    max_value=50,
    value=20
)

course = st.selectbox(
    "Course",
    encoders["What is your course?"].classes_
)

year = st.selectbox(
    "Current Year of Study",
    encoders["Your current year of Study"].classes_
)

cgpa = st.selectbox(
    "CGPA",
    encoders["What is your CGPA?"].classes_
)

marital = st.selectbox(
    "Marital Status",
    encoders["Marital status"].classes_
)

anxiety = st.selectbox(
    "Do you have Anxiety?",
    encoders["Do you have Anxiety?"].classes_
)

panic = st.selectbox(
    "Do you have Panic attack?",
    encoders["Do you have Panic attack?"].classes_
)

specialist = st.selectbox(
    "Did you seek specialist treatment?",
    encoders["Did you seek any specialist for a treatment?"].classes_
)

if st.button("Analyze Mental Health"):

    data = pd.DataFrame([{
        "Choose your gender": gender,
        "Age": str(age),
        "What is your course?": course,
        "Your current year of Study": year,
        "What is your CGPA?": cgpa,
        "Marital status": marital,
        "Do you have Anxiety?": anxiety,
        "Do you have Panic attack?": panic,
        "Did you seek any specialist for a treatment?": specialist
    }])

    # Encode input
    for col in data.columns:
        data[col] = encoders[col].transform(data[col])

    prediction = model.predict(data)[0]
    result = target_encoder.inverse_transform([prediction])[0]

    st.subheader("Assessment Result")

    if result == "Yes":
        st.error("⚠️ Depression Risk Detected")

        st.subheader("Recommendations")
        st.write("""
        - Consult a mental health professional
        - Practice meditation
        - Maintain healthy sleep
        - Seek family support
        - Reduce academic stress
        """)

    else:
        st.success("✅ No Significant Depression Risk")

        st.subheader("Recommendations")
        st.write("""
        - Continue healthy habits
        - Exercise regularly
        - Maintain social activities
        - Sleep properly
        """)

    record = data.copy()
    record["Prediction"] = result
    record["Date"] = datetime.now()

    if os.path.exists("student_records.csv"):
        record.to_csv(
            "student_records.csv",
            mode="a",
            header=False,
            index=False
        )
    else:
        record.to_csv(
            "student_records.csv",
            index=False
        )

    st.success("Record Saved Successfully")

# Display records
if os.path.exists("student_records.csv"):
    st.subheader("Student Wellness Records")
    records = pd.read_csv("student_records.csv")
    st.dataframe(records)