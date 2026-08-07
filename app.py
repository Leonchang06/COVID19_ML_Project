import joblib
import pandas as pd
import streamlit as st


# Page configuration
st.set_page_config(
    page_title="COVID-19 Prediction",
    page_icon="🩺",
    layout="centered"
)


# Load the trained model and preprocessor
@st.cache_resource
def load_model_bundle():
    bundle = joblib.load(
        "best_model_bundle.pkl"
    )

    return (
        bundle["model"],
        bundle["preprocessor"]
    )


model, preprocessor = load_model_bundle()


# Page title
st.title(
    "COVID-19 Test Result Prediction"
)

st.caption(
    "Prediction model: Random Forest"
)

st.write(
    "Enter the patient's basic information "
    "and symptoms to generate a prediction."
)

st.warning(
    "This application is an educational "
    "machine learning prototype and must not "
    "be used as a medical diagnosis."
)


# Input form
with st.form("prediction_form"):

    st.subheader("Symptoms")

    cough = st.selectbox(
        "Cough",
        ["No", "Yes"]
    )

    fever = st.selectbox(
        "Fever",
        ["No", "Yes"]
    )

    sore_throat = st.selectbox(
        "Sore throat",
        ["No", "Yes"]
    )

    shortness_of_breath = st.selectbox(
        "Shortness of breath",
        ["No", "Yes"]
    )

    head_ache = st.selectbox(
        "Headache",
        ["No", "Yes"]
    )

    st.subheader("Basic Information")

    age_60_and_above = st.selectbox(
        "Age 60 and above",
        ["No", "Yes", "Unknown"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male", "Unknown"]
    )

    test_indication = st.selectbox(
        "Reason for COVID-19 testing",
        [
            "Other",
            "Abroad",
            "Contact with confirmed"
        ]
    )

    predict_button = st.form_submit_button(
        "Predict Test Result"
    )


# Run prediction after button is clicked
if predict_button:

    yes_no_mapping = {
        "No": 0,
        "Yes": 1
    }

    input_data = pd.DataFrame({
        "cough": [
            yes_no_mapping[cough]
        ],
        "fever": [
            yes_no_mapping[fever]
        ],
        "sore_throat": [
            yes_no_mapping[sore_throat]
        ],
        "shortness_of_breath": [
            yes_no_mapping[
                shortness_of_breath
            ]
        ],
        "head_ache": [
            yes_no_mapping[head_ache]
        ],
        "age_60_and_above": [
            age_60_and_above.lower()
        ],
        "gender": [
            gender.lower()
        ],
        "test_indication": [
            test_indication.lower()
        ]
    })

    encoded_input = preprocessor.transform(
        input_data
    )

    prediction = model.predict(
        encoded_input
    )[0]

    positive_probability = (
        model.predict_proba(encoded_input)[0][1]
    )

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            "Predicted result: COVID-19 Positive"
        )
    else:
        st.success(
            "Predicted result: COVID-19 Negative"
        )

    st.metric(
        "Estimated positive probability",
        f"{positive_probability * 100:.2f}%"
    )

    with st.expander("View submitted information"):
        st.dataframe(
            input_data,
            use_container_width=True
        )