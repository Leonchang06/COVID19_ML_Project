from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =======================================
# Page configuration
# =======================================
st.set_page_config(
    page_title="COVID-19 ML Prediction",
    page_icon="🩺",
    layout="centered"
)


# =======================================
# Load all trained models
# =======================================
MODEL_PATH = (
    Path(__file__).parent
    / "all_models_bundle.pkl"
)


@st.cache_resource
def load_model_bundle():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "all_models_bundle.pkl was not found."
        )

    return joblib.load(MODEL_PATH)


try:
    bundle = load_model_bundle()

    preprocessor = bundle["preprocessor"]
    models = bundle["models"]
    metrics = bundle["metrics"]
    best_model_name = bundle["best_model"]

except Exception as error:
    st.error(
        "The trained models could not be loaded."
    )
    st.exception(error)
    st.stop()


# =======================================
# Prediction function
# =======================================
def generate_prediction(
    model_name,
    model,
    encoded_input
):
    prediction = int(
        model.predict(encoded_input)[0]
    )

    result = {
        "Model": model_name,
        "Prediction": (
            "Positive"
            if prediction == 1
            else "Negative"
        ),
        "Positive Probability": "Not available",
        "Decision Score": "Not available"
    }

    # Logistic Regression and Random Forest
    if hasattr(model, "predict_proba"):

        probability = float(
            model.predict_proba(
                encoded_input
            )[0][1]
        )

        result["Positive Probability"] = (
            f"{probability * 100:.2f}%"
        )

    # Linear SVM
    elif hasattr(model, "decision_function"):

        decision_score = float(
            model.decision_function(
                encoded_input
            )[0]
        )

        result["Decision Score"] = (
            f"{decision_score:.4f}"
        )

    return result


# =======================================
# Page heading
# =======================================
st.title(
    "COVID-19 Test Result Prediction System"
)

st.write(
    "This supervised machine learning system "
    "compares three classification algorithms "
    "developed by the group members."
)

st.warning(
    "This is an educational machine learning "
    "prototype. It is not a medical diagnosis "
    "and must not replace professional medical "
    "advice or laboratory testing."
)


# =======================================
# Model information
# =======================================
st.info(
    f"Best-performing model: "
    f"{best_model_name}"
)

with st.expander(
    "View model performance comparison"
):

    metrics_table = []

    for model_name, model_metrics in metrics.items():

        metrics_table.append({
            "Model": model_name,
            "Accuracy": (
                f"{model_metrics['Accuracy'] * 100:.2f}%"
            ),
            "Precision": (
                f"{model_metrics['Precision'] * 100:.2f}%"
            ),
            "Recall": (
                f"{model_metrics['Recall'] * 100:.2f}%"
            ),
            "F1-score": (
                f"{model_metrics['F1-score'] * 100:.2f}%"
            ),
            "Training Time": (
                f"{model_metrics['Training Time']:.2f}s"
            )
        })

    metrics_df = pd.DataFrame(
        metrics_table
    )

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Random Forest is recommended because "
        "it achieved the highest accuracy, "
        "precision, recall, and F1-score."
    )


# =======================================
# Input form
# =======================================
with st.form("prediction_form"):

    st.subheader("Select Classification Model")

    model_options = (
        list(models.keys())
        + ["Compare All Models"]
    )

    selected_model = st.selectbox(
        "Classification method",
        model_options,
        format_func=lambda model_name: (
            f"{model_name} (Recommended)"
            if model_name == best_model_name
            else model_name
        )
    )

    st.subheader("Symptoms")

    left_column, right_column = st.columns(2)

    with left_column:

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

    with right_column:

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
        "Generate Prediction",
        use_container_width=True
    )


# =======================================
# Process prediction
# =======================================
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

    try:
        encoded_input = (
            preprocessor.transform(
                input_data
            )
        )

        st.divider()
        st.subheader("Prediction Result")

        # Compare all three models
        if selected_model == "Compare All Models":

            prediction_results = []

            for model_name, model in models.items():

                result = generate_prediction(
                    model_name,
                    model,
                    encoded_input
                )

                result["Recommended"] = (
                    "Yes"
                    if model_name == best_model_name
                    else "No"
                )

                prediction_results.append(
                    result
                )

            results_df = pd.DataFrame(
                prediction_results
            )

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )

            st.info(
                f"The recommended result should "
                f"be interpreted using "
                f"{best_model_name}, which achieved "
                f"the best evaluation performance."
            )

        # Use one selected model
        else:

            selected_classifier = models[
                selected_model
            ]

            result = generate_prediction(
                selected_model,
                selected_classifier,
                encoded_input
            )

            st.write(
                f"Selected model: "
                f"**{selected_model}**"
            )

            if result["Prediction"] == "Positive":
                st.error(
                    "Model classification: Positive"
                )
            else:
                st.success(
                    "Model classification: Negative"
                )

            if (
                result["Positive Probability"]
                != "Not available"
            ):
                st.metric(
                    "Estimated positive probability",
                    result["Positive Probability"]
                )

            if (
                result["Decision Score"]
                != "Not available"
            ):
                st.metric(
                    "Linear SVM decision score",
                    result["Decision Score"]
                )

                st.caption(
                    "A positive decision score supports "
                    "the Positive class, while a negative "
                    "score supports the Negative class. "
                    "It is not a probability."
                )

        with st.expander(
            "View submitted information"
        ):
            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

    except Exception as error:

        st.error(
            "An error occurred while generating "
            "the prediction."
        )

        st.exception(error)


# =======================================
# Footer
# =======================================
st.divider()

st.caption(
    "Developed for a supervised machine "
    "learning assignment."
)