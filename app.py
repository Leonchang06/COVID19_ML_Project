from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =======================================
# Application configuration
# =======================================
st.set_page_config(
    page_title="COVID-19 ML Prediction",
    page_icon="🩺",
    layout="centered",
)

MODEL_PATH = Path(__file__).parent / "all_models_bundle.pkl"

EXPECTED_MODELS = {
    "Logistic Regression",
    "Random Forest",
    "Linear SVM",
}

REQUIRED_METRICS = {
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "Training Time",
}


# =======================================
# Model loading and validation
# =======================================
@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "all_models_bundle.pkl was not found."
        )

    loaded_bundle = joblib.load(MODEL_PATH)
    required_keys = {
        "preprocessor",
        "models",
        "metrics",
    }
    missing_keys = required_keys - set(loaded_bundle)

    if missing_keys:
        raise KeyError(
            "Missing bundle components: "
            + ", ".join(sorted(missing_keys))
        )

    loaded_models = loaded_bundle["models"]
    loaded_metrics = loaded_bundle["metrics"]

    if not hasattr(loaded_bundle["preprocessor"], "transform"):
        raise TypeError("The saved preprocessor is invalid.")

    missing_models = EXPECTED_MODELS - set(loaded_models)

    if missing_models:
        raise KeyError(
            "Missing trained models: "
            + ", ".join(sorted(missing_models))
        )

    for model_name in EXPECTED_MODELS:
        if not hasattr(loaded_models[model_name], "predict"):
            raise TypeError(
                f"{model_name} does not support prediction."
            )

        if model_name not in loaded_metrics:
            raise KeyError(
                f"Metrics are missing for {model_name}."
            )

        missing_metrics = (
            REQUIRED_METRICS
            - set(loaded_metrics[model_name])
        )

        if missing_metrics:
            raise KeyError(
                f"Missing metrics for {model_name}: "
                + ", ".join(sorted(missing_metrics))
            )

    return loaded_bundle


try:
    bundle = load_model_bundle()
    preprocessor = bundle["preprocessor"]
    models = bundle["models"]
    metrics = bundle["metrics"]

except Exception as error:
    st.error("The required model data could not be loaded.")

    with st.expander("View technical details"):
        st.exception(error)

    st.stop()


# =======================================
# Prediction and output functions
# =======================================
def generate_prediction(
    model_name,
    model,
    encoded_input,
):
    raw_prediction = model.predict(encoded_input)

    if len(raw_prediction) != 1:
        raise ValueError(
            "The model returned an unexpected number "
            "of predictions."
        )

    prediction = int(raw_prediction[0])

    if prediction not in (0, 1):
        raise ValueError(
            f"{model_name} returned an invalid class."
        )

    result = {
        "Model": model_name,
        "Prediction": (
            "Positive"
            if prediction == 1
            else "Negative"
        ),
        "Positive Probability": "Not available",
        "Decision Score": "Not available",
    }

    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)

        if 1 not in classes:
            raise ValueError(
                f"{model_name} does not contain "
                "the Positive class."
            )

        positive_index = classes.index(1)
        probability = float(
            model.predict_proba(encoded_input)[0][
                positive_index
            ]
        )

        if not 0 <= probability <= 1:
            raise ValueError(
                f"{model_name} returned an invalid probability."
            )

        result["Positive Probability"] = (
            f"{probability * 100:.2f}%"
        )

    elif hasattr(model, "decision_function"):
        decision_score = float(
            model.decision_function(encoded_input)[0]
        )
        result["Decision Score"] = f"{decision_score:.4f}"

    return result


def build_input_data(
    cough,
    fever,
    sore_throat,
    shortness_of_breath,
    head_ache,
    age_60_and_above,
    gender,
    test_indication,
):
    yes_no_mapping = {
        "No": 0,
        "Yes": 1,
    }

    model_input = pd.DataFrame({
        "cough": [yes_no_mapping[cough]],
        "fever": [yes_no_mapping[fever]],
        "sore_throat": [yes_no_mapping[sore_throat]],
        "shortness_of_breath": [
            yes_no_mapping[shortness_of_breath]
        ],
        "head_ache": [yes_no_mapping[head_ache]],
        "age_60_and_above": [
            age_60_and_above.lower()
        ],
        "gender": [gender.lower()],
        "test_indication": [test_indication.lower()],
    })

    input_summary = pd.DataFrame([{
        "Cough": cough,
        "Fever": fever,
        "Sore Throat": sore_throat,
        "Shortness of Breath": shortness_of_breath,
        "Headache": head_ache,
        "Age 60 and Above": age_60_and_above,
        "Gender": gender,
        "Test Indication": test_indication,
    }])

    return model_input, input_summary


def validate_encoded_input(encoded_input):
    if encoded_input.shape[0] != 1:
        raise ValueError(
            "Exactly one input record is required."
        )

    if encoded_input.shape[1] == 0:
        raise ValueError(
            "No encoded features were generated."
        )

    feature_count = encoded_input.shape[1]

    for model_name, model in models.items():
        expected_count = getattr(
            model,
            "n_features_in_",
            feature_count,
        )

        if feature_count != expected_count:
            raise ValueError(
                f"Feature mismatch for {model_name}: "
                f"expected {expected_count}, "
                f"received {feature_count}."
            )


def combine_input_and_results(
    input_summary,
    results_df,
):
    repeated_input = pd.concat(
        [input_summary] * len(results_df),
        ignore_index=True,
    )

    return pd.concat(
        [repeated_input, results_df],
        axis=1,
    )


def build_metrics_tables():
    numeric_rows = []

    for model_name, model_metrics in metrics.items():
        numeric_rows.append({
            "Model": model_name,
            "Accuracy": model_metrics["Accuracy"] * 100,
            "Precision": model_metrics["Precision"] * 100,
            "Recall": model_metrics["Recall"] * 100,
            "F1-score": model_metrics["F1-score"] * 100,
            "Training Time": model_metrics["Training Time"],
        })

    numeric_table = pd.DataFrame(numeric_rows)
    display_table = numeric_table.copy()

    for metric_name in (
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
    ):
        display_table[metric_name] = (
            display_table[metric_name]
            .map(lambda value: f"{value:.2f}%")
        )

    display_table["Training Time"] = (
        display_table["Training Time"]
        .map(lambda value: f"{value:.2f}s")
    )

    return numeric_table, display_table


# =======================================
# User interface
# =======================================
st.title("COVID-19 Prediction System")

prediction_tab, performance_tab = st.tabs([
    "Prediction",
    "Model Performance",
])


with prediction_tab:
    with st.form("prediction_form"):
        st.subheader("Select Classification Model")

        model_options = (
            list(models.keys())
            + ["Compare All Models"]
        )

        selected_model = st.selectbox(
            "Classification method",
            model_options,
        )

        st.subheader("Symptoms")
        left_column, right_column = st.columns(2)

        with left_column:
            cough = st.selectbox(
                "Cough",
                ["No", "Yes"],
            )
            fever = st.selectbox(
                "Fever",
                ["No", "Yes"],
            )
            sore_throat = st.selectbox(
                "Sore throat",
                ["No", "Yes"],
            )

        with right_column:
            shortness_of_breath = st.selectbox(
                "Shortness of breath",
                ["No", "Yes"],
            )
            head_ache = st.selectbox(
                "Headache",
                ["No", "Yes"],
            )

        st.subheader("Basic Information")

        age_60_and_above = st.selectbox(
            "Age 60 and above",
            ["No", "Yes", "Unknown"],
        )
        gender = st.selectbox(
            "Gender",
            ["Female", "Male", "Unknown"],
        )
        test_indication = st.selectbox(
            "Reason for COVID-19 testing",
            [
                "Other",
                "Abroad",
                "Contact with confirmed",
            ],
        )

        predict_button = st.form_submit_button(
            "Generate Prediction",
            use_container_width=True,
        )

    if predict_button:
        model_input, input_summary = build_input_data(
            cough,
            fever,
            sore_throat,
            shortness_of_breath,
            head_ache,
            age_60_and_above,
            gender,
            test_indication,
        )

        try:
            encoded_input = preprocessor.transform(
                model_input
            )
            validate_encoded_input(encoded_input)

            st.divider()
            st.subheader("Prediction Result")

            if selected_model == "Compare All Models":
                prediction_results = []

                for model_name, model in models.items():
                    prediction_results.append(
                        generate_prediction(
                            model_name,
                            model,
                            encoded_input,
                        )
                    )

                results_df = pd.DataFrame(
                    prediction_results
                )

                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True,
                )

                prediction_counts = results_df[
                    "Prediction"
                ].value_counts()
                total_models = len(results_df)
                majority_prediction = (
                    prediction_counts.idxmax()
                )
                majority_count = int(
                    prediction_counts.max()
                )

                if majority_count == total_models:
                    st.success(
                        "Model agreement: All "
                        f"{total_models} models predicted "
                        f"{majority_prediction}."
                    )
                else:
                    st.info(
                        "Model agreement: "
                        f"{majority_count} of "
                        f"{total_models} models predicted "
                        f"{majority_prediction}."
                    )

            else:
                selected_classifier = models[selected_model]
                result = generate_prediction(
                    selected_model,
                    selected_classifier,
                    encoded_input,
                )
                results_df = pd.DataFrame([result])

                st.write(
                    "Selected model: "
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
                        result["Positive Probability"],
                    )

                if (
                    result["Decision Score"]
                    != "Not available"
                ):
                    st.metric(
                        "Linear SVM decision score",
                        result["Decision Score"],
                    )
                    st.caption(
                        "A positive score supports the "
                        "Positive class, while a negative "
                        "score supports the Negative class. "
                        "It is not a probability."
                    )

            download_table = combine_input_and_results(
                input_summary,
                results_df,
            )

            st.download_button(
                "Download Prediction Results (CSV)",
                data=download_table.to_csv(index=False),
                file_name="prediction_results.csv",
                mime="text/csv",
                use_container_width=True,
            )

            with st.expander("View Submitted Information"):
                st.dataframe(
                    input_summary,
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as error:
            st.error(
                "The prediction could not be generated."
            )

            with st.expander("View technical details"):
                st.exception(error)


with performance_tab:
    numeric_metrics, display_metrics = (
        build_metrics_tables()
    )

    st.dataframe(
        display_metrics,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Metric Comparison")

    chart_data = numeric_metrics.set_index("Model")[[
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
    ]]

    st.bar_chart(chart_data)

    st.download_button(
        "Download Model Performance (CSV)",
        data=numeric_metrics.to_csv(index=False),
        file_name="model_performance.csv",
        mime="text/csv",
        use_container_width=True,
    )
