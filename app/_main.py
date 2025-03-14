import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


def main():
    st.title("ML Pipeline Predictor")
    st.write("Upload your CSV file to get predictions from the pre-trained model")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Load the data
            data = pd.read_csv(uploaded_file)

            # Show the data
            st.subheader("Data Preview")
            st.dataframe(data.head())

            # Data info
            st.subheader("Data Information")
            buffer = io.StringIO()
            data.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

            # Check if model exists
            if os.path.exists("total_model.joblib"):
                # Load the model
                model = joblib.load("total_model.joblib")

                # Make predictions
                if st.button("Run Prediction"):
                    try:
                        # Get predictions
                        predictions = model.predict(data)

                        # Display predictions
                        st.subheader("Predictions")
                        result_df = pd.DataFrame(predictions, columns=["Prediction"])
                        st.dataframe(result_df)

                        # Download predictions
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv",
                        )

                        # If model has predict_proba method, show probabilities
                        if hasattr(model, "predict_proba"):
                            try:
                                probabilities = model.predict_proba(data)
                                st.subheader("Prediction Probabilities")

                                # For binary classification
                                if probabilities.shape[1] == 2:
                                    prob_df = pd.DataFrame(
                                        probabilities, columns=["Class 0", "Class 1"]
                                    )
                                # For multi-class
                                else:
                                    prob_df = pd.DataFrame(
                                        probabilities,
                                        columns=[
                                            f"Class {i}"
                                            for i in range(probabilities.shape[1])
                                        ],
                                    )
                                st.dataframe(prob_df)
                            except Exception as e:
                                st.warning(f"Could not generate probabilities: {e}")

                    except Exception as e:
                        st.error(f"Error making predictions: {e}")
                        st.info(
                            "Make sure your CSV has the correct format expected by the model"
                        )
            else:
                st.error(
                    "Model file 'total_model.joblib' not found. Please make sure the model file is in the same directory as this script."
                )

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Please make sure you've uploaded a valid CSV file")


if __name__ == "__main__":
    import io

    main()
