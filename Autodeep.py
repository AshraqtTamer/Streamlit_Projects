import streamlit as st
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Streamlit UI
st.title("📊 Auto Deep Learning with Streamlit")

# File Upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])


# Function for Data Cleaning (autoclean)
def autoclean(df):
    # Remove columns with more than 50% missing values
    missing_threshold = 0.5
    df = df.loc[:, df.isnull().mean() < missing_threshold]

    # Fill missing numerical values with the median of each column
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Fill missing categorical values with the mode (most frequent) of each column
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Any additional cleaning operations (e.g., removing outliers) can be added here
    return df


# Function for Data Preprocessing (preprocessdata)
def preprocessdata(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handling categorical data in target
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    # Convert categorical columns in features to dummy variables
    X = pd.get_dummies(X, drop_first=True)

    # Scaling features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset", df.head())

    # Selecting Target Variable
    target_column = st.selectbox("Select the target variable", df.columns)

    if st.button("Preprocess & Train Model"):
        # Clean the dataset
        df = autoclean(df)

        # Preprocess the data
        X, y, scaler = preprocessdata(df, target_column)

        # Splitting Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define Deep Learning Model
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            keras.layers.BatchNormalization(),  # Normalize activations
            keras.layers.Dropout(0.3),  # Prevent overfitting

            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),

            keras.layers.Dense(32, activation='relu'),

            keras.layers.Dense(1, activation='sigmoid')  # Binary classification
        ])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Train Model
        history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=1)

        # Model Evaluation
        st.write("### Model Performance")
        loss, acc = model.evaluate(X_test, y_test)
        st.write(f"🔹 Test Accuracy: {acc:.4f}")

        # Plot Training History
        fig, ax = plt.subplots()
        ax.plot(history.history['accuracy'], label='Training Accuracy')
        ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()
        st.pyplot(fig)

        # Save Model
        model.save("auto_dl_model.h5")
        st.success("Model trained and saved successfully! ✅")

        # Download Model
        with open("auto_dl_model.h5", "rb") as file:
            st.download_button("Download Trained Model", file, "auto_dl_model.h5")

# Prediction Section
st.write("## 🔍 Make Predictions")
new_data = st.file_uploader("Upload new data for prediction (CSV)", type=["csv"])

if new_data is not None:
    new_df = pd.read_csv(new_data)
    st.write("### Preview of New Data", new_df.head())

    # Clean and preprocess new data
    new_df = autoclean(new_df)
    new_df, _, _ = preprocessdata(new_df, target_column)

    # Load Model
    model = keras.models.load_model("auto_dl_model.h5")

    # Make Predictions
    predictions = model.predict(new_df)
    st.write("### Predictions", predictions)

