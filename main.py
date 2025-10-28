import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit App Title
st.title("ML App: Regression, Classification, or Clustering")

# Sidebar for task selection
st.sidebar.title("Choose Task")
task = st.sidebar.selectbox("Select the task", ["Regression", "Classification", "Clustering"])

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())

    st.write("### Dataset Information")
    st.write(df.info())

    # Choose target variable
    target_column = None
    if task != "Clustering":
        target_column = st.sidebar.selectbox("Select the target column", df.columns)

    # Preprocessing: Handle missing values
    if st.checkbox("Drop rows with missing values"):
        df = df.dropna()
        st.write("Rows with missing values dropped.")

    # Separate features and target
    X = df.drop(columns=[target_column]) if target_column else df
    y = df[target_column] if target_column else None

    # Task-specific processing
    if task == "Regression":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = st.sidebar.selectbox("Select Model", ["Linear Regression", "Random Forest Regressor"])

        if model == "Linear Regression":
            reg = LinearRegression()
        elif model == "Random Forest Regressor":
            reg = RandomForestRegressor()

        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        st.write("### Regression Results")
        st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))

    elif task == "Classification":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Random Forest Classifier"])

        if model == "Logistic Regression":
            clf = LogisticRegression()
        elif model == "Random Forest Classifier":
            clf = RandomForestClassifier()

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        st.write("### Classification Results")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))

    elif task == "Clustering":
        num_clusters = st.sidebar.slider("Number of Clusters (for KMeans)", min_value=2, max_value=10, value=3)
        model = st.sidebar.selectbox("Select Model", ["KMeans"])

        if model == "KMeans":
            cluster = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = cluster.fit_predict(X)

        st.write("### Clustering Results")
        st.write("Cluster Labels:", np.unique(cluster_labels))

        # Visualize Clusters
        if X.shape[1] >= 2:
            st.write("### Cluster Visualization")
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=cluster_labels, palette="viridis")
            st.pyplot(plt)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed with ❤️ using Streamlit")
