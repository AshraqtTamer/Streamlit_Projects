import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit App Title
st.title("🤖 Machine Learning App")
st.info('This is app builds a machine learning model!:  Regression, Classification, or Clustering')

# Sidebar for task selection
st.sidebar.title("Choose Task")
task = st.sidebar.selectbox("Select the task", ["Regression", "Classification", "Clustering"])

# File uploader
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV format) 📄", type="csv")

if uploaded_file:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    st.write("### Dataset Preview")
    st.write(df.head())

    st.write("### Dataset Information")
    st.write(df.describe())
    st.write("### Dataset Columns")
    st.write(df.columns.tolist())

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
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

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
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")



    elif task == "Clustering":

        num_clusters = st.sidebar.slider("Number of Clusters (for KMeans)", min_value=2, max_value=10, value=3)

        cluster = KMeans(n_clusters=num_clusters, random_state=42)

        cluster_labels = cluster.fit_predict(X)

        centroids = cluster.cluster_centers_  # Extract centroids

        st.write("### Clustering Results")

        # Map cluster labels to a string format like "Cluster0", "Cluster1", etc.
        cluster_labels_str = [f"Cluster {label}" for label in cluster_labels]

        st.write("Cluster Labels:", np.unique(cluster_labels_str))

        # Add cluster labels to the dataframe
        df['Cluster'] = cluster_labels

        st.write("Clustered Data Preview:")

        st.write(df.head())

        # Allow user to select X and Y axes for visualization

        x_axis = st.sidebar.selectbox("Select X-axis for scatter plot", X.columns)

        y_axis = st.sidebar.selectbox("Select Y-axis for scatter plot", X.columns)

        # Visualize Clusters with Centroids

        if x_axis and y_axis:
            st.write("### Cluster Visualization")

            plt.figure(figsize=(8, 6))

            # Scatter plot of data points
            sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=cluster_labels_str, palette="viridis", s=50, alpha=0.8)

            # Overlay centroids
            plt.scatter(
                centroids[:, X.columns.get_loc(x_axis)],  # X-coordinate of centroids
                centroids[:, X.columns.get_loc(y_axis)],  # Y-coordinate of centroids
                c='red',
                s=200,
                marker='X',
                label='Centroids'
            )

            plt.title("K-Means Clustering Visualization with Centroids")
            plt.xlabel(x_axis)
            plt.ylabel(y_axis)
            plt.legend(title="Clusters")

            st.pyplot(plt)

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Built with care and Streamlit magic ✨")
