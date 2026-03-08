import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error

# ---------------------- App Title ----------------------
st.set_page_config(page_title="AI Smart ML Tool", layout="wide")
st.title("🚀 AI Smart Machine Learning Tool")
st.write("Upload → Auto Clean → Detect Task → Train → Visualize → Download Predictions")

# ---------------------- File Upload ----------------------
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    st.subheader("📌 Dataset Preview")
    st.dataframe(data.head())
    
    # ---------------------- Column Selection ----------------------
    if data.shape[1] < 2:
        st.warning("Dataset must contain at least 2 columns: 1 feature + 1 target")
    else:
        st.sidebar.subheader("Select Columns")
        all_columns = data.columns.tolist()
        target_column = st.sidebar.selectbox("Select Target Column", all_columns)
        feature_columns = st.sidebar.multiselect(
            "Select Feature Columns", 
            [col for col in all_columns if col != target_column],
            default=[col for col in all_columns if col != target_column]
        )
        
        # ---------------------- Outlier & Preprocessing ----------------------
        remove_outliers = st.sidebar.checkbox("Remove Outliers (Numeric Columns)", value=False)
        df = data.copy()
        
        # Encoding categorical columns
        le = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = le.fit_transform(df[col])
        
        # Removing outliers
        if remove_outliers:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[~((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR))]

        # ---------------------- Task Detection ----------------------
        if df[target_column].nunique() <= 20 and df[target_column].dtype in [np.int64, np.int32]:
            task_type = "Classification"
        else:
            task_type = "Regression"
        st.sidebar.info(f"Detected Task: {task_type}")

        # ---------------------- Train/Test Split ----------------------
        test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 20)
        X = df[feature_columns]
        y = df[target_column]

        # Scaling numeric features
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=feature_columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        # ---------------------- Model Training ----------------------
        if task_type == "Classification":
            model = RandomForestClassifier()
        else:
            model = RandomForestRegressor()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ---------------------- Model Evaluation ----------------------
        st.subheader("📊 Model Evaluation")

        if task_type == "Classification":
            if len(np.unique(y_test)) < 2:
                st.warning("Not enough classes in test set to calculate accuracy or confusion matrix.")
            else:
                acc = accuracy_score(y_test, y_pred)
                st.write(f"*Accuracy:* {acc:.4f}")
                st.write("*Confusion Matrix:*")
                cm = confusion_matrix(y_test, y_pred)
                st.write(cm)
                plt.figure(figsize=(5,4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                st.pyplot(plt)
        else:  # Regression
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            st.write(f"*R² Score:* {r2:.4f}")
            st.write(f"*RMSE:* {rmse:.4f}")

        # ---------------------- Visualization ----------------------
        st.subheader("📈 Visualizations")
        viz_col = st.sidebar.selectbox("Select Column for Visualization", df.columns)
        chart_type = st.sidebar.selectbox("Select Chart Type", ["Histogram", "Boxplot", "Countplot", "Scatter", "Correlation Heatmap"])

        if chart_type == "Histogram":
            plt.figure()
            sns.histplot(df[viz_col], kde=True)
            st.pyplot(plt)
        elif chart_type == "Boxplot":
            plt.figure()
            sns.boxplot(x=df[viz_col])
            st.pyplot(plt)
        elif chart_type == "Countplot":
            plt.figure()
            sns.countplot(x=df[viz_col])
            st.pyplot(plt)
        elif chart_type == "Scatter":
            scatter_col = st.sidebar.selectbox("Select Second Column for Scatter Plot", [c for c in df.columns if c != viz_col])
            plt.figure()
            sns.scatterplot(x=df[viz_col], y=df[scatter_col])
            st.pyplot(plt)
        elif chart_type == "Correlation Heatmap":
            plt.figure(figsize=(8,6))
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
            st.pyplot(plt)

        # ---------------------- Download Predictions ----------------------
        st.subheader("💾 Download Predictions")
        output_df = X_test.copy()
        output_df[target_column + "_predicted"] = y_pred
        csv = output_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "predictions.csv", "text/csv")

else:
    st.info("Please upload a CSV file to get started!")