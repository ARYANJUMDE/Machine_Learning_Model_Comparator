import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import plotly.graph_objects as go

st.set_page_config(page_title="ML Model Comparison", layout="wide")

st.title("🎓 Python Learning Exam Performance - ML Model Comparison")
st.markdown("Train and compare classification models on exam performance data")

# Load data - with upload option
st.header("📂 Data Upload")
uploaded_file = st.file_uploader(
    "Upload your dataset (CSV file)",
    type=["csv"],
    help="Required columns: age, hours_spent_learning_per_week, practice_problems_solved, "
         "prior_programming_experience, final_exam_score, passed_exam",
)
if uploaded_file is None:
    st.warning("Please upload a CSV file to continue.")
    st.info("Required columns: age, hours_spent_learning_per_week, practice_problems_solved,prior_programming_experience, final_exam_score, passed_exam")
    st.stop()
df = pd.read_csv(uploaded_file)
st.success("✅ Data loaded successfully!")

# --- DATA EXPLORATION & VISUALIZATION (from your project.py) ---
st.header("📊 Data Exploration & Visualization")

# Show basic info
with st.expander("📋 Dataset Info"):
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    
    st.write("First few rows:")
    st.dataframe(df.head())

# Drop unnecessary columns
df = df.drop(columns=["country", "student_id"])

# Visualizations (from your project.py)
viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    st.subheader("Age vs Hours Spent Learning Per Week")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=df["age"], y=df["hours_spent_learning_per_week"], ax=ax)
    plt.title("Age vs Hours Spent Learning Per Week")
    st.pyplot(fig)

with viz_col2:
    st.subheader("Practice Problems Solved Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df["practice_problems_solved"], ax=ax)
    plt.title("Practice Problems Solved Distribution")
    st.pyplot(fig)

viz_col3, viz_col4 = st.columns(2)

with viz_col3:
    st.subheader("Final Exam Score Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df["final_exam_score"], ax=ax)
    plt.title("Final Exam Score Distribution")
    st.pyplot(fig)

with viz_col4:
    st.subheader("Final Exam Score by Exam Passing Status")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x=df["passed_exam"], y=df["final_exam_score"], ax=ax)
    plt.title("Final Exam Score by Exam Passing Status")
    st.pyplot(fig)

viz_col5, viz_col6 = st.columns(2)

with viz_col5:
    st.subheader("Practice Problems vs Final Exam Score")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=df["practice_problems_solved"], y=df["final_exam_score"], ax=ax)
    plt.title("Practice Problems Solved vs Final Exam Score")
    st.pyplot(fig)

with viz_col6:
    st.subheader("Numeric Correlation Matrix")
    num = df.select_dtypes(include=['int64', 'float64'])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(num.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    plt.title('Numeric Correlation Matrix')
    st.pyplot(fig)

# --- DATA PREPROCESSING ---
st.header("🔧 Data Preprocessing")

df.dropna(inplace=True)
st.write(f"Rows after dropping NaN: {df.shape[0]}")

# Handle prior_programming_experience
df["prior_programming_experience"] = df["prior_programming_experience"].replace("Beginner", 0)
df["prior_programming_experience"] = df["prior_programming_experience"].replace("Intermediate", 1)
df["prior_programming_experience"] = df["prior_programming_experience"].replace("Advanced", 2)
df["prior_programming_experience"] = df["prior_programming_experience"].replace("None", np.nan)
df.dropna(subset=["prior_programming_experience"], inplace=True)

st.write("✅ Preprocessing complete")

# Prepare features and target
X = df.drop(columns=["passed_exam"])
y = df["passed_exam"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

st.write(f"✅ Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# --- MODEL TRAINING & EVALUATION ---
st.header("🤖 Model Training & Evaluation")

# Train all models
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
c = classification_report(y_test, y_pred)

model1 = DecisionTreeClassifier(random_state=42)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
ac1 = accuracy_score(y_test, y_pred1)
cm1 = confusion_matrix(y_test, y_pred1)
c1 = classification_report(y_test, y_pred1)

model2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
ac2 = accuracy_score(y_test, y_pred2)
cm2 = confusion_matrix(y_test, y_pred2)
c2 = classification_report(y_test, y_pred2)

st.success("✅ All models trained successfully!")

# --- METRICS TABLE (from your project.py) ---
st.subheader("📊 Detailed Metrics Comparison")

metrics = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
    'Accuracy': [accuracy_score(y_test, y_pred)*100,
                 accuracy_score(y_test, y_pred1)*100,
                 accuracy_score(y_test, y_pred2)*100],
    'Precision': [precision_score(y_test, y_pred)*100,
                  precision_score(y_test, y_pred1)*100,
                  precision_score(y_test, y_pred2)*100],
    'Recall': [recall_score(y_test, y_pred)*100,
               recall_score(y_test, y_pred1)*100,
               recall_score(y_test, y_pred2)*100],
    'F1-Score': [f1_score(y_test, y_pred)*100,
                 f1_score(y_test, y_pred1)*100,
                 f1_score(y_test, y_pred2)*100]
})

metrics['Combined_Score'] = (metrics['Accuracy'] + metrics['F1-Score']) / 2

st.dataframe(metrics.round(2), use_container_width=True)

# --- INDIVIDUAL MODEL RESULTS ---
st.header("📈 Individual Model Results")

tabs = st.tabs(["Logistic Regression", "Decision Tree", "Random Forest"])

with tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{ac*100:.2f}%")
        st.write("**Confusion Matrix:**")
        st.write(cm)
    with col2:
        st.write("**Classification Report:**")
        st.text(c)

with tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{ac1*100:.2f}%")
        st.write("**Confusion Matrix:**")
        st.write(cm1)
    with col2:
        st.write("**Classification Report:**")
        st.text(c1)

with tabs[2]:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{ac2*100:.2f}%")
        st.write("**Confusion Matrix:**")
        st.write(cm2)
    with col2:
        st.write("**Classification Report:**")
        st.text(c2)

# --- BEST MODEL SUMMARY (from your project.py) ---
st.header("🏆 Best Model Summary")

max_score = metrics['Combined_Score'].max()
best_models = metrics[metrics['Combined_Score'] == max_score]

if len(best_models) > 1:
    st.success("🎉 **MULTIPLE MODELS ACHIEVED PERFECT PERFORMANCE!** 🎉")
    st.write(f"**Best Models:** {', '.join(best_models['Model'].tolist())}")
    st.write(f"**Combined Score:** {max_score:.2f}%")
    st.write("**Final Ranking:**")
    st.dataframe(best_models[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].round(2), use_container_width=True)
    st.info("💡 **Recommendation:** You can use either model. Choose based on your needs:\n"
            "• **Decision Tree** → Faster & Fully Interpretable\n"
            "• **Random Forest** → More robust in real-world/new data")
else:
    best_model = best_models.iloc[0]
    st.success(f"🏆 **BEST MODEL: {best_model['Model'].upper()}**")
    st.write(f"**Accuracy:** {best_model['Accuracy']:.2f}% | **F1-Score:** {best_model['F1-Score']:.2f}%")

# --- VISUALIZATION COMPARISON ---
st.subheader("📊 Model Performance Comparison Chart")
fig = go.Figure(data=[
    go.Bar(name='Accuracy', x=metrics['Model'], y=metrics['Accuracy']),
    go.Bar(name='Precision', x=metrics['Model'], y=metrics['Precision']),
    go.Bar(name='Recall', x=metrics['Model'], y=metrics['Recall']),
    go.Bar(name='F1-Score', x=metrics['Model'], y=metrics['F1-Score'])
])
fig.update_layout(barmode='group', title='Model Metrics Comparison', yaxis_title='Score (%)')
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | [GitHub Repo](https://github.com/YOUR_USERNAME/Ai_Py_Pro)")
