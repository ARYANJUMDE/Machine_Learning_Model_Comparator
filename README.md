# 🎓 Python Learning Exam Performance Predictor

**Predict if students will pass their Python exam using Machine Learning!**  
An interactive Streamlit web app that lets you upload your dataset, explore data, train models, and compare results in real-time. Perfect for beginners learning ML or educators analyzing student performance.
---

### Project Description
This project helps predict whether students will **pass (1)** or **fail (0)** their Python learning exam based on factors like age, study hours, practice problems, and prior experience.  

**Problem Statement:**  
Many students struggle with programming exams — this app identifies at-risk learners early so teachers can provide targeted support.

**Key Objectives:**
- Interactive data exploration and visualization
- Train and compare popular classification models
- Achieve high accuracy for real-world use
- Make ML accessible via a user-friendly web app

---

### Features
- **Upload & Explore:** Upload your CSV dataset and get instant EDA with charts (scatter plots, histograms, heatmaps)
- **Preprocess Automatically:** Handles missing values and categorical encoding
- **Train Multiple Models:** Compare Logistic Regression, Decision Tree, and Random Forest side-by-side
- **Detailed Metrics:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Classification Report
- **Visual Comparisons:** Interactive bar charts and tabs for easy model evaluation
- **Best Model Recommendation:** Automatically picks the winner based on combined scores
- **User-Friendly:** Built with Streamlit — no coding required to run!

What makes this unique: It's a **full ML pipeline in one app**, great for portfolios and quick demos!

---

### Dataset Information
**Source:** Your uploaded CSV file (synthetic or real student data)  
**Samples:** Variable (based on upload; rows used here 3000)  
**Target Variable:** `passed_exam` (1 = Passed, 0 = Failed)  

| Column                          | Type        | Description                              | Example Values          |
|---------------------------------|-------------|------------------------------------------|-------------------------|
| age                             | int         | Student's age                            | 18–25                   |
| hours_spent_learning_per_week   | int         | Weekly study hours                       | 5–40                    |
| practice_problems_solved        | int         | Number of practice problems completed    | 10–200                  |
| prior_programming_experience    | categorical | Experience level                         | None/Beginner/Intermediate/Advanced |
| final_exam_score                | int         | Final exam score (%)                     | 40–95                   |
| **passed_exam**                 | binary      | Exam result                              | 0 / 1                   |

**Preprocessing Steps:**
- Drop unnecessary columns (e.g., `country`, `student_id`)
- Handle missing values (drop rows)
- Encode categorical `prior_programming_experience` (None → NaN, Beginner → 0, etc.)

---

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Ai_Py_Pro.git
cd Ai_Py_Pro

# 2. Install dependencies
pip install -r requirements.txt
