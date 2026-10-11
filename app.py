
import streamlit as st
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Page configuration
st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide"
)

# Load saved files
@st.cache_resource
def load_model():
    with open("student_model.pkl", "rb") as file:
        return pickle.load(file)

@st.cache_data
def load_data():
    return pd.read_csv("student_data.csv")

try:
    model = load_model()
    df = load_data()
except (FileNotFoundError, ValueError, OSError) as e:
    st.error(f"Could not load project files: {e}")
    st.stop()

FEATURES = [
    "study_hours",
    "attendance",
    "previous_score",
    "practice_tests"
]

# Model evaluation on the same reproducible test split
X = df[FEATURES]
y = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Header
st.title("🎓 Student Performance Analytics")
st.caption(
    "Machine Learning • Score Prediction • Academic Insights"
)

# Sidebar
st.sidebar.header("Student Details")

study_hours = st.sidebar.slider(
    "Daily Study Hours", 1.0, 10.0, 5.0, 0.5
)

attendance = st.sidebar.slider(
    "Attendance (%)", 50, 100, 75
)

previous_score = st.sidebar.slider(
    "Previous Exam Score", 30, 100, 60
)

practice_tests = st.sidebar.slider(
    "Practice Tests Completed", 1, 10, 5
)

predict_clicked = st.sidebar.button(
    "Predict Performance", type="primary", use_container_width=True
)

# Overview metrics
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Records", len(df))
c2.metric("Average Score", f"{df['final_score'].mean():.1f}/100")
c3.metric("Mean Absolute Error", f"{mae:.2f}")
c4.metric("Test R² Score", f"{r2:.3f}")

st.divider()

prediction_tab, analytics_tab, model_tab = st.tabs(
    ["🔮 Prediction", "📊 Dataset Analytics", "🧠 Model Evaluation"]
)

# Prediction tab
with prediction_tab:
    st.subheader("Student Performance Prediction")
    st.write(
        "Enter the student's details in the sidebar and predict their score."
    )

    if predict_clicked:
        input_data = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance": attendance,
            "previous_score": previous_score,
            "practice_tests": practice_tests
        }])

        score = float(model.predict(input_data)[0])
        score = max(0.0, min(100.0, score))

        if score >= 80:
            category = "Excellent"
            message = "Maintain consistent study habits and practice."
        elif score >= 65:
            category = "Good"
            message = "Review weak topics and continue regular practice."
        elif score >= 50:
            category = "Average"
            message = "Try a consistent study schedule and revise regularly."
        else:
            category = "Needs Improvement"
            message = "Focus on foundational topics and seek guidance."

        left, right = st.columns([1, 2])

        with left:
            st.metric("Predicted Final Score", f"{score:.2f}/100")
            st.progress(int(round(score)))
            st.write(f"**Performance category:** {category}")

        with right:
            st.markdown("### 📌 Suggested Next Steps")

            suggestions = []

            if study_hours < 4:
                suggestions.append(
                    "Try setting aside more regular study time."
                )
            if attendance < 75:
                suggestions.append(
                    "Work towards more consistent class attendance."
                )
            if previous_score < 60:
                suggestions.append(
                    "Revise foundational topics from previous exams."
                )
            if practice_tests < 5:
                suggestions.append(
                    "Practise more sample questions and review mistakes."
                )

            if suggestions:
                for item in suggestions:
                    st.write(f"- {item}")
            else:
                st.write(message)

        st.info(
            "This is an estimate from a model trained on synthetic data, "
            "not an actual exam result or a guarantee of future performance."
        )
    else:
        st.info(
            "Adjust the student details in the sidebar, "
            "then click Predict Performance."
        )

# Dataset analytics tab
with analytics_tab:
    st.subheader("Explore the Student Dataset")

    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.markdown("**Final Score Distribution**")
        histogram = (
            df.assign(score_band=(df["final_score"] // 10) * 10)
            .groupby("score_band")
            .size()
            .rename("Students")
        )
        st.bar_chart(histogram)

    with chart_right:
        st.markdown("**Study Hours vs Final Score**")
        scatter_data = df[["study_hours", "final_score"]].copy()
        st.scatter_chart(
            scatter_data,
            x="study_hours",
            y="final_score"
        )

    st.markdown("**Dataset Summary**")
    st.dataframe(
        df[FEATURES + ["final_score"]].describe().round(2),
        use_container_width=True
    )

    with st.expander("Preview student records"):
        st.dataframe(df.head(20), use_container_width=True)

# Model evaluation tab
with model_tab:
    st.subheader("Linear Regression Model")

    m1, m2 = st.columns(2)
    m1.metric("MAE", f"{mae:.3f}")
    m2.metric("R² Score", f"{r2:.3f}")

    st.write(
        "**MAE:** Average absolute difference between actual and "
        "predicted scores on the test set."
    )
    st.write(
        "**R²:** Indicates how much of the variation in test scores "
        "is explained by the model. A value closer to 1 is generally better."
    )

    st.write("**Training setup**")
    st.write("- Algorithm: Linear Regression")
    st.write("- Total records:", len(df))
    st.write("- Training records:", len(X_train))
    st.write("- Testing records:", len(X_test))
    st.write("- Test split: 20%")

    st.warning(
        "The dataset is synthetically generated. These evaluation "
        "metrics measure performance on synthetic data and do not "
        "establish real-world accuracy."
    )
