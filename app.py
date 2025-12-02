import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="🎓 Dean Dilemma – Placement Prediction",
    layout="centered"
)
st.title("🎓 Dean Dilemma – Placement Prediction App")

st.write(
    "Choose a model and enter student details to predict whether the "
    "student is likely to be **placed**."
)

# -------------------------------------------------------------------
# Model files
# -------------------------------------------------------------------
LOG_MODEL_FILE = "logistic_model.pkl"              # Logistic Regression
RF_MODEL_FILE = "random_forest_model.pkl"      # Random Forest

# Features for each model
LOG_FEATURES = ["Percentile_ET"]

RF_FEATURES = [
    "Percent_SSC",
    "Percent_HSC",
    "Percent_Degree",
    "Experience_Yrs",
    "S-TEST",
    "Percentile_ET",
    "Gender_M",
    "Board_SSC_ICSE",
    "Board_SSC_Others",
    "Board_HSC_ISC",
    "Board_HSC_Others",
    "Stream_HSC_Commerce",
    "Stream_HSC_Science",
    "Course_Degree_Commerce",
    "Course_Degree_Computer Applications",
    "Course_Degree_Engineering",
    "Course_Degree_Management",
    "Course_Degree_Others",
    "Course_Degree_Science",
    "Entrance_Test_G-MAT",
    "Entrance_Test_G-SAT",
    "Entrance_Test_GCET",
    "Entrance_Test_K-MAT",
    "Entrance_Test_MAT",
    "Entrance_Test_MGT",
    "Entrance_Test_PGCET",
    "Entrance_Test_XAT"
]

# -------------------------------------------------------------------
# Model selection
# -------------------------------------------------------------------
st.subheader("Select Model")
model_choice = st.radio(
    "Which model do you want to use?",
    ["Logistic Regression (only Percentile_ET)",
     "Random Forest (full profile)"]
)

if "Logistic Regression" in model_choice:
    model_file = LOG_MODEL_FILE
    active_features = LOG_FEATURES
else:
    model_file = RF_MODEL_FILE
    active_features = RF_FEATURES

# Load model
model = None
if os.path.exists(model_file):
    try:
        model = joblib.load(model_file)
        st.success(f"✅ {model_choice.split('(')[0].strip()} model loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
else:
    st.error(f"❌ Model file '{model_file}' not found")

st.markdown("---")

# -------------------------------------------------------------------
# USER INPUTS
# -------------------------------------------------------------------
if "Logistic Regression" in model_choice:
    # ---------- Logistic Regression Inputs (only Percentile_ET) ----------
    st.header("Enter Student Details (Logistic Regression)")

    percentile_et = st.number_input(
        "Entrance Test Percentile (Percentile_ET)",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=0.1
    )

    input_df = pd.DataFrame([{"Percentile_ET": percentile_et}])

else:
    # ---------- Random Forest Inputs (all features) ----------
    st.header("Enter Student Details (Random Forest)")

    col1, col2 = st.columns(2)

    with col1:
        percent_ssc = st.number_input(
            "Percent SSC (%)",
            min_value=30.0, max_value=100.0, value=70.0, step=0.1
        )
        percent_hsc = st.number_input(
            "Percent HSC (%)",
            min_value=30.0, max_value=100.0, value=70.0, step=0.1
        )
        percent_degree = st.number_input(
            "Percent Degree (%)",
            min_value=30.0, max_value=100.0, value=65.0, step=0.1
        )
        experience_yrs = st.number_input(
            "Work Experience (years)",
            min_value=0.0, max_value=10.0, value=0.0, step=0.5
        )

    with col2:
        percentile_et = st.number_input(
            "Entrance Test Percentile (Percentile_ET)",
            min_value=0.0, max_value=100.0, value=60.0, step=0.1
        )
        gender = st.radio("Gender", ["Male", "Female"])
        ssc_board = st.selectbox("SSC Board", ["CBSE", "ICSE", "Others"])
        hsc_board = st.selectbox("HSC Board", ["State/Other", "ISC", "Others"])
        hsc_stream = st.selectbox("HSC Stream", ["Others", "Commerce", "Science"])

    # Degree
    degree_course = st.selectbox(
        "Undergraduate Degree",
        ["Others", "Commerce", "Computer Applications", "Engineering",
         "Management", "Science"]
    )

    # Entrance test
    entrance_test = st.selectbox(
        "Entrance Test Appeared",
        ["None", "G-MAT", "G-SAT", "GCET", "K-MAT", "MAT", "MGT",
         "PGCET", "XAT"]
    )

    # ----- Build the feature dict for RF -----
    rf_input = {feat: 0 for feat in RF_FEATURES}  # default 0

    # Numeric features
    rf_input["Percent_SSC"] = percent_ssc
    rf_input["Percent_HSC"] = percent_hsc
    rf_input["Percent_Degree"] = percent_degree
    rf_input["Experience_Yrs"] = experience_yrs
    rf_input["Percentile_ET"] = percentile_et

    # Gender_M: 1 if Male else 0
    rf_input["Gender_M"] = 1 if gender == "Male" else 0

    # S-TEST: 1 if any entrance test taken, else 0
    rf_input["S-TEST"] = 0 if entrance_test == "None" else 1

    # SSC Board dummies
    rf_input["Board_SSC_ICSE"] = 1 if ssc_board == "ICSE" else 0
    rf_input["Board_SSC_Others"] = 1 if ssc_board == "Others" else 0

    # HSC Board dummies
    rf_input["Board_HSC_ISC"] = 1 if hsc_board == "ISC" else 0
    rf_input["Board_HSC_Others"] = 1 if hsc_board == "Others" else 0

    # HSC Stream dummies
    rf_input["Stream_HSC_Commerce"] = 1 if hsc_stream == "Commerce" else 0
    rf_input["Stream_HSC_Science"] = 1 if hsc_stream == "Science" else 0

    # Degree dummies
    rf_input["Course_Degree_Commerce"] = 1 if degree_course == "Commerce" else 0
    rf_input["Course_Degree_Computer Applications"] = 1 if degree_course == "Computer Applications" else 0
    rf_input["Course_Degree_Engineering"] = 1 if degree_course == "Engineering" else 0
    rf_input["Course_Degree_Management"] = 1 if degree_course == "Management" else 0
    rf_input["Course_Degree_Science"] = 1 if degree_course == "Science" else 0
    rf_input["Course_Degree_Others"] = 1 if degree_course == "Others" else 0

    # Entrance test dummies
    rf_input["Entrance_Test_G-MAT"]  = 1 if entrance_test == "G-MAT"  else 0
    rf_input["Entrance_Test_G-SAT"]  = 1 if entrance_test == "G-SAT"  else 0
    rf_input["Entrance_Test_GCET"]   = 1 if entrance_test == "GCET"   else 0
    rf_input["Entrance_Test_K-MAT"]  = 1 if entrance_test == "K-MAT"  else 0
    rf_input["Entrance_Test_MAT"]    = 1 if entrance_test == "MAT"    else 0
    rf_input["Entrance_Test_MGT"]    = 1 if entrance_test == "MGT"    else 0
    rf_input["Entrance_Test_PGCET"]  = 1 if entrance_test == "PGCET"  else 0
    rf_input["Entrance_Test_XAT"]    = 1 if entrance_test == "XAT"    else 0

    input_df = pd.DataFrame([rf_input])[active_features]

st.subheader("Input Preview")
st.dataframe(input_df, use_container_width=True)
st.markdown("---")

# -------------------------------------------------------------------
# PREDICTION
# -------------------------------------------------------------------
if st.button("🔮 Predict Placement"):
    if model is None:
        st.error("Model is not loaded correctly.")
    else:
        try:
            X_pred = input_df[active_features]
            pred = model.predict(X_pred)[0]
            proba = model.predict_proba(X_pred)[0][int(pred)]

            if int(pred) == 1:
                st.success(
                    f"🎉 Prediction: **PLACED**\n\n"
                    f"Estimated probability: **{proba:.2%}**"
                )
            else:
                st.warning(
                    f"⚠️ Prediction: **NOT PLACED**\n\n"
                    f"Estimated probability: **{proba:.2%}**"
                )
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

