import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Load or train model
@st.cache_resource
def load_model():
    from sklearn.model_selection import train_test_split
    df = pd.read_csv("c:/Users/sneha/OneDrive/Documents/kaggle-data/train.csv")
    df = df[["Survived", "Pclass", "Sex", "Age", "Fare", "Embarked"]].dropna()
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X

model, X_train = load_model()

# Sidebar trivia
st.sidebar.title("🧠 Titanic Trivia")
st.sidebar.info(
    """
    - 🚢 Titanic was 882 feet long.
    - 💺 There were 3 classes: 1st, 2nd, 3rd.
    - 👥 Around 2,224 people were onboard.
    - 💔 Only ~32% survived.
    """
)

st.title("🎯 Titanic Survival Predictor")

# User input form
with st.form("input_form"):
    sex = st.selectbox("Sex", ["Male", "Female"])
    pclass = st.selectbox("Passenger Class (1 = 1st, 3 = 3rd)", [1, 2, 3])
    age = st.slider("Age", 0, 100, 25)
    fare = st.slider("Fare", 0.0, 600.0, 32.0)
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])
    submitted = st.form_submit_button("Predict")

if submitted:
    # Preprocess input
    input_df = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": 0 if sex == "Male" else 1,
        "Age": age,
        "Fare": fare,
        "Embarked": {"C": 0, "Q": 1, "S": 2}[embarked]
    }])

    # Prediction and probability
    prob = model.predict_proba(input_df)[0][1]
    st.subheader(f"🧾 Prediction Probability: {prob:.2%}")
    st.write("💡 This is the model's confidence that the passenger would **survive**.")


