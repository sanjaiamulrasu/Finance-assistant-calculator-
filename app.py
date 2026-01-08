import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("finance_data.csv")

X = df.drop("risk_label", axis=1)
y = df["risk_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    objective="multi:softmax",
    num_class=3,
    random_state=42
)

model.fit(X_train, y_train)

st.set_page_config(page_title="AI Finance Assistant", layout="centered")

st.title("💰 AI Finance Assistant")
st.write("Analyze your expenses and predict your financial risk level.")

st.sidebar.header("📊 Enter Monthly Details")

income = st.sidebar.number_input("Income (₹)", 10000, 200000, 50000)
food = st.sidebar.number_input("Food Expense (₹)", 1000, 30000, 8000)
rent = st.sidebar.number_input("Rent / EMI (₹)", 2000, 60000, 15000)
shopping = st.sidebar.number_input("Shopping (₹)", 500, 40000, 7000)
travel = st.sidebar.number_input("Travel (₹)", 0, 20000, 3000)
savings = st.sidebar.number_input("Savings (₹)", 0, 50000, 5000)
credit = st.sidebar.slider("Credit Card Usage (%)", 0, 100, 40)

if st.button("🔍 Analyze Financial Risk"):
    input_data = pd.DataFrame([{
        "income": income,
        "food_expense": food,
        "rent": rent,
        "shopping": shopping,
        "travel": travel,
        "savings": savings,
        "credit_usage": credit
    }])

    prediction = model.predict(input_data)[0]

    labels = {0: "Low Risk 🟢", 1: "Medium Risk 🟡", 2: "High Risk 🔴"}
    result = labels[prediction]

    st.subheader("📌 Risk Assessment Result")
    # st.success(result) if prediction == 0 else st.warning(result) if prediction == 1 else st.error(result)
    st.subheader("🧠 Why This Risk Level?")

    reasons = []

    if savings < 0.1 * income:
        reasons.append("Savings are less than 10% of income")

    if credit > 70:
        reasons.append("High credit card usage")

    if shopping > 0.3 * income:
        reasons.append("Shopping expense is very high compared to income")

    if rent > 0.4 * income:
        reasons.append("Rent/EMI consumes a large portion of income")

    if reasons:
        for r in reasons:
            st.write("🔴", r)
    else:
        st.write("🟢 Financial indicators are within safe limits")
    
    st.subheader("💡 Smart Suggestions")
    if prediction == 2:
        st.write("- Reduce shopping & discretionary expenses")
        st.write("- Increase savings to at least 20% of income")
        st.write("- Lower credit card usage below 50%")
    elif prediction == 1:
        st.write("- Monitor monthly spending closely")
        st.write("- Improve savings consistency")
    else:
        st.write("- Excellent financial discipline!")
        st.write("- Consider investments for wealth growth")
    st.subheader("📊 Expense Breakdown")

    expenses = {
        "Food": food,
        "Rent": rent,
        "Shopping": shopping,
        "Travel": travel
    }

    fig1, ax1 = plt.subplots()
    ax1.pie(expenses.values(), labels=expenses.keys(), autopct="%1.1f%%", startangle=90)
    ax1.axis("equal")

    st.pyplot(fig1)
    st.subheader("💸 Income Utilization Overview")

    total_expense = food + rent + shopping + travel

    fig2, ax2 = plt.subplots()
    ax2.bar(
        ["Income", "Expenses", "Savings"],
        [income, total_expense, savings]
    )

    st.pyplot(fig2)
    expense_ratio = (total_expense / income) * 100

    st.write(f"📌 **Expense Ratio:** {expense_ratio:.2f}%")

    if expense_ratio > 70:
        st.error("⚠️ You are spending more than 70% of your income.")
    elif expense_ratio > 50:
        st.warning("⚠️ Moderate spending level.")
    else:
        st.success("✅ Healthy spending pattern.")

st.markdown("---")
st.caption("Built using Machine Learning (XGBoost)")
