import streamlit as st
import pandas as pd
import pickle

# === Load Model and Metadata from Pickle ===
with open("bigmart_best_model.pkl", "rb") as f:
    model_artifacts = pickle.load(f)

model = model_artifacts['pipeline']
sklearn_version = model_artifacts['sklearn_version']
columns = model_artifacts['columns']

st.title("🛒 BigMart Sales Prediction App")
st.markdown(f"Model trained with **scikit-learn v{sklearn_version}**")

# === User Inputs ===
st.header("📋 Enter Product and Outlet Details")

Item_Identifier = st.text_input("Item Identifier", "FDA15")
Item_Weight = st.number_input("Item Weight", min_value=0.0, value=12.5)
Item_Fat_Content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
Item_Visibility = st.slider("Item Visibility", min_value=0.0, max_value=0.3, step=0.01, value=0.1)
Item_Type = st.selectbox("Item Type", [
    "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household",
    "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast", 
    "Health and Hygiene", "Hard Drinks", "Canned", "Breads", 
    "Starchy Foods", "Others", "Seafood"
])
Item_MRP = st.number_input("Item MRP", min_value=0.0, value=150.0)

Outlet_Identifier = st.selectbox("Outlet Identifier", [
    "OUT027", "OUT013", "OUT049", "OUT035", "OUT046", 
    "OUT017", "OUT045", "OUT018", "OUT019", "OUT010"
])
Outlet_Size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
Outlet_Location_Type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
Outlet_Type = st.selectbox("Outlet Type", [
    "Supermarket Type1", "Supermarket Type2", 
    "Supermarket Type3", "Grocery Store"
])
Outlet_Age = st.slider("Outlet Age (Years)", 0, 40, 15)

# === Predict Button ===
if st.button("🔮 Predict Sales"):
    input_df = pd.DataFrame([{
        "Item_Identifier": Item_Identifier,
        "Item_Weight": Item_Weight,
        "Item_Fat_Content": Item_Fat_Content,
        "Item_Visibility": Item_Visibility,
        "Item_Type": Item_Type,
        "Item_MRP": Item_MRP,
        "Outlet_Identifier": Outlet_Identifier,
        "Outlet_Size": Outlet_Size,
        "Outlet_Location_Type": Outlet_Location_Type,
        "Outlet_Type": Outlet_Type,
        "Outlet_Age": Outlet_Age
    }])

    # Make prediction
    prediction = model.predict(input_df)[0]
    st.success(f"📈 Predicted Item Outlet Sales: ₹{prediction:,.2f}")
