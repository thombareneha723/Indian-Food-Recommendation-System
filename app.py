import streamlit as st
import joblib
import pandas as pd

# ============================
# LOAD SAVED MODEL FILES
# ============================
df = joblib.load("df.pkl")                     # Your cleaned dataframe
similarity_matrix = joblib.load("similarity.pkl")
scaler = joblib.load("scaler.pkl")            # (Not used directly here, but kept for completeness)

# ============================
# RECOMMENDATION FUNCTION
# ============================
def recommend_dishes(dish_name, df, similarity_matrix, top_n=10):
    if dish_name not in df['Dish Name'].values:
        return ["Dish not found in dataset"]
    
    idx = df[df['Dish Name'] == dish_name].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_results = sorted_scores[1:top_n+1]
    
    return [df.iloc[i]['Dish Name'] for i, score in top_results]

# ============================
# STREAMLIT UI
# ============================
st.set_page_config(page_title="Food Recommendation System", layout="centered")

st.title("🍽️ Food Recommendation System")
st.write("Select a dish to get top similar nutritional recommendations.")

# Dropdown for dish selection
dish_list = sorted(df['Dish Name'].unique())
selected_dish = st.selectbox("Choose a Dish:", dish_list)

# Button
if st.button("Get Recommendations"):
    with st.spinner("Finding similar dishes..."):
        recommendations = recommend_dishes(selected_dish, df, similarity_matrix, top_n=10)

    st.subheader("Recommended Dishes:")
    for i, dish in enumerate(recommendations, start=1):
        st.write(f"**{i}. {dish}**")
