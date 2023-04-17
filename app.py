import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import optuna
import shap
import matplotlib.pyplot as plt
path = "train.csv"
data = pd.read_csv(path)
y = data['SalePrice']
X = data[["LotArea","OverallQual", "OverallCond", "YearBuilt","TotRmsAbvGrd","GarageArea"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = xgb.XGBRegressor(objective ='reg:squarederror', 
                 colsample_bytree = 1, 
                 eta=0.3, 
                 learning_rate = 0.26364097356592897,
                 max_depth = 9, 
                 alpha = 10, 
                 n_estimators = 700)
model.fit(X_train, y_train)
sidebar = st.sidebar
sidebar.title("Input Features")
lot_area = sidebar.slider("Lot Area", 1300, 215245, 1300)
overall_qual = sidebar.slider("Overall Quality", 1, 10, 6)
overall_cond = sidebar.slider("Overall Condition", 1, 10, 6)
year_built = sidebar.slider("Year Built", 1872, 2010, 1980)
tot_rooms_above_grade = sidebar.slider("Total Rooms Above Grade", 2, 14, 5)
garage_area = sidebar.slider("Garage Area", 0, 1418, 462)
input_df = pd.DataFrame({
    "LotArea": [lot_area],
    "OverallQual": [overall_qual],
    "OverallCond": [overall_cond],
    "YearBuilt": [year_built],
    "TotRmsAbvGrd": [tot_rooms_above_grade],
    "GarageArea": [garage_area]
})
prediction = model.predict(input_df)
st.write(f"The estimated house price range is ${prediction[0]:,.2f}")
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
st.pyplot(fig, bbox_inches='tight')
fig2, ax2 = plt.subplots()
shap.plots.beeswarm(shap_values, show=False)
st.pyplot(fig2, bbox_inches='tight')
fig3, ax3 = plt.subplots()
shap.plots.waterfall(shap_values[0])
st.pyplot(fig3, bbox_inches='tight')