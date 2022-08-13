import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import joblib
import shap

# Add and resize an image to the top of the app
img_fuel = Image.open("img/fuel_efficiency.png")
st.image(img_fuel, width=700)

st.markdown("<h1 style='text-align: center; color: black;'>Fuel Efficiency</h1>", unsafe_allow_html=True)

# Import train dataset to DataFrame
train_df = pd.read_csv("dat/X_train.csv", index_col=0)
model_results_df = pd.read_csv("dat/Results.csv", index_col=0)
test_df = pd.read_csv("dat/X_test.csv", index_col=0)
tree_test_df = pd.read_csv("dat/X_tree_test.csv", index_col=0)

# Import shap datasets to DataFrame
dnn_shap_local = pd.read_csv("dat/dnn_shap_local.csv")
dnn_shap_global = pd.read_csv("dat/dnn_shap_global.csv")

tpot_shap_local = pd.read_csv("dat/tpot_shap_local.csv")
tpot_shap_global = pd.read_csv("dat/tpot_shap_global.csv")

# Create sidebar for user selection
with st.sidebar:
    # Add FB logo
    st.image("https://user-images.githubusercontent.com/37101144/161836199-fdb0219d-0361-4988-bf26-48b0fad160a3.png" )    

    # Available models for selection

    # YOUR CODE GOES HERE!
    models = ["Linear", "DNN", "TPOT"]

    # Add model select boxes
    model1_select = st.selectbox(
        "Choose Model 1:",
        (models)
    )
    
    # Remove selected model 1 from model list
    # App refreshes with every selection change.
    models.remove(model1_select)
    
    model2_select = st.selectbox(
        "Choose Model 2:",
        (models)
    )

# Create tabs for separation of tasks
tab1, tab2, tab3 = st.tabs(["🗃 Data", "🔎 Model Results", "🤓 Model Explainability"])

with tab1:    
    # Data Section Header
    st.header("Raw Data")

    # Display first 100 samples of the dateframe
    st.dataframe(train_df.head(100))

    st.header("Correlations")

    # Heatmap
    corr = train_df.corr()
    fig = px.imshow(corr)
    st.write(fig)

with tab2:    
    
    # YOUR CODE GOES HERE!

    # Columns for side-by-side model comparison
    col1, col2 = st.columns(2)

    model_dict = {"DNN": "dnn_model", "TPOT": "tpot_extratrees", "Linear": "linear_model"}
    # Build the confusion matrix for the first model.
    with col1:
        st.header(model1_select)

        # YOUR CODE GOES HERE!
        row_name = model_dict[model1_select]
        st.write(model_results_df.loc[row_name])

    # Build confusion matrix for second model
    with col2:
        st.header(model2_select)

        # YOUR CODE GOES HERE!
        row_name = model_dict[model2_select]
        st.write(model_results_df.loc[row_name])


with tab3: 
    # YOUR CODE GOES HERE!
        # Use columns to separate visualizations for models
        # Include plots for local and global explanability!
    # st.header(model1_select)
    st.header("DNN")

    # shap_dict = {"DNN": "dnn_explainer", "TPOT": "tpot_explainer"}
    # filename_1 = "dat/" + shap_dict[model1_select] + ".bz2"
    filename_1 = "dat/dnn_explainer.bz2"

    explainer_1 = joblib.load(filename=filename_1)
    st.write(shap.initjs())
    st.write(shap.force_plot(explainer_1.expected_value, dnn_shap_local[0], \
                test_df.iloc[5,:]))

    # st.header(model2_select)
    st.header("TPOT")

    # filename_2 = "dat/" + shap_dict[model2_select] + ".bz2"
    filename_2 = "dat/tpot_explainer.bz2"

    explainer_2 = joblib.load(filename=filename_2)
    shap.initjs()
    shap.force_plot(explainer_2.expected_value, tpot_shap_local[0], \
                tree_test_df.iloc[5,:])
    
