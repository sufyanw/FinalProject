import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import codecs
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title='Housing Crisis App')

df = pd.read_csv("housing.csv")


selected = option_menu(
    menu_title=None,
    options=["Introduction", "Exploration", "Visualization", "Prediction", "MLFlow", "Explainable AI", "Conclusion"],
    icons=["house", "search","bar-chart-line", "lightbulb", "cloud", "robot", "check-circle"],
    default_index=0,
    orientation="horizontal",
)

if selected == 'Introduction':
    st.title("Housing Crisis 🏠")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image_path = Image.open("housing_image.jpg")
        st.image(image_path, width=400)

    st.write("""
    ## Introduction
    Housing affordability and availability are pressing issues in California, impacting millions of residents and the state's economy. This app explores California housing price data to uncover trends, correlations, and potential solutions for combating the housing crisis.

    ## Objective
    This app aims to:
    - Explore factors influencing housing prices.
    - Analyze trends in affordability and availability.
    - Provide actionable insights and potential solutions to address the housing crisis.

    ## Key Features
    - Visualization of housing price trends and influential factors.
    - Analysis of correlations between demographics, geography, and housing costs.
    - Predictive modeling for housing prices.
    """)

elif selected == 'Exploration':
    
    st.title("Data Exploration 🔍")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dataset Head", "Dataset Tail", "Description", "Missing Values", "Generate Report"])

    with tab1:
        st.subheader("Head of the Dataset")
        st.dataframe(df.head())

    with tab2:
        st.subheader("Tail of the Dataset")
        st.dataframe(df.tail())

    with tab3:
        st.subheader("Description of the Dataset")
        st.dataframe(df.describe())

    with tab4:
        df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)
        st.subheader("Missing values")
        dfnull = df.isnull()/len(df)*100
        total_missing = dfnull.sum().round(2)
        st.write(total_missing)
        if total_missing[0] == 0.0:
            st.success("Congrats, there are no missing values!")
        else:
            st.error("There are missing values.")

    with tab5:
        if st.button("Generate Report"):
            def read_html_report(file_path):
                with codecs.open(file_path, 'r', encoding="utf-8") as f:
                    return f.read()
            
            html_report = read_html_report('housing_report.html')
            
            st.title("Streamlit Quality Report")
            st.components.v1.html(html_report, height=1000, scrolling=True)


elif selected == 'Visualization':
    st.title("Data Visualization 📊")
    tab1, tab2, tab3, tab4 = st.tabs(["Price Distribution", "Geographic Heatmap", "Correlation Heatmap", "Feature Relationships"])

    with tab1:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['median_house_value'], bins=50, kde=True, ax=ax)
        ax.set_title("Distribution of Housing Prices")
        st.pyplot(fig)

    with tab2:
        st.subheader("Geographic Heatmap of House Values")

        cubehelix_cmap = sns.cubehelix_palette(start=2, rot=0, dark=0, light=0.95, reverse=True, as_cmap=True)

        min_value = df['median_house_value'].min()
        max_value = df['median_house_value'].max()

        df['normalized_value'] = (df['median_house_value'] - min_value) / (max_value - min_value)

        def get_rgb_color(value):
            rgba = cubehelix_cmap(value)
            return [int(c * 255) for c in rgba[:3]] 

        df['color'] = df['normalized_value'].apply(get_rgb_color)

        df['size'] = df['normalized_value'] * 100

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position=["longitude", "latitude"],
            get_fill_color="color",
            get_radius="size",
            radius_scale=10,
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=df['latitude'].mean(),
            longitude=df['longitude'].mean(),
            zoom=6,
            pitch=0,
        )

        map = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={"text": "Price: {median_house_value}"},
        )

        st.pydeck_chart(map)

    with tab3:
        st.subheader("Correlation Heatmap")
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numerical_columns].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig)

    with tab4:
        st.subheader("Relationships Between Features")
        x_feature = st.selectbox("Select X-axis Feature:", df.columns)
        y_feature = st.selectbox("Select Y-axis Feature:", df.columns)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_feature, y=y_feature, ax=ax)
        ax.set_title(f"Relationship Between {x_feature} and {y_feature}")
        st.pyplot(fig)

elif selected == "Prediction":
    df.fillna(-1, inplace=True)
    st.title("Predicting Housing Prices and Categories 💡")
    
    tab1, tab2 = st.tabs(["Linear Regression", "Decision Tree"])
    df = df.drop('ocean_proximity', axis=1)
    
    with tab1:
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        features = st.multiselect("Select Features for Prediction", numeric_columns)
        target = st.selectbox("Select Target Variable", ["median_house_value"])

        if features:
            X = df[features]
            y = df[target]
            test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            mae = metrics.mean_absolute_error(y_test, predictions)
            mae = mae / 100000
            r2 = metrics.r2_score(y_test, predictions)
            
            st.write("### Prediction Results")
            avg_predicted_price = np.mean(predictions) 
            st.write("Average Predicted House Price:", round(avg_predicted_price, 2))
            st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
            st.write(f"R² Score: {r2:.2f}")
    
    with tab2:
        st.title("Decision Tree 🌳")
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        features = st.multiselect("Select Features for Prediction", numeric_columns, key="dt_features")
        target = st.selectbox("Select Target Variable", ["median_house_value"], key="dt_target")

        if features:
            X = df[features]
            y = df[target].values.ravel()
            test_size = st.slider("Test Size (%)", 10, 50, 20, key="dt_test_size") / 100
            max_depth = st.slider("Max Depth of Tree", 1, 20, 5, key="dt_max_depth")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

            model = DecisionTreeRegressor(max_depth=max_depth)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            mae = metrics.mean_absolute_error(y_test, predictions)
            mae = mae/100000
            r2 = metrics.r2_score(y_test, predictions)
            st.write("### Prediction Results")
            avg_predicted_price = np.mean(predictions)
            st.write("Average Predicted House Price:", round(avg_predicted_price, 2))
            st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
            st.write(f"R² Score: {r2:.2f}")


elif selected == "MLFlow":
    st.title("MLFlow Integration 🌩️")
    st.write("""
    ## Model Tracking with MLFlow
    This app integrates MLFlow through DagHub to track the following:
    - Experiment runs and parameters.
    - Performance metrics (MAE, R² Score).
    - Model artifacts for reproducibility.
    """)
    import dagshub
    import mlflow
    import streamlit_shadcn_ui as ui
    ui.link_button(text="👉 Go to ML Flow",url="https://dagshub.com/sufyanw/FinalProject.mlflow/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D",key="link_btnmlflow")

elif selected == "Explainable AI":
    st.title("Explainable AI 🔎🤖")
    st.write("""
    This section uses **Shapash**, an explainability library, to provide insights into the predictions made by the housing price prediction model.
    """)

    from shapash.explainer.smart_explainer import SmartExplainer

    df.fillna(-1, inplace=True)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    features = st.multiselect("Select Features for Explainability", numeric_columns, key="xai_features")
    target = st.selectbox("Select Target Variable", ["median_house_value"], key="xai_target")

    if features:
        X = df[features]
        y = df[target]
        test_size = st.slider("Test Size (%)", 10, 50, 20, key="xai_test_size") / 100
        random_state = st.number_input("Random State (optional)", value=0, min_value=0, key="xai_random_state")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        model = DecisionTreeRegressor(random_state=random_state)
        model.fit(X_train, y_train)

        explainer = SmartExplainer(model=model)
        y_pred = model.predict(X_test)
        y_pred_series = pd.Series(y_pred, index=X_test.index, name="Predictions")
        explainer.compile(x=X_test, y_pred = y_pred_series)

        st.write("### Global Explainability")
        importance_plot = explainer.plot.features_importance()
        st.plotly_chart(importance_plot, use_container_width=True)

        st.write("### Local Explainability")
        selected_index = st.slider("Select an Index for Explanation", 0, len(X_test) - 1, 0)
        local_explainability = explainer.plot.local_plot(index=selected_index)
        st.plotly_chart(local_explainability, use_container_width=True)


elif selected == 'Conclusion':
    st.title("Conclusion 🏁")
    st.write("""
    ### Key Insights:
    1. **Housing Affordability**: Rising housing costs in California are closely linked to population density and proximity to urban centers.
    2. **Influential Factors**: Features like household income, location, and proximity to amenities significantly impact housing prices.

    ### Proposed Solutions:
    1. **Affordable Housing Initiatives**: Increase funding for affordable housing projects and incentivize developers.
    2. **Zoning Reforms**: Encourage high-density housing developments through zoning changes.
    3. **Public Transportation Investments**: Improve transportation infrastructure to connect remote areas with urban job markets.
    """)