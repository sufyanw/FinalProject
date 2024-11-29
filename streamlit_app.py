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

st.set_page_config(page_title='Housing Crisis App')

df = pd.read_csv("housing.csv")

import dagshub
dagshub.init(repo_owner='sufyanw', repo_name='FinalProject', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)


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
    df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)
    st.title("Predicting Housing Prices and Categories 💡")
    
    tab1, tab2 = st.tabs(["Linear Regression", "KNN Confusion Matrix"])
    
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
            st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
            st.write(f"R² Score: {r2:.2f}")
    
    with tab2:
        df['price_category'] = pd.cut(df['median_house_value'], 
                                      bins=[0, 100000, 200000, 300000, 400000, 500000, float('inf')],
                                      labels=[0, 1, 2, 3, 4, 5])
        X = df[['median_income', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households']]
        y = df['price_category']

        X = pd.get_dummies(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"### KNN Model Accuracy: {accuracy * 100:.2f}%")
        
        option = st.selectbox(
            'What would you like to explore?',
            ('Confusion Matrix 📈', 'Classification Report 📑')
        )

        if option == 'Confusion Matrix 📈':
            conf_matrix = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=knn.classes_, yticklabels=knn.classes_)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix: KNN Model Prediction')
            st.pyplot(plt)

        elif option == 'Classification Report 📑':
            report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
            st.text('Classification Report:')
            st.table(report)


elif selected == "MLFlow":
    df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)
    st.title("MLFlow Integration 🌩️")
    st.write("""
    ## Model Tracking with MLFlow
    This app integrates MLFlow through DagHub to track the following:
    - Experiment runs and parameters.
    - Performance metrics (MAE, R² Score).
    - Model artifacts for reproducibility.
    """)

    MODELS = {
        "regression": {
            "Linear Regression": LinearRegression,
        }
    }

    task_type = "regression"

    st.write("### Housing Data Overview")
    st.dataframe(df)

    model_options = list(MODELS[task_type].keys())
    model_choice = st.selectbox("Choose a model ⚙️", model_options)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_choice = st.multiselect("Select Features for Prediction", numeric_columns)
    target_choice = st.selectbox("Select Target Variable", ["median_house_value"])

    track_with_mlflow = st.checkbox("Track with MLflow?")

    if st.button("Start Training"):
        if not feature_choice or not target_choice:
            st.error("Please select features and a target variable before training.")
        else:
            if track_with_mlflow:
                if mlflow.active_run():
                    mlflow.end_run()
                mlflow.set_experiment("Housing_Price_Prediction")
            with mlflow.start_run():
                if track_with_mlflow:
                    mlflow.log_param("model", model_choice)
                    mlflow.log_param("features", feature_choice)

                model = MODELS[task_type][model_choice]()
                X = df[feature_choice]
                y = df[target_choice]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                model.fit(X_train, y_train)

                preds_train = model.predict(X_train)
                preds_test = model.predict(X_test)
                mae_train = metrics.mean_absolute_error(y_train, preds_train)
                mae_test = metrics.mean_absolute_error(y_test, preds_test)
                r2_train = metrics.r2_score(y_train, preds_train)
                r2_test = metrics.r2_score(y_test, preds_test)

                st.write("### Training Metrics")
                st.write(f"MAE (Train): {mae_train:.2f}")
                st.write(f"MAE (Test): {mae_test:.2f}")
                st.write(f"R² (Train): {r2_train:.2f}")
                st.write(f"R² (Test): {r2_test:.2f}")

                if track_with_mlflow:
                    mlflow.log_metric("MAE_train", mae_train)
                    mlflow.log_metric("MAE_test", mae_test)
                    mlflow.log_metric("R2_train", r2_train)
                    mlflow.log_metric("R2_test", r2_test)
                    mlflow.sklearn.log_model(model, "model")

            with open('housing_model.pkl', 'wb') as file:
                pickle.dump(model, file)

    def download_file():
        file_path = 'housing_model.pkl'
        with open(file_path, 'rb') as file:
            contents = file.read()
        b64 = base64.b64encode(contents).decode()
        href = f'<a href="data:file/pkl;base64,{b64}" download="housing_model.pkl">Download housing_model.pkl</a>'
        st.markdown(href, unsafe_allow_html=True)

    st.title("Download Model Example")
    st.write("Click the button below to download the housing_model.pkl file.")
    if st.button("Download"):
        download_file()


elif selected == "Explainable AI":
    st.title("Explainable AI 🔎🤖")
    st.write("""
    This pip install shapash
ion uses **Shapash**, an explainability library, to provide insights into the predictions made by the housing price prediction model.
    """)

    from shapash.explainer.smart_explainer import SmartExplainer
    import random

    df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

    X = df[['median_income', 'total_rooms', 'housing_median_age', 'total_bedrooms', 'population', 'households']]
    y = df['median_house_value']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestRegressor(max_depth=10, random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    option = st.selectbox(
        'What would you like to explore❓',
        ('Feature Importance', 'Feature Contribution', 'Local Explanation')
    )

    xpl = SmartExplainer(model=model)
    xpl.compile(x=X_test, y_pred=model.predict(X_test), features_dict={col: col for col in X.columns})

    if option == 'Feature Importance':
        st.write("### Feature Importance")
        fig = xpl.plot.features_importance(title="Feature Importance in Predicting Housing Prices")
        st.pyplot(fig.figure)

    if option == 'Feature Contribution':
        st.write("### Feature Contribution")
        feature_list = X_test.columns.tolist()
        selected_feature = st.selectbox('Select a feature to analyze its contribution:', feature_list)
        fig = xpl.plot.contribution_plot(selected_feature, title=f"Contribution of {selected_feature}")
        st.pyplot(fig.figure)

    if option == 'Local Explanation':
        st.write("### Local Explanation")
        random_index = random.choice(X_test.index)
        st.write(f"Local explanation for index: {random_index}")
        fig = xpl.plot.local_plot(index=random_index, title=f"Local Explanation for Index {random_index}")
        st.pyplot(fig.figure)


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