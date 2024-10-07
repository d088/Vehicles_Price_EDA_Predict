import os
import pickle
import zipfile
import pandas as pd
import plotly.express as px
import streamlit as st

# Function to extract zip file
def extract_zip(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Load dataset function
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_car_prices.csv")

# Load model function
@st.cache_resource
def load_model():
    with open('car_price_prediction_model.pkl', 'rb') as file:
        return pickle.load(file)

# Main function to run the app
def main():
    # Check if model file is already extracted
    if not os.path.exists("car_price_prediction_model.pkl"):
        extract_zip("car_price_prediction_model.zip", ".")

    # Load the dataset and model
    df = load_data()
    model = load_model()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Dashboard", "Model Prediction"])

    if page == "Dashboard":
        # Title
        st.title("Car Price Dashboard")

        # Sidebar for filters
        st.sidebar.header("Filters")

        all_makes = df['make'].unique().tolist()
        selected_makes = st.sidebar.multiselect(
            "Select car makes",
            options=["all"] + all_makes,
            default=["all"]
        )

        # Filter the dataframe based on selected makes
        filtered_df = df if "all" in selected_makes else df[df['make'].isin(selected_makes)]

        # Car model selection based on make
        all_models = filtered_df['model'].unique().tolist()
        selected_models = st.sidebar.multiselect(
            "Select car models",
            options=["all"] + all_models,
            default=["all"]
        )

        # Filter based on selected models
        filtered_df = filtered_df if "all" in selected_models else filtered_df[filtered_df['model'].isin(selected_models)]

        # Additional filters
        transmissions = df['transmission'].unique().tolist()
        selected_transmissions = st.sidebar.multiselect("Select transmission types", options=["all"] + transmissions, default=["all"])
        filtered_df = filtered_df if "all" in selected_transmissions else filtered_df[filtered_df['transmission'].isin(selected_transmissions)]

        states = df['state'].unique().tolist()
        selected_states = st.sidebar.multiselect("Select states", options=["all"] + states, default=["all"])
        filtered_df = filtered_df if "all" in selected_states else filtered_df[filtered_df['state'].isin(selected_states)]

        odometer_range = st.sidebar.slider(
            "Select odometer range",
            min_value=int(df['odometer'].min()),
            max_value=int(df['odometer'].max()),
            value=(int(df['odometer'].min()), int(df['odometer'].max()))
        )
        filtered_df = filtered_df[(filtered_df['odometer'] >= odometer_range[0]) & (filtered_df['odometer'] <= odometer_range[1])]

        year_range = st.sidebar.slider(
            "Select year range",
            min_value=int(df['year'].min()),
            max_value=int(df['year'].max()),
            value=(int(df['year'].min()), int(df['year'].max()))
        )
        filtered_df = filtered_df[(filtered_df['year'] >= year_range[0]) & (filtered_df['year'] <= year_range[1])]

        sell_price_range = st.sidebar.slider(
            "Select sell price range",
            min_value=int(df['sell_price'].min()),
            max_value=int(df['sell_price'].max()),
            value=(int(df['sell_price'].min()), int(df['sell_price'].max()))
        )
        filtered_df = filtered_df[(filtered_df['sell_price'] >= sell_price_range[0]) & (filtered_df['sell_price'] <= sell_price_range[1])]

        # KPIs
        make_count = filtered_df['make'].nunique()
        model_count = filtered_df['model'].nunique()
        average_sell_price = filtered_df['sell_price'].mean()

        # Display KPIs
        st.subheader("Key Performance Indicators")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Unique Makes", make_count)
        with col2:
            st.metric("Unique Models", model_count)
        with col3:
            st.metric("Average Sell Price", f"${average_sell_price:,.2f}")

        # Descriptive statistics
        st.subheader("Descriptive Statistics")
        desc_stats = filtered_df[['sell_price', 'odometer']].describe()
        html_table = desc_stats.to_html(classes='table table-striped', justify='center', border=0)

        # Center the table
        st.markdown(f"<div style='text-align: center;'>{html_table}</div>", unsafe_allow_html=True)

        # Graph type selection
        graph_type = st.sidebar.selectbox("Select graph type", options=['bar', 'count bar', 'line', 'bi_axes_line', 'scatter', 'pie'])

        # Graph generation
        if filtered_df.empty:
            st.warning("No data available for the selected filters.")
        else:
            if graph_type == 'bar':
                fig = px.bar(filtered_df.groupby('make', as_index=False).agg({'sell_price': 'mean'}), x='make', y='sell_price', title='Average Sell Price by Make')
            elif graph_type == 'count bar':
                fig = px.bar(filtered_df['make'].value_counts().reset_index(), x='make', y='count', title='Cars Count by Make')
            elif graph_type == 'line':
                fig = px.line(filtered_df.groupby('year').agg({'sell_price': 'mean'}).reset_index(), x='year', y='sell_price', title='Average Sell Price Over Years')
            elif graph_type == 'bi_axes_line':
                fig = px.line(filtered_df.groupby('year').agg(avg_sell_price=('sell_price', 'mean'), avg_mmr=('mmr', 'mean')).reset_index(),
                              x='year', y=['avg_sell_price', 'avg_mmr'], markers=True, title='Average Sell Price & MMR Over Years')
            elif graph_type == 'pie':
                fig = px.pie(filtered_df['transmission'].value_counts().reset_index(), names='transmission', values='count', title='Transmission Distribution')
            else:
                fig = px.scatter(filtered_df, x='odometer', y='sell_price', color='make', title='Odometer vs Sell Price')

            st.plotly_chart(fig)

            # Colors graphs
            col1, col2 = st.columns(2)
            color_count_fig = px.bar(filtered_df['color'].value_counts().reset_index(), x='count', y='color', title='Count by Car Color', labels={'count': 'Count', 'color': 'Color'})
            interior_color_count_fig = px.bar(filtered_df['interior'].value_counts().reset_index(), x='count', y='interior', title='Count by Interior Color', labels={'count': 'Count', 'interior': 'Interior Color'})

            with col1:
                st.plotly_chart(color_count_fig, use_container_width=True)
            with col2:
                st.plotly_chart(interior_color_count_fig, use_container_width=True)

            # Condition and state tables
            condition_sell_price_count_df = filtered_df.groupby('condition').agg(average_sell_price=('sell_price', 'mean'), count=('condition', 'size')).reset_index()
            condition_sell_price_count_df = condition_sell_price_count_df.sort_values(by='average_sell_price', ascending=False)

            state_sell_price_df = filtered_df.groupby('state').agg(average_sell_price=('sell_price', 'mean'), count=('state', 'size')).reset_index()
            state_sell_price_df = state_sell_price_df.sort_values(by='average_sell_price', ascending=False).head(5)

            # Display condition and state tables
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Average Sell Price by Condition")
                st.dataframe(condition_sell_price_count_df)

            with col2:
                st.subheader("Average Sell Price by State")
                st.dataframe(state_sell_price_df)

            # Pivot table graph
            pivot_table = filtered_df.pivot_table(values='sell_price', index='make', columns='transmission', aggfunc='mean', fill_value=0)
            pivot_fig = px.imshow(pivot_table, title='Average Sell Price by Make and Transmission', labels=dict(x="Transmission", y="Make", color="Average Sell Price"))
            st.plotly_chart(pivot_fig)

        # Reset filters button
        if st.sidebar.button("Reset Filters"):
            st.experimental_rerun()

        # Download data button
        if st.sidebar.button("Download Filtered Data"):
            filtered_df.to_csv("filtered_data.csv", index=False)
            st.sidebar.success("Data downloaded successfully!")

    elif page == "Model Prediction":
        # Title
        st.title("Car Price Prediction")

        # Input features for prediction
        year = st.selectbox("Year", options=df['year'].unique())
        make = st.selectbox("Make", options=df['make'].unique())
        model_input = st.selectbox("Model", options=df['model'].unique())
        body = st.selectbox("Body", options=df['body'].unique())
        transmission = st.selectbox("Transmission", options=df['transmission'].unique())
        state = st.selectbox("State", options=df['state'].unique())
        odometer = st.number_input("Odometer", min_value=0, max_value=int(df['odometer'].max()), value=0)
        color = st.selectbox("Color", options=df['color'].unique())
        sale_month = st.selectbox("Sale Month", options=df['sale_month'].unique())
        condition = st.selectbox("Condition", options=df['condition'].unique())

        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'year': [year],
            'make': [make],
            'model': [model_input],
            'body': [body],
            'transmission': [transmission],
            'state': [state],
            'odometer': [odometer],
            'color': [color],
            'sale_month': [sale_month],
            'condition': [condition]
        }, index=[0])

        # Predict button
        if st.button("Predict Price"):
            prediction = model.predict(input_data)
            st.success(f"Predicted Price: ${prediction[0]:,.2f}")

# Run the main function
if __name__ == "__main__":
    main()
