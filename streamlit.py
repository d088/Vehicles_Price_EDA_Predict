import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import joblib


# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_car_prices.csv")

df = load_data()
df.columns.str.strip()
# Load the machine learning model
@st.cache_resource
def load_model():
    with open('car_price_prediction_model.pkl', 'rb') as file:
        return pickle.load(file)

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
        "select car makes",
        options=["all"] + all_makes,
        default=["all"]
    )

    # filter the dataframe based on selected makes
    if "all" in selected_makes:
        filtered_df = df
    else:
        filtered_df = df[df['make'].isin(selected_makes)]

    # car model selection based on make
    all_models = filtered_df['model'].unique().tolist()
    selected_models = st.sidebar.multiselect(
        "select car models",
        options=["all"] + all_models,
        default=["all"]
    )

    # filter the dataframe based on selected models
    if "all" in selected_models:
        filtered_df = filtered_df
    else:
        filtered_df = filtered_df[filtered_df['model'].isin(selected_models)]

    # additional filters
    transmissions = df['transmission'].unique().tolist()
    selected_transmissions = st.sidebar.multiselect("select transmission types", options=["all"] + transmissions, default=["all"])

    if "all" not in selected_transmissions:
        filtered_df = filtered_df[filtered_df['transmission'].isin(selected_transmissions)]

    states = df['state'].unique().tolist()
    selected_states = st.sidebar.multiselect("select states", options=["all"] + states, default=["all"])

    if "all" not in selected_states:
        filtered_df = filtered_df[filtered_df['state'].isin(selected_states)]

    odometer_range = st.sidebar.slider(
        "select odometer range",
        min_value=int(df['odometer'].min()),
        max_value=int(df['odometer'].max()),
        value=(int(df['odometer'].min()), int(df['odometer'].max()))
    )

    filtered_df = filtered_df[(filtered_df['odometer'] >= odometer_range[0]) & (filtered_df['odometer'] <= odometer_range[1])]

    # year and sell_price sliders
    year_range = st.sidebar.slider(
        "select year range",
        min_value=int(df['year'].min()),
        max_value=int(df['year'].max()),
        value=(int(df['year'].min()), int(df['year'].max()))
    )

    filtered_df = filtered_df[(filtered_df['year'] >= year_range[0]) & (filtered_df['year'] <= year_range[1])]

    sell_price_range = st.sidebar.slider(
        "select sell_price range",
        min_value=int(df['sell_price'].min()),
        max_value=int(df['sell_price'].max()),
        value=(int(df['sell_price'].min()), int(df['sell_price'].max()))
    )

    filtered_df = filtered_df[(filtered_df['sell_price'] >= sell_price_range[0]) & (filtered_df['sell_price'] <= sell_price_range[1])]

    # kpis
    make_count = filtered_df['make'].nunique()
    model_count = filtered_df['model'].nunique()
    average_sell_price = filtered_df['sell_price'].mean()

    # display kpis in one row
    st.subheader("key performance indicators")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("unique makes", make_count)
    with col2:
        st.metric("unique models", model_count)
    with col3:
        st.metric("average sell_price", f"${average_sell_price:,.2f}")

    # descriptive statistics
    st.subheader("descriptive statistics")
    desc_stats = filtered_df[['sell_price', 'odometer']].describe()
    html_table = desc_stats.to_html(classes='table table-striped', justify='center', border=0)

    # center the table using markdown
    st.markdown(
        f"<div style='text-align: center;'>{html_table}</div>",
        unsafe_allow_html=True
    )

    # graph type selection
    graph_type = st.sidebar.selectbox(
        "select graph type",
        options=['bar', 'count bar', 'line', 'bi_axes_line', 'scatter', 'pie']
    )

    # graph generation
    if filtered_df.empty:
        st.warning("no data available for the selected filters.")
    else:
        if graph_type == 'bar':
            fig = px.bar(filtered_df.groupby('make', as_index=False).agg({'sell_price': 'mean'}), x='make', y='sell_price', title='average sell_price by make')
        elif graph_type == 'count bar':
            fig = px.bar(filtered_df['make'].value_counts().reset_index(), x='make', y='count', title='cars count by make')
        elif graph_type == 'line':
            fig = px.line(filtered_df.groupby('year').agg({'sell_price': 'mean'}).reset_index(), x='year', y='sell_price', title='average sell_price over years')
        elif graph_type == 'bi_axes_line':
            fig = px.line(filtered_df.groupby('year').agg(avg_sell_price=('sell_price', 'mean'), avg_mmr=('mmr', 'mean')).reset_index(),
                        x='year', y=['avg_sell_price', 'avg_mmr'], markers=True, title='average sell_price & mmr over years')
        elif graph_type == 'pie':
            fig = px.pie(filtered_df['transmission'].value_counts().reset_index(), names='transmission', values='count', title='transmission distribution')
        else:
            fig = px.scatter(filtered_df, x='odometer', y='sell_price', color='make', title='odometer vs sell_price')

        st.plotly_chart(fig)

        # colors graphs
        col1, col2 = st.columns(2)
        color_count_fig = px.bar(filtered_df['color'].value_counts().reset_index(), x='count', y='color', title='count by car color', labels={'count': 'count', 'color': 'color'})
        interior_color_count_fig = px.bar(filtered_df['interior'].value_counts().reset_index(), x='count', y='interior', title='count by interior color', labels={'count': 'count', 'interior': 'interior color'})

        with col1:
            st.plotly_chart(color_count_fig, use_container_width=True)
        with col2:
            st.plotly_chart(interior_color_count_fig, use_container_width=True)

        # condition and state tables in columns
        condition_sell_price_count_df = filtered_df.groupby('condition').agg(average_sell_price=('sell_price', 'mean'), count=('condition', 'size')).reset_index()
        condition_sell_price_count_df = condition_sell_price_count_df.sort_values(by='average_sell_price', ascending=False)

        state_sell_price_df = filtered_df.groupby('state').agg(average_sell_price=('sell_price', 'mean'), count=('state', 'size')).reset_index()
        state_sell_price_df = state_sell_price_df.sort_values(by='average_sell_price', ascending=False).head(5)

        # display condition and state tables side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("average sell_price by condition")
            st.dataframe(condition_sell_price_count_df)

        with col2:
            st.subheader("average sell_price by state")
            st.dataframe(state_sell_price_df)

        # pivot table graph
        pivot_table = filtered_df.pivot_table(values='sell_price', index='make', columns='transmission', aggfunc='mean', fill_value=0)
        pivot_fig = px.imshow(pivot_table, title='average sell_price by make and transmission', labels=dict(x="transmission", y="make", color="average sell_price"))
        st.plotly_chart(pivot_fig)

    # reset filters button
    if st.sidebar.button("reset filters"):
        st.experimental_rerun()

    # download data button
    if st.sidebar.button("download filtered data"):
        filtered_df.to_csv("filtered_data.csv", index=False)
        st.sidebar.success("data downloaded successfully!")

elif page == "Model Prediction":
    # Title
    st.title("Car Price Prediction")

    # Input features for prediction
    year = st.selectbox("year", options=df['year'].unique())
    make = st.selectbox("make", options=df['make'].unique())
    model_input = st.selectbox("model", options=df['model'].unique())
    body = st.selectbox("body", options=df['body'].unique())
    transmission = st.selectbox("transmission", options=df['transmission'].unique())
    State = st.selectbox("state", options=df['state'].unique())
    odometer = st.number_input("odometer", min_value=0, max_value=int(df['odometer'].max()), value=0)
    color = st.selectbox("color", options=df['color'].unique())
    sale_month=st.selectbox("sale_month", options=df['sale_month'].unique())
    condition = st.selectbox("condition", options=df['condition'].unique())

    # Prepare input data for prediction
    input_data = pd.DataFrame({  
        'year': [year],  
        'make': [make],  
        'model': [model_input],  
        'body': [body],  
        'transmission': [transmission],  
        'state': [State],  
        'odometer': [odometer],  
        'color': [color],  
        'sale_month': [sale_month],  
        'condition': [condition]  
    }, index=[0])

    # Predict button
    if st.button("Predict Price"):
        prediction = model.predict(input_data)
        st.success(f"Predicted Price: ${prediction[0]:,.2f}")
