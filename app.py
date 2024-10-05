import streamlit as st 
import pandas as pd
import plotly.express as px

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_car_prices.csv")

df = load_data()

df2=df.copy()
df2=df2.columns = df2.columns.str.strip().str.capitalize()
df2=df2.rename(columns={'Odometer': 'Mileage', 'SELLINGPRICE': 'Price'}, inplace=True)



# Title
st.title("Car Price Dashboard")

# Sidebar for filters
st.sidebar.header("Filters")

# Car make selection with "All" as default
all_makes = df2['Make'].unique().tolist()
selected_makes = st.sidebar.multiselect(
    "Select Car Makes",
    options=["All"] + all_makes,
    default=["All"]
)

# Filter the DataFrame based on selected makes
if "All" in selected_makes:
    filtered_df2 = df2
else:
    filtered_df2 = df2[df2['Make'].isin(selected_makes)]

# Car model selection based on make
all_models = filtered_df2['Model'].unique().tolist()
selected_models = st.sidebar.multiselect(
    "Select Car Models",
    options=["All"] + all_models,
    default=["All"]
)

# Filter the DataFrame based on selected models
if "All" in selected_models:
    filtered_df2 = filtered_df2
else:
    filtered_df2 = filtered_df2[filtered_df2['Model'].isin(selected_models)]

# Additional Filters
transmissions = df2['Transmission'].unique().tolist()
selected_transmissions = st.sidebar.multiselect("Select Transmission Types", options=["All"] + transmissions, default=["All"])

if "All" not in selected_transmissions:
    filtered_df2 = filtered_df2[filtered_df2['Transmission'].isin(selected_transmissions)]

states = df2['State'].unique().tolist()
selected_states = st.sidebar.multiselect("Select States", options=["All"] + states, default=["All"])

if "All" not in selected_states:
    filtered_df2 = filtered_df2[filtered_df2['State'].isin(selected_states)]

Mileage_range = st.sidebar.slider(
    "Select Mileage Range",
    min_value=int(df2['Mileage'].min()),
    max_value=int(df2['Mileage'].max()),
    value=(int(df2['Mileage'].min()), int(df2['Mileage'].max()))
)

filtered_df2 = filtered_df2[(filtered_df2['Mileage'] >= Mileage_range[0]) & (filtered_df2['Mileage'] <= Mileage_range[1])]

# Year and Price sliders
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=int(df2['Year'].min()),
    max_value=int(df2['Year'].max()),
    value=(int(df2['Year'].min()), int(df2['Year'].max()))
)

filtered_df2 = filtered_df2[(filtered_df2['Year'] >= year_range[0]) & (filtered_df2['Year'] <= year_range[1])]

price_range = st.sidebar.slider(
    "Select Price Range",
    min_value=int(df2['Price'].min()),
    max_value=int(df2['Price'].max()),
    value=(int(df2['Price'].min()), int(df2['Price'].max()))
)

filtered_df2 = filtered_df2[(filtered_df2['Price'] >= price_range[0]) & (filtered_df2['Price'] <= price_range[1])]

# KPIs
make_count = filtered_df2['Make'].nunique()
model_count = filtered_df2['Model'].nunique()
average_price = filtered_df2['Price'].mean()

# Display KPIs in one row
st.subheader("Key Performance Indicators")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Unique Makes", make_count)
with col2:
    st.metric("Unique Models", model_count)
with col3:
    st.metric("Average Price", f"${average_price:,.2f}")

# Descriptive Statistics
st.subheader("Descriptive Statistics")
desc_stats = filtered_df2[['Price', 'Mileage']].describe()
html_table = desc_stats.to_html(classes='table table-striped', justify='center', border=0)

# Center the table using Markdown
st.markdown(
    f"<div style='text-align: center;'>{html_table}</div>",
    unsafe_allow_html=True
)

# Graph type selection
graph_type = st.sidebar.selectbox(
    "Select Graph Type",
    options=['Bar', 'Count Bar', 'Line', 'Bi_axes_line', 'Scatter', 'Pie']
)

# Graph generation
if filtered_df2.empty:
    st.warning("No data available for the selected filters.")
else:
    if graph_type == 'Bar':
        fig = px.bar(filtered_df2.groupby('Make', as_index=False).agg({'Price': 'mean'}), x='Make', y='Price', title='Average Price by Make')
    elif graph_type == 'Count Bar':
        fig = px.bar(filtered_df2['Make'].value_counts().reset_index(), x='Make', y='count', title='Cars Count by Make')
    elif graph_type == 'Line':
        fig = px.line(filtered_df2.groupby('Year').agg({'Price': 'mean'}).reset_index(), x='Year', y='Price', title='Average Price Over Years')
    elif graph_type == 'Bi_axes_line':
        fig = px.line(filtered_df2.groupby('Year').agg(Avg_price=('Price', 'mean'), Avg_mmr=('Mmr', 'mean')).reset_index(),
                      x='Year', y=['Avg_price', 'Avg_mmr'], markers=True, title='Average Price & MMR Over Years')
    elif graph_type == 'Pie':
        fig = px.pie(filtered_df2['Transmission'].value_counts().reset_index(), names='Transmission', values='count', title='Transmission Distribution')
    else:
        fig = px.scatter(filtered_df2, x='Mileage', y='Price', color='Make', title='Mileage vs Price')

    st.plotly_chart(fig)

    # Colors Graphs
    col1, col2 = st.columns(2)
    color_count_fig = px.bar(filtered_df2['Color'].value_counts().reset_index(), x='count', y='Color', title='Count by Car Color', labels={'count': 'Count', 'Color': 'Color'})
    interior_color_count_fig = px.bar(filtered_df2['Interior'].value_counts().reset_index(), x='count', y='Interior', title='Count by Interior Color', labels={'count': 'Count', 'Interior': 'Interior Color'})

    with col1:
        st.plotly_chart(color_count_fig, use_container_width=True)
    with col2:
        st.plotly_chart(interior_color_count_fig, use_container_width=True)

    # Condition and State Tables in columns
    condition_price_count_df2 = filtered_df2.groupby('Tight_condition').agg(Average_Price=('Price', 'mean'), Count=('Tight_condition', 'size')).reset_index()
    condition_price_count_df2 = condition_price_count_df2.sort_values(by='Average_Price', ascending=False)

    state_price_df2 = filtered_df2.groupby('State').agg(Average_Price=('Price', 'mean'), Count=('State', 'size')).reset_index()
    state_price_df2 = state_price_df2.sort_values(by='Average_Price', ascending=False).head(5)

    # Display Condition and State Tables side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Average Price by Condition")
        st.dataframe(condition_price_count_df2)

    with col2:
        st.subheader("Average Price by State")
        st.dataframe(state_price_df2)

    # Pivot Table Graph
    pivot_table = filtered_df2.pivot_table(values='Price', index='Make', columns='Transmission', aggfunc='mean', fill_value=0)
    pivot_fig = px.imshow(pivot_table, title='Average Price by Make and Transmission', labels=dict(x="Transmission", y="Make", color="Average Price"))
    st.plotly_chart(pivot_fig)

# Reset Filters Button
if st.sidebar.button("Reset Filters"):
    st.experimental_rerun()

# Download Data Button
if st.sidebar.button("Download Filtered Data"):
    filtered_df2.to_csv("filtered_data.csv", index=False)
    st.sidebar.success("Data downloaded successfully!")
