import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Aircraft Engine Predictive Maintenance",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("✈️ Aircraft Engine Predictive Maintenance System")

st.write(
    "Predict Engine Health State using KMeans Clustering"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model_data = pickle.load(
    open("unique.pkl", "rb")
)

kmeans = model_data["kmeans"]
scaler = model_data["scaler"]
pca = model_data["pca"]
sensor_cols = model_data["sensor_cols"]

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

df_train = pd.read_csv("PM_train.csv")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("⚙️ Controls")

# ---------------------------------------------------
# RANDOM SENSOR BUTTON
# ---------------------------------------------------

if st.sidebar.button("🎲 Generate Random Sensors"):

    random_data = {}

    # SETTINGS RANDOM VALUES
    random_data['setting1'] = round(
        np.random.uniform(
            df_train['setting1'].min(),
            df_train['setting1'].max()
        ),
        4
    )

    random_data['setting2'] = round(
        np.random.uniform(
            df_train['setting2'].min(),
            df_train['setting2'].max()
        ),
        4
    )

    random_data['setting3'] = round(
        np.random.uniform(
            df_train['setting3'].min(),
            df_train['setting3'].max()
        ),
        4
    )

    # SENSOR RANDOM VALUES
    for col in sensor_cols:

        min_val = df_train[col].min()
        max_val = df_train[col].max()

        random_data[col] = round(
            np.random.uniform(min_val, max_val),
            4
        )

    # RANDOM CYCLE
    random_data['cycle'] = np.random.randint(1, 350)

    # SAVE SESSION
    st.session_state.random_data = random_data

    # REFRESH PAGE
    st.rerun()

# ---------------------------------------------------
# DEFAULT DATA
# ---------------------------------------------------

default_data = st.session_state.get(
    "random_data",
    {}
)

# ---------------------------------------------------
# INPUT DATA
# ---------------------------------------------------

input_data = {}

# ---------------------------------------------------
# ENGINE SETTINGS
# ---------------------------------------------------

st.markdown("## ⚙️ Engine Settings")

setting_cols = st.columns(4)

input_data['setting1'] = setting_cols[0].number_input(
    "setting1",
    value=float(default_data.get('setting1', 0.0))
)

input_data['setting2'] = setting_cols[1].number_input(
    "setting2",
    value=float(default_data.get('setting2', 0.0))
)

input_data['setting3'] = setting_cols[2].number_input(
    "setting3",
    value=float(default_data.get('setting3', 100.0))
)

input_data['cycle'] = setting_cols[3].number_input(
    "cycle",
    value=int(default_data.get('cycle', 1))
)

# ---------------------------------------------------
# SENSOR VALUES IN SINGLE HORIZONTAL BOX
# ---------------------------------------------------

st.markdown("## 📡 Sensor Values")

# Create one row dataframe
sensor_values = {
    sensor: float(default_data.get(sensor, 0.0))
    for sensor in sensor_cols
}

sensor_df = pd.DataFrame([sensor_values])

# Horizontal editable table
edited_df = st.data_editor(
    sensor_df,
    use_container_width=True,
    hide_index=True,
    key="sensor_editor"
)

# SAVE VALUES
for sensor in sensor_cols:
    input_data[sensor] = edited_df.loc[0, sensor]

# ---------------------------------------------------
# PREDICT BUTTON
# ---------------------------------------------------

if st.button("🚀 Predict Health State"):

    input_df = pd.DataFrame([input_data])

    # SCALE
    input_scaled = scaler.transform(
        input_df[sensor_cols]
    )

    # PCA
    input_pca = pca.transform(
        input_scaled
    )

    # CLUSTER
    pred_cluster = kmeans.predict(
        input_scaled
    )[0]

    # HEALTH STATE
    if pred_cluster == 0:

        health_state = "Healthy"
        alert = "Normal"

    elif pred_cluster == 1:

        health_state = "Warning"
        alert = "Inspection Required"

    else:

        health_state = "Critical"
        alert = "Immediate Maintenance"

    # HEALTH INDEX
    max_cycle = 350

    health_index = round(
        input_data['cycle'] / max_cycle,
        2
    )

    # ---------------------------------------------------
    # METRICS
    # ---------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Health State",
            health_state
        )

    with col2:
        st.metric(
            "Health Index",
            health_index
        )

    with col3:
        st.metric(
            "Alert",
            alert
        )

    # ---------------------------------------------------
    # ALERT MESSAGE
    # ---------------------------------------------------

    if health_state == "Critical":

        st.error(
            "🔴 Immediate Maintenance Required"
        )

    elif health_state == "Warning":

        st.warning(
            "🟡 Inspection Recommended"
        )

    else:

        st.success(
            "🟢 Engine Operating Normally"
        )

    # ---------------------------------------------------
    # GAUGE CHART
    # ---------------------------------------------------

    gauge_value = health_index * 100

    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=gauge_value,

            title={
                'text': "Engine Health Index"
            },

            gauge={

                'axis': {
                    'range': [0, 100]
                },

                'bar': {
                    'color': "black"
                },

                'steps': [

                    {
                        'range': [0, 40],
                        'color': "green"
                    },

                    {
                        'range': [40, 70],
                        'color': "yellow"
                    },

                    {
                        'range': [70, 100],
                        'color': "red"
                    }
                ]
            }
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------------------------------
# CSV PREDICTION
# ---------------------------------------------------

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload CSV",
    type=['csv']
)

if uploaded_file is not None:

    st.subheader("📂 CSV Prediction Results")

    upload_df = pd.read_csv(
        uploaded_file
    )

    upload_scaled = scaler.transform(
        upload_df[sensor_cols]
    )

    upload_pca = pca.transform(
        upload_scaled
    )

    pred_clusters = kmeans.predict(
        upload_pca
    )

    health_states = []
    alerts = []

    for cluster in pred_clusters:

        if cluster == 0:

            health_states.append("Healthy")
            alerts.append("Normal")

        elif cluster == 1:

            health_states.append("Warning")
            alerts.append("Inspection Required")

        else:

            health_states.append("Critical")
            alerts.append("Immediate Maintenance")

    upload_df['health_state'] = health_states
    upload_df['alert'] = alerts

    upload_df['health_index'] = (
        upload_df['cycle'] / 350
    ).round(2)

    st.dataframe(
        upload_df,
        use_container_width=True
    )