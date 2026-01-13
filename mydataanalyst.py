import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Prajwal's AI Analyst - Altair", layout="wide")

# --- 2. CSS STYLING ---
st.markdown("""
<style>
    .metric-container {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    h3 { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

# --- 3. ANALYST ENGINE ---
class DataAnalystAgent:
    def __init__(self, df):
        self.df = df.copy()
        self.cols = {"time": None, "user": None, "rev": None, "dimensions": [], "numeric": []}
        
    def analyze_structure(self):
        """Scans columns to find dimensions and metrics."""
        for col in self.df.columns:
            clean = str(col).lower().strip()
            dtype = self.df[col].dtype
            
            # Time Detection
            if not self.cols["time"] and (any(x in clean for x in ['date', 'time', 'created']) or np.issubdtype(dtype, np.datetime64)):
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    self.cols["time"] = col
                except: pass
            
            # Revenue Detection
            elif not self.cols["rev"] and any(x in clean for x in ['revenue', 'amount', 'price', 'sales', 'cost']) and np.issubdtype(dtype, np.number):
                self.cols["rev"] = col
                self.cols["numeric"].append(col)

            # User ID
            elif not self.cols["user"] and any(x in clean for x in ['user_id', 'customer_id', 'email', 'uid']):
                self.cols["user"] = col
                
            # Numeric Measures (for scatter plots)
            elif np.issubdtype(dtype, np.number):
                self.cols["numeric"].append(col)
                
            # Categorical Dimensions (for slicing)
            elif self.df[col].nunique() < 100 and dtype == 'object':
                self.cols["dimensions"].append(col)
        
        return self

# --- 4. APP INTERFACE ---
st.title("Prajwal's AI Analyst - Altair")
st.markdown("Cheers!")

uploaded_file = st.file_uploader("Upload Data (CSV)", type="csv")

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file, encoding_errors='replace')
        agent = DataAnalystAgent(df_raw).analyze_structure()
        
        # --- SIDEBAR CONTROLS ---
        st.sidebar.header("Slice & Dice")
        
        # 1. Date Filter
        df_filtered = agent.df.copy()
        if agent.cols["time"]:
            min_date = df_filtered[agent.cols["time"]].min()
            max_date = df_filtered[agent.cols["time"]].max()
            date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
            if len(date_range) == 2:
                df_filtered = df_filtered[(df_filtered[agent.cols["time"]] >= pd.to_datetime(date_range[0])) & 
                                          (df_filtered[agent.cols["time"]] <= pd.to_datetime(date_range[1]))]

        # 2. Dimension Filter
        if agent.cols["dimensions"]:
            filter_dim = st.sidebar.selectbox("Filter by Segment:", ["All"] + agent.cols["dimensions"])
            if filter_dim != "All":
                selected_val = st.sidebar.multiselect(f"Select {filter_dim}", df_filtered[filter_dim].unique())
                if selected_val:
                    df_filtered = df_filtered[df_filtered[filter_dim].isin(selected_val)]

        # --- A. TOP METRICS ---
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        
        rows = len(df_filtered)
        rev = df_filtered[agent.cols["rev"]].sum() if agent.cols["rev"] else 0
        users = df_filtered[agent.cols["user"]].nunique() if agent.cols["user"] else 0
        
        m1.metric("Rows Analyzed", f"{rows:,}")
        if agent.cols["rev"]: m2.metric("Total Revenue", f"${rev:,.0f}")
        if agent.cols["user"]: m3.metric("Active Users", f"{users:,}")
        if agent.cols["rev"] and agent.cols["user"] and users > 0: m4.metric("ARPU", f"${rev/users:,.2f}")

        # --- B. VISUALIZATION GRID ---
        st.divider()
        
        # ROW 1: Trends & Breakdown
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("Trends over Time")
            if agent.cols["time"]:
                metric_to_plot = agent.cols["rev"] if agent.cols["rev"] else agent.cols["numeric"][0] if agent.cols["numeric"] else None
                if metric_to_plot:
                    freq = st.select_slider("Granularity", options=["D", "W", "M"], value="M")
                    trend = df_filtered.set_index(agent.cols["time"]).resample(freq)[metric_to_plot].sum().reset_index()
                    fig_trend = px.area(trend, x=agent.cols["time"], y=metric_to_plot, title=f"{metric_to_plot} Trend")
                    st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("No Time Column Found")

        with c2:
            st.subheader("Slice by Category")
            if agent.cols["dimensions"]:
                slice_col = st.selectbox("Group By:", agent.cols["dimensions"], index=0)
                metric_col = agent.cols["rev"] if agent.cols["rev"] else "Count"
                
                if metric_col == "Count":
                    pie_data = df_filtered[slice_col].value_counts().reset_index().head(10)
                    pie_data.columns = [slice_col, "Count"]
                    fig_pie = px.pie(pie_data, names=slice_col, values="Count", hole=0.4)
                else:
                    pie_data = df_filtered.groupby(slice_col)[metric_col].sum().reset_index().sort_values(metric_col, ascending=False).head(10)
                    fig_pie = px.pie(pie_data, names=slice_col, values=metric_col, hole=0.4)
                
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No Categories Found")

        # ROW 2: Deep Dive (Bar & Box)
        c3, c4 = st.columns(2)
        
        with c3:
            st.subheader("Ranking Top Segments")
            if agent.cols["dimensions"]:
                x_col = slice_col # Reuse the dropdown from above
                y_col = agent.cols["rev"] if agent.cols["rev"] else agent.cols["numeric"][0]
                
                bar_data = df_filtered.groupby(x_col)[y_col].sum().reset_index().sort_values(y_col, ascending=False).head(15)
                fig_bar = px.bar(bar_data, x=x_col, y=y_col, color=y_col, title=f"Top {x_col} by {y_col}")
                st.plotly_chart(fig_bar, use_container_width=True)

        with c4:
            st.subheader("Distribution & Outliers")
            if agent.cols["numeric"]:
                dist_col = st.selectbox("Select Metric for Distribution:", agent.cols["numeric"])
                fig_box = px.box(df_filtered, y=dist_col, title=f"Distribution of {dist_col}")
                st.plotly_chart(fig_box, use_container_width=True)

        # --- C. CORRELATION LAB ---
        st.divider()
        st.subheader("Correlation Lab")
        if len(agent.cols["numeric"]) > 1:
            col_x, col_y, col_color = st.columns(3)
            x_axis = col_x.selectbox("X-Axis", agent.cols["numeric"], index=0)
            y_axis = col_y.selectbox("Y-Axis", agent.cols["numeric"], index=1 if len(agent.cols["numeric"]) > 1 else 0)
            color_dim = col_color.selectbox("Color By (Optional)", ["None"] + agent.cols["dimensions"])
            
            color_arg = None if color_dim == "None" else color_dim
            fig_scatter = px.scatter(df_filtered, x=x_axis, y=y_axis, color=color_arg, title=f"{x_axis} vs {y_axis}")
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis.")

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Upload CSV to enable interactive slicing.")
