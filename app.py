# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import pyodbc
import pandas as pd
import streamlit as st
from itertools import combinations
from collections import Counter
import plotly.express as px
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mticker

# theme colour

def set_dark_theme():
    st.markdown("""
        <style>
        /* 🌙 App background and base text */
        body, .stApp {
            background-color: #000000;
            color: #FFFFFF;
        }

        /* 📦 Sidebar background */
        section[data-testid="stSidebar"] {
            background-color: #1e1e1e !important;
        }

        /* 📝 Sidebar text */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4,
        section[data-testid="stSidebar"] h5,
        section[data-testid="stSidebar"] h6,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div,
        div[data-baseweb="radio"] > div {
            color: #FFFFFF !important;
        }

        /* 🔘 Sidebar buttons */
        section[data-testid="stSidebar"] button {
            background-color: #999999 !important;
            color: #ffffff !important;
            font-weight: bold !important;
            border-radius: 8px !important;
            padding: 10px 20px !important;
            border: none !important;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        section[data-testid="stSidebar"] button span {
            color: #ffffff !important;
            font-weight: bold !important;
        }

        /* 📌 Sidebar layout for sticky logout */
        [data-testid="stSidebar"] > div:first-child {
            display: flex;
            flex-direction: column;
            height: 100vh;
        }

        .logout-container {
            margin-top: auto;
        }
        </style>
    """, unsafe_allow_html=True)

# kpi theme

def custom_kpi(label, value, delta=None, value_color="#ffffff", delta_color="#00ff99"):
    st.markdown(f"""
        <div style="padding: 15px 20px; border-radius: 12px; background-color: #1e1e1e;
                    color: white; border: 1px solid #333; margin-bottom: 10px;">
            <div style="font-size: 14px; font-weight: 500; color: #cccccc;">{label}</div>
            <div style="font-size: 24px; font-weight: bold; color: {value_color};">{value}</div>
            {f'<div style="font-size: 12px; color: {delta_color};">{delta}</div>' if delta else ''}
        </div>
    """, unsafe_allow_html=True)


# USER LOGIN CREDENTIALS
# --------------------------
Users = {
    'admin': '1234' }
# ------------------------
# 1. Load & preprocess data
# ------------------------
@st.cache_data
def load_data():
    df= pd.read_csv("sqldata.csv")
    
    return df

def login_page():
    set_dark_theme()

    st.markdown("""
        <style>
        .stButton > button {
            background-color: #444;
            color: white;
            border-radius: 8px;
            padding: 0.5em 2em;
            transition: 0.3s;
            border: none;
        }

        .stButton > button:hover {
            background-color: #6c63ff;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            "<h4 style='text-align:center; color:#ccc;'>Welcome to the Analytics Dashboard</h4>",
            unsafe_allow_html=True
        )
        st.title("🔐 Login")

        # ✅ Assigning unique keys to all widgets
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        # Track if login was attempted
        if "login_attempted" not in st.session_state:
            st.session_state.login_attempted = False

        if st.button("Login", key="login_button"):
            st.session_state.login_attempted = True
            if username in Users and Users[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.page = "intro"
                st.success("Login Successful")
                st.rerun()

        # Show error only after login attempt
        if st.session_state.login_attempted and not st.session_state.logged_in:
            st.error("Invalid Username or Password", icon="🚫")

# --------------------------
# PROJECT INTRO PAGE
# --------------------------
def project_intro():
    set_dark_theme()
    st.title("📘 VODAFONE QATAR ANALYSIS")
    
    st.markdown("""
        <style>
        div.stButton > button {
            background-color: #0099FF;
            color: white;
            font-size: 16px;
            padding: 10px 30px;
            border: none;
            border-radius: 8px;
            transition: 0.3s;
        }
        div.stButton > button:hover {
            background-color: #007ACC;
        }
        </style>
    """, unsafe_allow_html=True)

    # Functional button
    if st.button("👉 Continue to Business Context", key="to_business_context_btn"):
        st.session_state.page = "business"
        st.rerun()

# --------------------------
# BUSINESS CONTEXT PAGE
# --------------------------
def business_context():
    set_dark_theme()
    # Widen content area and style the button
    st.markdown("""
        <style>
        .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 90% !important;
        }
        div.stButton > button {
            background-color: #0099FF;
            color: white;
            font-size: 16px;
            padding: 10px 30px;
            border: none;
            border-radius: 8px;
            transition: 0.3s;
            display: inline-flex;
            white-space: nowrap;
        }
        div.stButton > button:hover {
            background-color: #007ACC;
        }
        </style>
    """, unsafe_allow_html=True)

# Image from direct URL (centered)
    img_col1, img_col2, img_col3 = st.columns([2, 3, 1])
    with img_col2:
        st.image("https://www.opentext.com/assets/images/resources/customer-success/vodafone-qatar-logo.jpg", width=300)

    # Title
    st.markdown("""
        <h1 style='text-align:center; font-size:36px;'>💼 Business Context</h1>
    """, unsafe_allow_html=True)

    # Business context paragraph (wide and centered)
    text_col1, text_col2, text_col3 = st.columns([1, 6, 1])
    with text_col2:
        st.markdown("""
            <p style='text-align:justify; font-size:18px; color:#CCCCCC;'>
               Vodafone Qatar is a major telecom provider offering mobile, broadband, IoT, managed services, and devices. 
               The company is in the middle of a digital transformation expanding 5G, boosting digital adoption, and focusing on customer experience. Vodafone Qatar serves thousands of customers in a fast-changing telecom market.
               Main goals: Keep customers happy, grow revenue, and maintain strong network performance.
              <br><br>
               This project aims to analyze Vodafone Qatar's data to find patterns and forecast future trends.
               By using advanced analytics and machine learning, we want to help Vodafone Qatar make better decisions, 
               improve customer satisfaction, and stay competitive in the telecom market.
            </p>
        """, unsafe_allow_html=True)

    # button
    btn_col1, btn_col2, btn_col3 = st.columns([4, 2, 4])
    with btn_col2:
        if st.button("👉 Continue to Dashboard", key="continue_to_dashboard_btn"):
            st.session_state.page = "dashboard"
            st.rerun()

# --------------------------
# DASHBOARD PAGE
# --------------------------
def dashboard():
    set_dark_theme()
    #test_db_connection()

    df = load_data()
    st.sidebar.title("📂 Navigation")
    selected_page = st.sidebar.radio("Go to", ["KPIs", "Revenue & ARPU Analysis", "Subscriber Analysis", "Churn Analysis", "Customer Satisfaction (CSI)", "Data Usage Analysis", "Complaints & Service Quality","Network Availability Analysis","Retail & Recharge Analysis","Voice Usage & eSIM Activations"])

    st.sidebar.markdown("---")
    st.sidebar.title("🔍 Filters")
    # Normalize column names (optional, keeps things consistent)
    df.columns = df.columns.str.strip()

    df["Year"] = df["Year"].astype(int)
    df["Quarter"] = df["Quarter"].astype(str)
    df["Region"] = df["Region"].astype(str)
    df["Segment"] = df["Segment"].astype(str)

    # ---- Sidebar global filters ----
    years = sorted(df["Year"].dropna().unique())
    quarters = sorted(df["Quarter"].dropna().unique())
    regions = sorted(df["Region"].dropna().unique())
    segments = sorted(df["Segment"].dropna().unique())

    selected_years = st.sidebar.multiselect("Year", options=years, default=years)
    selected_quarters = st.sidebar.multiselect("Quarter", options=quarters, default=quarters)
    selected_regions = st.sidebar.multiselect("Region", options=regions, default=regions)
    selected_segments = st.sidebar.multiselect("Segment", options=segments, default=segments)

    # ---- Base filter condition ----
    base_filter = (
        df["Year"].isin(selected_years) &
        df["Quarter"].isin(selected_quarters) &
        df["Region"].isin(selected_regions) &
        df["Segment"].isin(selected_segments)
    )


#---------------------page specific filters--------------------

    if selected_page == "Subscriber Analysis":
        df_filtered = df[base_filter]

    elif selected_page == "Data Usage Analysis":
        df_filtered = df[base_filter] 

    elif selected_page == "Voice Usage & eSIM Activations": 
        df_filtered = df[base_filter]  

    elif selected_page == "Retail & Recharge Analysis":  
        df_filtered = df[base_filter]   

    elif selected_page == "Network Availability Analysis":  
        df_filtered = df[base_filter]  

    elif selected_page == "Complaints & Service Quality":
        df_filtered = df[base_filter]    

    elif selected_page == "Revenue & ARPU Analysis":  
        df_filtered = df[base_filter]     

    elif selected_page == "Churn Analysis":
        df_filtered = df[base_filter] 

    elif selected_page == "Customer Satisfaction (CSI)":
        df_filtered = df[base_filter]  

    elif selected_page == "Revenue & ARPU Analysis":  
        df_filtered = df[base_filter]   

    elif selected_page == "KPIs": 
        df_filtered = df[base_filter]       






#log out button block

    st.sidebar.markdown('<div class="logout-container">', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 👤 Logged in as: `{st.session_state.username}`")

    if st.sidebar.button("🚪 Logout" , key="logout_button"):
        st.session_state.logged_in = False
        st.rerun()

    st.sidebar.markdown('</div>', unsafe_allow_html=True)

  #----------------------  KPI'S  ---------------------------------


    if selected_page == "KPIs":
        st.markdown(
    """
    <h1 style='text-align: center; text-decoration: underline; font-weight: bold;font-size: 51px; margin-bottom: 30px;'>
        Key Performance Indicators
    </h1>
    """,
    unsafe_allow_html=True
)   
        
# Calculate KPIs (replace column names with actual names in your dataset)
        total_revenue = (df_filtered["Revenue_QR_Mn"] * 1_000_000).sum()  # convert to QAR
        Total_Revenue = total_revenue / 1_000_000_000 
        
        total_subscribers = df_filtered["Subscribers"].sum()
        Total_Subscribers = total_subscribers / 1_000_000  # convert to millions

        arpu = Total_Revenue/Total_Subscribers
        
        churn_rate = (df_filtered["Churn_Rate_Pct"].mean())

        avg_csi = df_filtered["Customer_Satisfaction_Index"].mean()

        total_complaints = df_filtered["Complaints_Resolved_Pct"].mean()

        network_uptime = df_filtered["Network_Availability_Pct"].mean()

        total_esim = df_filtered["eSIM_Activations"].sum()
        total_esimm = total_esim / 1_000_000  # convert to millions

        data_usage = df_filtered["Data_Usage_GB"].mean()

        voice_usage = df_filtered["Voice_Minutes_Mn"].mean()

        New_Activations = df_filtered["New_Activations"].sum()
        activations = New_Activations / 1_000_000  # convert to millions

        Deactivations = df_filtered["Deactivations"].sum()
        deactivations = Deactivations / 1_000_000  # convert to millions

# =============================
    # 📈 SECTION 1: Core Metrics
    # =============================
        st.divider()
        st.markdown("### 💼 Core Business Metrics")

        row1_col1, row1_col2, row1_col3 = st.columns(3)
        with row1_col1:
            custom_kpi("💰 Total Revenue", f"{Total_Revenue:,.2f} bn QAR")
        with row1_col2:
            custom_kpi("👥 Total Subscribers", f"{Total_Subscribers:,.2f} M")
        with row1_col3:
            custom_kpi("💸 ARPU", f"{arpu:,.2f} k QAR")


    # =============================
    # 📉 SECTION 2: Customer Metrics
    # =============================
        st.divider()
        st.markdown("### 📉 Customer Metrics")

        row2_col1, row2_col2, row2_col3 = st.columns(3)
        with row2_col1:
            custom_kpi("🔁 Churn Rate", f"{churn_rate:,.2f}%")
        with row2_col2:
            custom_kpi("⭐ Avg CSI", f"{avg_csi:,.2f}")
        with row2_col3:
            custom_kpi("📩 Complaints Resolved", f"{total_complaints:,.2f}%")


    # =============================
    # 📡 SECTION 3: Network & Usage
    # =============================
        st.divider()
        st.markdown("### 📡 Network & Usage")

        row3_col1, row3_col2, row3_col3 = st.columns(3)
        with row3_col1:
            custom_kpi("📶 Network Availability", f"{network_uptime:,.2f}%")
        with row3_col2:
            custom_kpi("📱 eSIM Activations", f"{total_esimm:,.2f} M")
        with row3_col3:
            custom_kpi("💾 Avg Data Usage", f"{data_usage:,.2f} GB")

        row4_col1, row4_col2, row4_col3 = st.columns(3)
        with row4_col1:
            custom_kpi("📞 Avg Voice Usage", f"{voice_usage:,.2f} Min")
        with row4_col2:
            custom_kpi("🆕 New Activations", f"{activations:,.2f} M")
        with row4_col3:
            custom_kpi("📴 Deactivations", f"{deactivations:,.2f} M")   
     
#---------Revenue & ARPU Analysis--------- #

    elif selected_page == "Revenue & ARPU Analysis":
        st.markdown(
        """
        <h1 style='text-align: center; text-decoration: underline; font-weight: bold;font-size: 51px; margin-bottom: 30px;'>
            Revenue & ARPU Analysis
        </h1>
        """,
        unsafe_allow_html=True
    )

    # 1. Revenue by Region
        reg_rev = df_filtered.groupby('Region')['Revenue_QR_Mn'].sum().reset_index()
        st.subheader("Revenue by Region")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(
        data=reg_rev, 
        x='Region', 
        y='Revenue_QR_Mn', 
        palette="viridis",
        ax=ax
    )

        # Format y-axis in thousands with decimals (K)       
    
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000000000:.0f}bn"))

    # Add data labels
        for i, row in reg_rev.iterrows():
            ax.text(i, row['Revenue_QR_Mn'], f"{row['Revenue_QR_Mn']/1000000000:.3f}bn",
                ha='center', va='bottom', fontsize=9)

        ax.set_title("Total Revenue by Region")
        ax.set_xlabel("Region")
        ax.set_ylabel("Revenue (in K)")
        
        st.pyplot(fig)


    # 2. ARPU by Region
        seg_rev = df_filtered.groupby('Region')['ARPU_QR'].sum().reset_index()
        st.subheader("ARPU by Region")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(
        data=seg_rev, 
        x='Region', 
        y='ARPU_QR', 
        palette="viridis",
        ax=ax
    )
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
        for i, row in seg_rev.iterrows():
            ax.text(i, row['ARPU_QR'], f"{row['ARPU_QR']/1000:.3f}K",
                ha='center', va='bottom', fontsize=9)

        ax.set_title("ARPU by Region")
        ax.set_xlabel("Region")
        ax.set_ylabel("ARPU_QR (in K)")

        st.pyplot(fig)

    # 3. Revenue Trend over Years
        rev_year = df_filtered.groupby('Year')['Revenue_QR_Mn'].sum().reset_index()
        st.subheader("Revenue Trend over Years")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.lineplot(
        data=rev_year, 
        x='Year', 
        y='Revenue_QR_Mn', 
        marker='o', 
        linewidth=2,
        ax=ax
    )

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000000000:.0f}bn"))
        for x, y in zip(rev_year['Year'], rev_year['Revenue_QR_Mn']):
            ax.text(x, y, f"{y/1000000000:.3f}bn", ha='center', va='bottom', fontsize=9)

        ax.set_title("Revenue Trend Over Years")
        ax.set_xlabel("Year")
        ax.set_ylabel("Revenue (in K)")
        ax.set_xticks(rev_year['Year'])
        ax.set_ylim(0, rev_year['Revenue_QR_Mn'].max() * 1.2)

        st.pyplot(fig)


    # 4. ARPU Trend over Years
        rev_year = df_filtered.groupby('Year')['ARPU_QR'].sum().reset_index()
        st.subheader("ARPU Trend over Years")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.lineplot(
        data=rev_year, 
        x='Year', 
        y='ARPU_QR', 
        marker='o', 
        linewidth=2,
        ax=ax
    )

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.1f}"))
        for x, y in zip(rev_year['Year'], rev_year['ARPU_QR']):
            ax.text(x, y, f"{y/1000:.3f}", ha='center', va='bottom', fontsize=9)

        ax.set_title("ARPU Trend Over Years")
        ax.set_xlabel("Year")
        ax.set_ylabel("ARPU")
        ax.set_xticks(rev_year['Year'])
        ax.set_ylim(0, rev_year['ARPU_QR'].max() * 1.2)
  
        st.pyplot(fig)


    # 5. Revenue by Segment
        seg_rev = df_filtered.groupby('Segment')['Revenue_QR_Mn'].sum().reset_index()
        st.subheader("Revenue by Segment")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(
        data=seg_rev, 
        x='Segment', 
        y='Revenue_QR_Mn', 
        palette="viridis",
        ax=ax
    )

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000000000:.0f}bn"))
        for i, row in seg_rev.iterrows():
            ax.text(i, row['Revenue_QR_Mn'], f"{row['Revenue_QR_Mn']/1000000000:.3f}bn", ha='center', va='bottom', fontsize=9)

        ax.set_title("Total Revenue by Segment")
        ax.set_xlabel("Segment")
        ax.set_ylabel("Revenue (in K)")

        st.pyplot(fig)


    # 6. ARPU by Segment
        seg_rev = df_filtered.groupby('Segment')['ARPU_QR'].sum().reset_index()
        st.subheader("ARPU by Segment")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(
        data=seg_rev, 
        x='Segment', 
        y='ARPU_QR', 
        palette="viridis",
        ax=ax
    )

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
        for i, row in seg_rev.iterrows():
            ax.text(i, row['ARPU_QR'], f"{row['ARPU_QR']/1000:.3f}K", ha='center', va='bottom', fontsize=9)

        ax.set_title("ARPU by Segment")
        ax.set_xlabel("Segment")
        ax.set_ylabel("ARPU_QR (in K)")

        st.pyplot(fig)


    #7. ARPU by Quarters
        arpu_quarter = df_filtered.groupby(['Year','Quarter'])['ARPU_QR'].mean().reset_index()
        st.subheader("ARPU by Quarters")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.lineplot(
        data=arpu_quarter,
        x='Quarter',
        y='ARPU_QR',
        hue='Year',
        marker="o",
        ax=ax
    )

        for year in arpu_quarter['Year'].unique():
            year_df = arpu_quarter[arpu_quarter['Year']==year]
        for x, y in zip(year_df['Quarter'], year_df['ARPU_QR']):
            ax.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=8)

        ax.set_title("ARPU by Quarters")
        ax.set_xlabel("Quarter")
        ax.set_ylabel("ARPU in QR")

        st.pyplot(fig)


    # 8. Revenue by Quarters
        rev_quarter = df_filtered.groupby(['Year','Quarter'])['Revenue_QR_Mn'].mean().reset_index()
        st.subheader("Revenue by Quarters")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.lineplot(
        data=rev_quarter,
        x='Quarter',
        y='Revenue_QR_Mn',
        hue='Year',
        marker="o",
        ax=ax
    )

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}M"))
        for year in rev_quarter['Year'].unique():
            year_df = rev_quarter[rev_quarter['Year']==year]
        for x, y in zip(year_df['Quarter'], year_df['Revenue_QR_Mn']):
            ax.text(x, y, f"{y:.3f}M", ha='center', va='bottom', fontsize=8)

        ax.set_title("Revenue by Quarters")
        ax.set_xlabel("Quarter")
        ax.set_ylabel("Revenue (in Millions)")
        ax.set_ylim(0, rev_quarter['Revenue_QR_Mn'].max() * 1.2)

        st.pyplot(fig)

        st.markdown("<br><br>", unsafe_allow_html=True)

    # --------- Subscriber Analysis --------- #
    
    elif selected_page == "Subscriber Analysis":
        st.markdown(
        """
        <h1 style='text-align: center; text-decoration: underline; font-weight: bold;font-size: 51px; margin-bottom: 30px;'>
            Subscriber Analysis
        </h1>
        """,
        unsafe_allow_html=True
    )

   
    # 1. Subscribers by Region

        subs_region = df_filtered.groupby('Region')['Subscribers'].sum().reset_index()
        subs_region['Subscribers_M'] = subs_region['Subscribers'] / 1_000_000  # convert to millions
        st.subheader("Subscribers by Region")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(
        data=subs_region, 
        x='Region', 
        y='Subscribers_M', 
        palette='crest',
        ax=ax
    )

    # Add data labels in Millions
        for i, row in subs_region.iterrows():
            ax.text(i, row['Subscribers_M'], f"{row['Subscribers_M']:.3f}M",
                ha='center', va='bottom', fontsize=10)

        ax.set_title("Total Subscribers by Region")
        ax.set_xlabel("Region")
        ax.set_ylabel("Subscribers (in Millions)")
        ax.set_xticklabels(subs_region['Region'], rotation=45)

    # Sensible y-axis range
        ymax = subs_region['Subscribers_M'].max()
        ax.set_ylim(0, ymax * 1.2)

        st.pyplot(fig)

    # 2. Subscribers by Segment

    # Group by Segment
        subscribers_by_segment = df_filtered.groupby("Segment")["Subscribers"].sum().reset_index()

    # Sort for better visualization
        subscribers_by_segment = subscribers_by_segment.sort_values("Subscribers", ascending=False)

    # Define a color palette (auto-adjusts if more segments are added)
        colors = plt.cm.tab20.colors[:len(subscribers_by_segment)]
        st.subheader("Subscribers by Segment")
        fig, ax = plt.subplots(figsize=(10,6))
        bars = ax.bar(
        subscribers_by_segment["Segment"],
        subscribers_by_segment["Subscribers"],
        color=colors
    )

        ax.set_title("Total Subscribers by Segment")
        ax.set_xlabel("Segment")
        ax.set_ylabel("Subscribers (in Millions)")
        ax.set_xticklabels(subscribers_by_segment["Segment"], rotation=45)

    # Format y-axis in millions
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1_000_000)}'))

    # Add data labels in Millions
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f"{height/1_000_000:.2f}M",
                ha="center", va="bottom", fontsize=9)

        st.pyplot(fig)

    # 3. Subscribers by Quarter

    # Group subscribers by Year & Quarter
        subs_quarter = (
        df_filtered.groupby(['Year','Quarter'])['Subscribers']
          .sum()
          .reset_index()
    )

    # Combine Year + Quarter for x-axis label (like 2022-Q1)
        subs_quarter['YearQuarter'] = subs_quarter['Year'].astype(str) + '-' + subs_quarter['Quarter']
        st.subheader("Subscribers by Quarter")
        fig, ax = plt.subplots(figsize=(12,6))
        sns.lineplot(
        data=subs_quarter,
        x='YearQuarter',
        y='Subscribers',
        marker='o',
        ax=ax
    )

    # Add data labels in Millions
        for x, y in zip(subs_quarter['YearQuarter'], subs_quarter['Subscribers']):
            ax.text(x, y, f"{y/1e6:.2f}M", ha='center', va='bottom', fontsize=9)

        ax.set_title("Total Subscribers by Quarter")
        ax.set_xlabel("Quarter")
        ax.set_ylabel("Subscribers (in Millions)")
        ax.set_xticklabels(subs_quarter['YearQuarter'], rotation=45)

        st.pyplot(fig)

        st.markdown("<br><br>", unsafe_allow_html=True)

    # --------- Churn Analysis --------- #
    elif selected_page == "Churn Analysis":
        st.markdown(
            """
            <h1 style='text-align: center; text-decoration: underline; font-weight: bold;font-size: 51px; margin-bottom: 30px;'>
                Churn Analysis
            </h1>
            """,
            unsafe_allow_html=True
        )
    
    
    # 1. Churn Rate Over Years (by Quarter)

    # Group by Year & Quarter → average churn
        churn_quarter = (
        df_filtered.groupby(['Year','Quarter'])['Churn_Rate_Pct']
          .mean()
          .reset_index()
    )

    # Unique years
        years = sorted(churn_quarter['Year'].unique())

        st.subheader("Churn Rate Over Years")
    # Create subplots - one for each year
        fig, axes = plt.subplots(len(years), 1, figsize=(10, 5*len(years)), sharex=True)

    # If only one subplot, wrap in list for iteration
        if len(years) == 1:
            axes = [axes]
            
        for i, year in enumerate(years):
            ax = axes[i]
            year_df = churn_quarter[churn_quarter['Year'] == year]
        
        sns.lineplot(
            data=year_df,
            x='Quarter',
            y='Churn_Rate_Pct',
            marker="o",
            ax=ax
        )
        
        # Data labels
        for x, y in zip(year_df['Quarter'], year_df['Churn_Rate_Pct']):
            ax.text(x, y, f"{y:.3f}%", ha='center', va='bottom', fontsize=8)
        
        ax.set_title(f"Average Churn Rate by Quarter - {year}")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_ylim(0, churn_quarter['Churn_Rate_Pct'].max() * 1.2)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

        axes[-1].set_xlabel("Quarter")

        plt.tight_layout()
        st.pyplot(fig)


    # 2. Churn Rate by Region

    # Group by Region → average churn
        churn_region = (
        df_filtered.groupby('Region')['Churn_Rate_Pct']
          .mean()
          .reset_index()
    )
        
        st.subheader("Churn Rate by Region")
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(
        data=churn_region,
        x='Region',
        y='Churn_Rate_Pct',
        palette="viridis",
        ax=ax
    )

    # Add data labels
        for i, row in churn_region.iterrows():
            ax.text(i, row['Churn_Rate_Pct'] + 0.2, f"{row['Churn_Rate_Pct']:.3f}%",
                ha='center', va='bottom', fontsize=9)
            
        ax.set_title("Average Churn Rate by Region")
        ax.set_xlabel("Region")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_ylim(0, churn_region['Churn_Rate_Pct'].max() * 1.3)
        st.pyplot(fig)

    # 3. Churn Rate by Segment
    # Group by Segment and calculate average churn rate
        avg_churn_by_segment = (
        df_filtered.groupby("Segment")["Churn_Rate_Pct"]
          .mean()
          .reset_index()
          .sort_values("Churn_Rate_Pct", ascending=False)
    )

        st.subheader("📊 Average Churn Rate by Segment")

    # Create figure
        fig, ax = plt.subplots(figsize=(10,6))
        bars = ax.bar(
        avg_churn_by_segment["Segment"], 
        avg_churn_by_segment["Churn_Rate_Pct"], 
        color=plt.cm.Set2.colors
    )

        ax.set_title("Average Churn Rate by Segment")
        ax.set_xlabel("Segment")
        ax.set_ylabel("Avg Churn Rate (%)")
        ax.set_xticklabels(avg_churn_by_segment["Segment"], rotation=45)

    # Add data labels above bars
        for bar in bars:
            height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2, 
            height + 0.2,
            f"{height:.3f}%", 
            ha="center", va="bottom", 
            fontsize=9
        )

        fig.tight_layout()
        st.pyplot(fig)
   
        st.markdown("<br><br>", unsafe_allow_html=True)

    # --------- Customer Satisfaction (CSI) --------- #
    elif selected_page == "Customer Satisfaction (CSI)":
        st.markdown(
            """
            <h1 style='text-align: center; text-decoration: underline; font-weight: bold;font-size: 51px; margin-bottom: 30px;'>
                Customer Satisfaction (CSI)
            </h1>
            """,
            unsafe_allow_html=True
        )
    


    # 1.  Avg CSI by Year
        avg_csi_by_year = (
    df_filtered.groupby("Year")["Customer_Satisfaction_Index"]
      .mean()
      .reset_index()
      .sort_values("Year")
)   
        st.subheader("Average CSI by Year")
        fig1, ax1 = plt.subplots(figsize=(10,6))
        ax1.plot(
    avg_csi_by_year["Year"], 
    avg_csi_by_year["Customer_Satisfaction_Index"], 
    marker="o", linestyle="-", color="teal"
)
        ax1.set_title("Average Customer Satisfaction Index by Year")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Avg Customer Satisfaction Index")
        ax1.set_xticks(avg_csi_by_year["Year"])

        for i, val in enumerate(avg_csi_by_year["Customer_Satisfaction_Index"]):
            ax1.text(avg_csi_by_year["Year"].iloc[i], val, f"{val:.2f}", 
             ha="center", va="bottom", fontsize=9)

        fig1.tight_layout()
        st.pyplot(fig1)


# -----------------------------
# 2. Avg CSI by Quarter
# -----------------------------
        avg_csi_by_qtr = (
    df_filtered.groupby("Quarter")["Customer_Satisfaction_Index"]
      .mean()
      .reset_index()
      .sort_values("Quarter")
)

        st.subheader("Average CSI by Quarter")
        fig2, ax2 = plt.subplots(figsize=(8,5))
        ax2.plot(
    avg_csi_by_qtr["Quarter"], 
    avg_csi_by_qtr["Customer_Satisfaction_Index"], 
    marker="o", linestyle="-", color="teal"
)
        ax2.set_title("Average Customer Satisfaction Index by Quarter")
        ax2.set_xlabel("Quarter")
        ax2.set_ylabel("Avg Customer Satisfaction Index")
        for i, val in enumerate(avg_csi_by_qtr["Customer_Satisfaction_Index"]):
            ax2.text(i, val, f"{val:.2f}",
                  ha="center", va="bottom", fontsize=9)

        fig2.tight_layout()
        st.pyplot(fig2)


# -----------------------------
# 3. Avg CSI by Region
# -----------------------------
        avg_csi_by_region = (
    df_filtered.groupby("Region")["Customer_Satisfaction_Index"]
      .mean()
      .reset_index()
      .sort_values("Customer_Satisfaction_Index", ascending=False)
)

        colors = plt.cm.tab20.colors
        bar_colors = [colors[i % len(colors)] 
                  for i in range(len(avg_csi_by_region))]

        st.subheader("Average CSI by Region")
        fig3, ax3 = plt.subplots(figsize=(10,6))
        bars = ax3.bar(
    avg_csi_by_region["Region"], 
    avg_csi_by_region["Customer_Satisfaction_Index"], 
    color=bar_colors
)
        ax3.set_title("Average Customer Satisfaction Index by Region")
        ax3.set_xlabel("Region")
        ax3.set_ylabel("Avg Customer Satisfaction Index")

        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2,
                  height, f"{height:.2f}", 
             ha='center', va='bottom', fontsize=9)

        fig3.tight_layout()
        st.pyplot(fig3)


# -----------------------------
# 4. Avg CSI by Segment
# -----------------------------
        avg_csi_by_segment = (
    df_filtered.groupby("Segment")["Customer_Satisfaction_Index"]
      .mean()
      .reset_index()
      .sort_values("Customer_Satisfaction_Index", ascending=False)
)

        bar_colors = [colors[i % len(colors)] 
                  for i in range(len(avg_csi_by_segment))]

        st.subheader("Average CSI by Segment")
        fig4, ax4 = plt.subplots(figsize=(10,6))
        bars = ax4.bar(
    avg_csi_by_segment["Segment"], 
    avg_csi_by_segment["Customer_Satisfaction_Index"], 
    color=bar_colors
)
        ax4.set_title("Average Customer Satisfaction Index by Segment")
        ax4.set_xlabel("Customer Segment")
        ax4.set_ylabel("Avg Customer Satisfaction Index")

        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height, f"{height:.2f}", 
             ha='center', va='bottom', fontsize=9)

        fig4.tight_layout()
        st.pyplot(fig4)

        
        st.markdown("<br><br>", unsafe_allow_html=True)

    # --------- Data Usage Analysis --------- #
    elif selected_page == "Data Usage analysis":
        st.markdown(
            """
            <h1 style='text-align: center; text-decoration: underline; font-weight: bold;font-size: 51px; margin-bottom: 30px;'>
                Data Usage Analysis
            </h1>
            """,
            unsafe_allow_html=True
        )
    
    # -----------------------------
# 1. Avg Data Usage over Quarter (by Year)
# -----------------------------
        avg_data = (
    df_filtered.groupby(["Year", "Quarter"])["Data_Usage_GB"]
      .mean()
      .reset_index()
)

# Ensure Quarter is categorical for proper ordering
        quarter_order = ["Q1", "Q2", "Q3", "Q4"]
        avg_data["Quarter"] = pd.Categorical(avg_data["Quarter"].astype(str),
                                     categories=quarter_order,
                                     ordered=True)
        avg_data = avg_data.sort_values(["Year", "Quarter"])

        st.subheader("Average Data Usage by Quarter (Year-wise)")
        fig1, ax1 = plt.subplots(figsize=(10,6))
        years = avg_data["Year"].unique()
        colors = plt.cm.tab10.colors
  
        for i, year in enumerate(years):
            year_data = avg_data[avg_data["Year"] == year]
            ax1.plot(year_data["Quarter"], year_data["Data_Usage_GB"], 
             marker="o", linestyle="-", color=colors[i % len(colors)], label=str(year))
        for x, y in zip(year_data["Quarter"], year_data["Data_Usage_GB"]):
            ax1.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=9)

        ax1.set_title("Average Data Usage")
        ax1.set_xlabel("Quarter")
        ax1.set_ylabel("Avg Data Usage (GB)")
        ax1.legend(title="Year")
        fig1.tight_layout()
        st.pyplot(fig1)


# -----------------------------
# 2. Avg Data Usage by Region
# -----------------------------
        avg_data_by_region = (
    df_filtered.groupby("Region")["Data_Usage_GB"]
      .mean()
      .reset_index()
      .sort_values("Data_Usage_GB", ascending=False)
)

        colors = plt.cm.tab20.colors
        bar_colors = [colors[i % len(colors)] for i in range(len(avg_data_by_region))]

        st.subheader("Average Data Usage by Region")
        fig2, ax2 = plt.subplots(figsize=(10,6))
        bars = ax2.bar(avg_data_by_region["Region"], avg_data_by_region["Data_Usage_GB"], color=bar_colors)
        ax2.set_title("Average Data Usage by Region")
        ax2.set_xlabel("Region")
        ax2.set_ylabel("Avg Data Usage (GB)")

        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2,
                  height, f"{height:.2f}",
             ha='center', va='bottom', fontsize=9)

        fig2.tight_layout()
        st.pyplot(fig2)


# -----------------------------
# 3. Avg Data Usage by Segment
# -----------------------------
        avg_data_by_segment = (
    df_filtered.groupby("Segment")["Data_Usage_GB"]
      .mean()
      .reset_index()
      .sort_values("Data_Usage_GB", ascending=False)
)

        bar_colors = [colors[i % len(colors)] 
                  for i in range(len(avg_data_by_segment))]

        st.subheader("Average Data Usage by Segment")
        fig3, ax3 = plt.subplots(figsize=(10,6))
        bars = ax3.bar(avg_data_by_segment["Segment"],
                    avg_data_by_segment["Data_Usage_GB"], 
                    color=bar_colors)
        ax3.set_title("Average Data Usage by Segment")
        ax3.set_xlabel("Customer Segment")
        ax3.set_ylabel("Avg Data Usage (GB)")

        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2,
              height, f"{height:.2f}",
             ha='center', va='bottom', fontsize=9)

        fig3.tight_layout()
        st.pyplot(fig3)



        st.markdown("<br><br>", unsafe_allow_html=True)

    # --------- Complaints & Service Quality --------- #
    elif selected_page == "Complaints & Service Quality":
        st.markdown(
            """
            <h1 style='text-align: center; text-decoration: underline; font-weight: bold;font-size: 51px; margin-bottom: 30px;'>
                Complaints & Service Quality
            </h1>
            """,
            unsafe_allow_html=True
        )
        
    # -----------------------------
# 1. Average Complaints Resolved % by Region
# -----------------------------
        complaints_region = (
    df_filtered.groupby('Region')['Complaints_Resolved_Pct']
      .mean()
      .reset_index()
)

        st.subheader("Average Complaints Resolved % by Region")
        fig1, ax1 = plt.subplots(figsize=(10,6))
        sns.barplot(
                data=complaints_region,
    x='Region',
    y='Complaints_Resolved_Pct',
    palette="viridis",
    ax=ax1
)
    
        for index, row in complaints_region.iterrows():
            ax1.text(
        index, row['Complaints_Resolved_Pct'],
        f"{row['Complaints_Resolved_Pct']:.1f}%",
        ha='center', va='bottom', fontsize=9
    )
        
        ax1.set_title("Average Complaints Resolved Percentage by Region")
        ax1.set_xlabel("Region")
        ax1.set_ylabel("Avg Complaints Resolved (%)")
        ax1.set_ylim(0, 100)
        fig1.tight_layout()
        st.pyplot(fig1)


# 2. Average Complaints Resolved Percentage by Segment


# Group average Complaints Resolved % 
        complaints_segment = (
    df_filtered.groupby('Segment')['Complaints_Resolved_Pct']
      .mean()
      .reset_index()
)
        st.subheader("Average Complaints Resolved Percentage by Segment")
# Create the plot
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(
    data=complaints_segment,
    x='Segment',
    y='Complaints_Resolved_Pct',
    palette="viridis",
    ax=ax
)

# Add data labels (with %)
        for index, row in complaints_segment.iterrows():
            ax.text(
        index, row['Complaints_Resolved_Pct'],
        f"{row['Complaints_Resolved_Pct']:.1f}%",  # one decimal + %
        ha='center', va='bottom', fontsize=9
    )
        ax.set_title("Average Complaints Resolved Percentage by Segment")
        ax.set_xlabel("Segment")
        ax.set_ylabel("Avg Complaints Resolved (%)")
        ax.set_ylim(0, 100)
        plt.tight_layout()
        st.pyplot(fig)



# -----------------------------
# 3. Average Complaints Resolved % over Quarters (Year-wise)
# -----------------------------
        complaints_quarter = (
    df_filtered.groupby(['Year','Quarter'])['Complaints_Resolved_Pct']
      .mean()
      .reset_index()
)
    
        years = complaints_quarter['Year'].unique()
    
        st.subheader("Average Complaints Resolved % by Quarter (Year-wise)")
        fig2, axes = plt.subplots(len(years), 1, figsize=(10, 5 * len(years)), sharex=True)

# Handle single-year case
        if len(years) == 1:
            axes = [axes]  
              
        for ax, year in zip(axes, years):
            year_df = complaints_quarter[complaints_quarter['Year'] == year]

        sns.lineplot(
        data=year_df,
        x='Quarter',
        y='Complaints_Resolved_Pct',
        marker="o",
        ax=ax
    )

        for x, y in zip(year_df['Quarter'], year_df['Complaints_Resolved_Pct']):
            ax.text(x, y, f"{y:.1f}%", ha='center', va='bottom', fontsize=9)

        ax.set_title(f"Average Complaints Resolved % - Year {year}")
        ax.set_ylabel("Resolved %")
        ax.set_ylim(0, 100)

        axes[-1].set_xlabel("Quarter")
        fig2.tight_layout()
        st.pyplot(fig2)



        st.markdown("<br><br>", unsafe_allow_html=True)

    # --------- Network Availability Analysis --------- #
    elif selected_page == "Network Availability analysis":
        st.markdown(
            """
            <h1 style='text-align: center; text-decoration: underline; font-weight: bold;font-size: 51px; margin-bottom: 30px;'>
                Network Availability Analysis
            </h1>
            """,
            unsafe_allow_html=True
        )

     # -----------------------
# 1. Network Availability by Year
# -----------------------


        net_avail_year = (
    df_filtered.groupby('Year')['Network_Availability_Pct']
      .mean()
      .reset_index()
)
        st.subheader("Average Network Availability % by Year")
        fig1, ax1 = plt.subplots(figsize=(8,6))
        sns.barplot(
    data=net_avail_year,
    x='Year',
    y='Network_Availability_Pct',
    palette="Set2",
    ax=ax1
)

# Add data labels inside bars
        for i, row in net_avail_year.iterrows():
            ax1.text(
        i, row['Network_Availability_Pct'] - 0.5,
        f"{row['Network_Availability_Pct']:.1f}%",
        ha='center', va='top', color='black', fontsize=9, weight='bold'
    )
        
        ax1.set_ylim(90, 100)
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Network Availability (%)")
        plt.tight_layout()
        st.pyplot(fig1)


# -----------------------
# 2. Network Availability by Region
# -----------------------


        net_avail_region = (
    df_filtered.groupby('Region')['Network_Availability_Pct']
      .mean()
      .reset_index()
)
        st.subheader("Average Network Availability % by Region")
        fig2, ax2 = plt.subplots(figsize=(10,6))
        sns.barplot(
    data=net_avail_region,
    x='Region',
    y='Network_Availability_Pct',
    palette="Set2",
    ax=ax2
)
    
        for i, row in net_avail_region.iterrows():
            ax2.text(
        i, row['Network_Availability_Pct'] - 0.5,
        f"{row['Network_Availability_Pct']:.1f}%",
        ha='center', va='top', color='black', fontsize=9, weight='bold'
    )
        
        ax2.set_ylim(90, 100)
        ax2.set_xlabel("Region")
        ax2.set_ylabel("Network Availability (%)")
        plt.tight_layout()
        st.pyplot(fig2)


# -----------------------
# 3. Network Availability by Segment
# -----------------------

        net_avail_segment = (
    df_filtered.groupby('Segment')['Network_Availability_Pct']
      .mean()
      .reset_index()
)
        st.subheader("Average Network Availability % by Segment")
    
        fig3, ax3 = plt.subplots(figsize=(10,6))
        sns.barplot(    data=net_avail_segment,
    x='Segment',
    y='Network_Availability_Pct',
    palette="Set2",
    ax=ax3
)
    
        for i, row in net_avail_segment.iterrows():
            ax3.text(
        i, row['Network_Availability_Pct'] - 0.5,
        f"{row['Network_Availability_Pct']:.1f}%",
        ha='center', va='top', color='black', fontsize=9, weight='bold'
    )
        
        ax3.set_ylim(90, 100)
        ax3.set_xlabel("Segment")
        ax3.set_ylabel("Network Availability (%)")
        plt.tight_layout()
        st.pyplot(fig3)   
        
        st.markdown("<br><br>", unsafe_allow_html=True)

    # --------- Retail & Recharge Analysis --------- #
    elif selected_page == "Retail & Recharge analysis":
        st.markdown(
            """
            <h1 style='text-align: center; text-decoration: underline; font-weight: bold;font-size: 51px; margin-bottom: 30px;'>
                Retail & Recharge Analysis
            </h1>
            """,
            unsafe_allow_html=True
        )
    
    
    # -----------------------
# 1. Retail Store Visits by Region
# -----------------------


        store_visits = (
    df_filtered.groupby('Region')['Retail_Store_Visits_Thousand']
      .mean()
      .reset_index()
)
        st.subheader("Retail Store Visits by Region")
        fig1, ax1 = plt.subplots(figsize=(10,6))
        sns.barplot(
    data=store_visits,
    x='Region',
    y='Retail_Store_Visits_Thousand',
    palette="Set2",
    ax=ax1
)
# Format Y axis
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}K"))

# Add labels
        for p in ax1.patches:
            val = p.get_height()
            ax1.annotate(
        f"{val:.2f}K",
        (p.get_x() + p.get_width()/2., val),
        ha='center', va='bottom',
        fontsize=9, color='black'
    )
    
        ax1.set_xlabel("Region")
        ax1.set_ylabel("Visits (in K)")
        ax1.set_title("Retail Store Visits by Region")
        plt.tight_layout()
        st.pyplot(fig1)


# -----------------------
# 2. Recharge Transactions by Region
# -----------------------

        recharge_region = (
    df_filtered.groupby('Region')['Recharge_Transactions_Thousand']
      .sum()
      .reset_index()
)
    
        recharge_region['Recharge_Transactions_Million'] = recharge_region['Recharge_Transactions_Thousand'] / 1000
        st.subheader("Recharge Transactions by Region (in Millions)")

        fig2, ax2 = plt.subplots(figsize=(10,6))
        sns.barplot(
    data=recharge_region,
    x='Region',
    y='Recharge_Transactions_Million',
    palette="magma",
    ax=ax2
)
    
        for p in ax2.patches:
            val = p.get_height()
            ax2.annotate(
        f"{val:.3f}M",
        (p.get_x() + p.get_width()/2., val/2),
        ha='center', va='center',
        fontsize=9, color='white', fontweight='bold'
    )   
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}M"))
        ax2.set_xlabel("Region")
        ax2.set_ylabel("Recharge Transactions (M)")
        ax2.set_title("Recharge Transactions by Region")
        plt.tight_layout()
        st.pyplot(fig2)


# -----------------------
# 3. Recharge Transactions by Segment
# -----------------------


        recharge_seg = (
    df_filtered.groupby('Segment')['Recharge_Transactions_Thousand']
      .sum()
      .reset_index()
)
        st.subheader("Recharge Transactions by Segment (in Millions)")
        recharge_seg['Recharge_Transactions_Million'] = recharge_seg['Recharge_Transactions_Thousand'] / 1000

        fig3, ax3 = plt.subplots(figsize=(10,6))
        sns.barplot(
    data=recharge_seg,
    x='Segment',
    y='Recharge_Transactions_Million',
    palette="magma",
    ax=ax3
)
    
        for p in ax3.patches:
            val = p.get_height() 
            ax3.annotate(
        f"{val:.3f}M",
        (p.get_x() + p.get_width()/2., val/2),
        ha='center', va='center',
        fontsize=9, color='white', fontweight='bold'
    )
    
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}M"))
        ax3.set_xlabel("Segment")
        ax3.set_ylabel("Recharge Transactions (M)")
        ax3.set_title("Recharge Transactions by Segment")
    
        plt.tight_layout()

        st.pyplot(fig3)


# -----------------------
# 4. Recharge Transactions over Quarters (by Year)
# -----------------------


        recharge_qtr = (
    df_filtered.groupby(['Year','Quarter'])['Recharge_Transactions_Thousand']
      .sum()
      .reset_index()
)

        recharge_qtr['Recharge_Transactions_Million'] = recharge_qtr['Recharge_Transactions_Thousand'] / 1000
        st.subheader("Recharge Transactions over Quarters (by Year)")
        fig4, ax4 = plt.subplots(figsize=(10,6))
        sns.lineplot(
    data=recharge_qtr,
    x='Quarter',
    y='Recharge_Transactions_Million',
    hue='Year',
    marker='o',
    linewidth=2,
    ax=ax4
)

# Add data labels
        for year in recharge_qtr['Year'].unique():
            year_df = recharge_qtr[recharge_qtr['Year'] == year]
        for x, y in zip(year_df['Quarter'], year_df['Recharge_Transactions_Million']):
            ax4.text(x, y, f"{y:.2f}M", ha='center', va='bottom', fontsize=8)

        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}M"))
        ax4.set_xlabel("Quarter")
        ax4.set_ylabel("Recharge Transactions (M)")
        ax4.set_title("Recharge Transactions over Quarters (by Year)")
        plt.tight_layout()
        st.pyplot(fig4)
     
    
    
    
        st.markdown("<br><br>", unsafe_allow_html=True)

    # --------- Voice Usage & eSIM Activations --------- #
    
    elif selected_page == "Voice Usage & eSIM Activations":
        st.markdown(
            """
            <h1 style='text-align: center; text-decoration: underline; font-weight: bold;font-size: 51px; margin-bottom: 30px;'>
                Voice Usage & eSIM Activations
            </h1>
            """,
            unsafe_allow_html=True
        )
        
     # -----------------------
# 1. Average Voice Minutes by Region
# -----------------------


        voice_region = (
    df_filtered.groupby('Region')['Voice_Minutes_Mn']
      .mean()
      .reset_index()
)
        st.subheader("Average Voice Minutes by Region")
        fig1, ax1 = plt.subplots(figsize=(10,6))
        sns.barplot(
    data=voice_region,
    x='Region',
    y='Voice_Minutes_Mn',
    palette="viridis",
    ax=ax1
)

# Add data labels
        for index, row in voice_region.iterrows():
            ax1.text(
        index, row['Voice_Minutes_Mn'],
        f"{row['Voice_Minutes_Mn']:.1f}min",
        ha='center', va='bottom', fontsize=9
    )
        
        ax1.set_title("Average Voice Minutes by Region")
        ax1.set_xlabel("Region")
        ax1.set_ylabel("Voice Minutes")
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig1)


# -----------------------
# 2. Average Voice Minutes over Quarters
# -----------------------


        voice_quarter = (
    df_filtered.groupby(['Year','Quarter'])['Voice_Minutes_Mn']
      .mean()
      .reset_index()
)
        st.subheader("Average Voice Minutes over Quarters (by Year)")
        fig2, ax2 = plt.subplots(figsize=(10,6))
        sns.lineplot(
    data=voice_quarter,
    x='Quarter',
    y='Voice_Minutes_Mn',
    hue='Year',
    marker='o',
    ax=ax2
)

# Add data labels
        for year in voice_quarter['Year'].unique():
            year_df = voice_quarter[voice_quarter['Year'] == year]
        for x, y in zip(year_df['Quarter'], year_df['Voice_Minutes_Mn']):
            ax2.text(x, y, f"{y:.1f}min", ha='center', va='bottom', fontsize=8)

        ax2.set_title("Average Voice Minutes over Quarters (by Year)")
        ax2.set_xlabel("Quarter")
        ax2.set_ylabel("Average Voice Minutes")
        plt.tight_layout()
        st.pyplot(fig2)   


    # ===================== 3. eSIM_Activations by Regions =====================


# Group by Year and Region and sum eSIM activations
        avg_esim_by_region = (
    df_filtered.groupby(["Year", "Region"])["eSIM_Activations"]
      .sum()
      .reset_index()
      .sort_values(["Region", "Year"])
)
        st.subheader("📶 eSIM Activations by Region Over Years")
# Plot using Matplotlib
        fig, ax = plt.subplots(figsize=(12, 6))
        regions = avg_esim_by_region["Region"].unique()
        colors = plt.cm.tab20.colors

        for i, region in enumerate(regions):
            region_data = avg_esim_by_region[avg_esim_by_region["Region"] == region]
    # Convert activations to thousands for plotting
        ax.plot(region_data["Year"],
            region_data["eSIM_Activations"] / 1000,
            marker="o",
            linestyle="-",
            color=colors[i % len(colors)],
            label=region)
    # Add data labels
        for x, y in zip(region_data["Year"], region_data["eSIM_Activations"] / 1000):
            ax.text(x, y, f"{y:.2f}K", ha="center", va="bottom", fontsize=9)

        ax.set_title("eSIM Activations by Region Over Years")
        ax.set_xlabel("Year")
        ax.set_ylabel("Total eSIM Activations (in Thousands)")
        ax.set_xticks(sorted(df["Year"].unique()))
        ax.legend(title="Region")
        fig.tight_layout()
    
        st.pyplot(fig)
        
# -----------------------
# 4. New Activations vs Deactivations over Years
# -----------------------


        act_deact = (
    df_filtered.groupby('Year')[['New_Activations','Deactivations']]
      .sum()
      .reset_index()
)
        st.subheader("New Activations vs Deactivations over Years")
        fig1, ax1 = plt.subplots(figsize=(10,6))
        act_deact.plot(
    x='Year',
    kind='bar',
    stacked=False,
    color=['#1f77b4','#ff7f0e'],
    ax=ax1
)

# Format Y-axis in millions (M)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1_000_000:.1f}M"))

# Add data labels
        for p in ax1.patches:
            if p.get_height() > 0:
                value_m = p.get_height() / 1_000_000
                ax1.annotate(
                                f"{value_m:.2f}M",
            (p.get_x() + p.get_width()/2., p.get_height()),
            ha='center', va='bottom',
            fontsize=9, color='black'
        )
        
        ax1.set_title("New Activations vs Deactivations Over Years")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Count (in M)")
        ax1.set_xticklabels(act_deact['Year'], rotation=0)
        ax1.legend(title="Metric")
        plt.tight_layout()
        st.pyplot(fig1)


# -----------------------
# 5. New Activations vs Deactivations by Regions
# -----------------------


        act_deact_region = (
    df_filtered.groupby('Region')[['New_Activations','Deactivations']]
      .sum()
      .reset_index()
)
        st.subheader("New Activations vs Deactivations by Regions")
        fig2, ax2 = plt.subplots(figsize=(10,6))
        act_deact_region.plot(
                x='Region',
                        kind='bar',
                                stacked=False,
                                        color=['#1f77b4','#ff7f0e'],
            ax=ax2
)

# Format Y-axis in millions (M)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1_000_000:.1f}M"))

# Add data labels
        for p in ax2.patches:
            if p.get_height() > 0:
                value_m = p.get_height() / 1_000_000
                ax2.annotate(
            f"{value_m:.2f}M",
            (p.get_x() + p.get_width()/2., p.get_height()),
            ha='center', va='bottom',
            fontsize=9, color='black'
        )
        
        ax2.set_title("New Activations vs Deactivations by Regions")
        ax2.set_xlabel("Region")
        ax2.set_ylabel("Count (in M)")
        ax2.set_xticklabels(act_deact_region['Region'], rotation=45)
        ax2.legend(["Activations SIM", "Deactivations SIM"], title="Metric", loc="best")
        plt.tight_layout()
        st.pyplot(fig2)


        st.markdown("<br><br>", unsafe_allow_html=True)
# --------------------------
# MAIN APP
# --------------------------
def main():
        if "logged_in" not in st.session_state:
             st.session_state.logged_in = False
        if "page" not in st.session_state:
             st.session_state.page = "login"

        if not st.session_state.logged_in:
             login_page()
        elif st.session_state.page == "business":
             business_context()
        elif st.session_state.page == "dashboard":
             dashboard()
        else:
             project_intro()

if __name__ == "__main__":
    main()

