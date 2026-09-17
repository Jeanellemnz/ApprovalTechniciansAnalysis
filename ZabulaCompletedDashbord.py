import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import glob
import os

# --- 1. PAGE CONFIG & STYLING ---
st.set_page_config(
    page_title="Technician Approval Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for styling metrics and headers
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        h1 { color: #2c3e50; }
        h2, h3 { color: #34495e; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Mini Case: Technician Approval Anomaly Dashboard")
st.markdown("Approval Analysis.")

# --- 2. DATA LOADING & CACHING ---
@st.cache_data
def load_data():
    folder_path = "./data"  
    files = os.path.join(folder_path, "*.xlsx")
    filelist = glob.glob(files)

    if not filelist:
        st.error(f"No Excel files found in `{folder_path}`. Please ensure your 'data' folder and files are uploaded to GitHub.")
        st.stop()

    dataframe1 = []
    for file in filelist:
        TechCases = pd.read_excel(file, engine="openpyxl")
        dataframe1.append(TechCases)
        TechCases.columns = TechCases.columns.str.strip().str.upper().str.replace(" ", "_")

    CompleteFile = pd.concat(dataframe1, ignore_index=True)
    
    # Pre-processing
    CompleteFile["APPROVAL_DATE"] = pd.to_datetime(CompleteFile["APPROVAL_DATE"])
    CompleteFile = CompleteFile.sort_values(by=["PROVIDER_APPROVING_NAME", "APPROVAL_DATE"]).reset_index(drop=True)
    CompleteFile["ApprovalTime"] = CompleteFile.groupby("PROVIDER_APPROVING_NAME")["APPROVAL_DATE"].diff()
    CompleteFile["ApprovalDuration"] = CompleteFile["ApprovalTime"].dt.total_seconds()
    CompleteFile = CompleteFile.drop(columns=["ApprovalTime"])
    
    return CompleteFile

CompleteFile = load_data()

# --- 3. SIDEBAR FILTERS ---
st.sidebar.header("Filter Dashboard")
technicians = CompleteFile["PROVIDER_APPROVING_NAME"].dropna().unique().tolist()
selected_techs = st.sidebar.multiselect("Select Technician(s)", technicians, default=technicians)

# Filter dataset based on sidebar
FilteredFile = CompleteFile[CompleteFile["PROVIDER_APPROVING_NAME"].isin(selected_techs)]

# --- 4. DATA TRANSFORMATIONS ---
DurationBlocks = (FilteredFile["ApprovalDuration"] > 600) | (FilteredFile["ApprovalDuration"].isna())
FilteredFile["BlockID"] = DurationBlocks.cumsum()

WorkingBlocksData = FilteredFile[FilteredFile["ApprovalDuration"] <= 600].copy()
WorkingBlocksData["IsZeroSec"] = (WorkingBlocksData["ApprovalDuration"] == 0).astype(int)
WorkingBlocksData["IsUnder2"] = (WorkingBlocksData["ApprovalDuration"] <= 2).astype(int)
WorkingBlocksData["IsUnder5"] = (WorkingBlocksData["ApprovalDuration"] <= 5).astype(int)

# Block Analysis Aggregation
BlockAnalysis = WorkingBlocksData.groupby(["PROVIDER_APPROVING_NAME", "BlockID"]).agg(
    TotalCases=("ApprovalDuration", "count"), 
    AverageDuration=("ApprovalDuration", "mean"), 
    ShiftStart=("APPROVAL_DATE", "min"), 
    ShiftEnd=("APPROVAL_DATE", "max"),
    MedianDuration=("ApprovalDuration", "median"), 
    PaceFluctuation=("ApprovalDuration", "std"), 
    InstantApprovals=("IsZeroSec", "sum"), 
    ReflexApproval=("IsUnder2", "sum"),
    HighRiskApproval=("IsUnder5", "sum")
).reset_index()

ShiftDurationSeconds = (BlockAnalysis["ShiftEnd"] - BlockAnalysis["ShiftStart"]).dt.total_seconds()
ShiftDurationMinutes = ShiftDurationSeconds / 60

BlockAnalysis["HighRiskRatio"] = ((BlockAnalysis["HighRiskApproval"] / BlockAnalysis["TotalCases"]) * 100).round(2)
BlockAnalysis["BlockDensityCPM"] = (BlockAnalysis["TotalCases"] / ShiftDurationMinutes.clip(lower=1/60)).round(2)
BlockAnalysis["AverageDistortion"] = (BlockAnalysis["AverageDuration"] - BlockAnalysis["MedianDuration"]).round(2)
BlockAnalysis["LateNightApprovals"] = BlockAnalysis["ShiftStart"].dt.hour.between(0, 4)

BlockAnalysisGaps = FilteredFile.groupby("BlockID")["ApprovalDuration"].first().reset_index()
BlockAnalysisGaps = BlockAnalysisGaps.rename(columns={"ApprovalDuration": "ClaimedPrepTime"})
BlockAnalysis = BlockAnalysis.merge(BlockAnalysisGaps, on="BlockID", how="left")
BlockAnalysis["ClaimedApprovalAverage"] = (BlockAnalysis["ClaimedPrepTime"] / BlockAnalysis["TotalCases"]).round(2)
BlockAnalysis["FatigueIndex"] = (ShiftDurationSeconds / BlockAnalysis["ClaimedPrepTime"]).round(2)
BlockAnalysis["PaceFluctuation"] = BlockAnalysis["PaceFluctuation"].fillna(0)

BlockAnalysis["PayRate"] = np.where(BlockAnalysis["ShiftStart"] < "2020-06-01", 50, 17)
BlockAnalysis["BlockEarnings"] = BlockAnalysis["TotalCases"] * BlockAnalysis["PayRate"]

# Separate individual technicians for specific visualizations
Matt = WorkingBlocksData[WorkingBlocksData["PROVIDER_APPROVING_NAME"] == "Matt Shawn"]
Juan = WorkingBlocksData[WorkingBlocksData["PROVIDER_APPROVING_NAME"] == "Juan Mendez"]
Gary = WorkingBlocksData[WorkingBlocksData["PROVIDER_APPROVING_NAME"] == "Gary arnold"]

# --- 5. TOP-LEVEL METRICS ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Filtered Records", len(FilteredFile))
col2.metric("Total Analysis Blocks", len(BlockAnalysis))
col3.metric("Selected Technicians", len(selected_techs))

st.markdown("---")

# --- 6. INTERACTIVE TABS FOR VISUALIZATIONS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview & Distributions", 
    "📦 Technician Boxplots", 
    "📉 Histograms & Skewness", 
    "⏱️ Timelines & Pay Cuts", 
    "🧠 Fatigue & Correlations"
])

with tab1:
    st.subheader("Approval Duration Distribution (Breaks <= 10 minutes)")
    fig, ax = plt.subplots(figsize=(10, 4))
    FilteredFile[FilteredFile["ApprovalDuration"] <= 600]["ApprovalDuration"].plot(
        kind="hist", bins=75, edgecolor="black", color="#FF1493", ax=ax
    )
    ax.set_title("Overall Approval Duration Distribution")
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("Distribution of Average Duration (Block Level)")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(BlockAnalysis["AverageDuration"], bins=100, color="#980EDD", kde=True, ax=ax)
    ax.set_xlim(0, 100)
    ax.set_title("Distribution of Average Duration")
    st.pyplot(fig)

with tab2:
    st.subheader("Technician Boxplots (Durations <= 600s)")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Matt
    Matt["ApprovalDuration"].plot(kind="box", ax=axes[0], patch_artist=True, boxprops=dict(facecolor="blue", color="black"), showfliers=False)
    axes[0].set_title("Matt Shawn")
    axes[0].set_ylabel("Duration (seconds)")
    
    # Juan
    Juan["ApprovalDuration"].plot(kind="box", ax=axes[1], patch_artist=True, boxprops=dict(facecolor="green", color="black"), showfliers=False)
    axes[1].set_title("Juan Mendez")
    
    # Gary
    Gary["ApprovalDuration"].plot(kind="box", ax=axes[2], patch_artist=True, boxprops=dict(facecolor="hotpink", color="black"), showfliers=False)
    axes[2].set_title("Gary Arnold")
    
    plt.suptitle("Approval Duration Distribution by Technician", fontsize=14, weight="bold")
    st.pyplot(fig)

with tab3:
    st.subheader("Technician Histograms (Durations <= 100s)")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    Matt["ApprovalDuration"].plot(kind="hist", bins=50, range=(0, 100), edgecolor="black", color="blue", ax=axes[0])
    axes[0].set_title("Matt Shawn")
    axes[0].set_xlabel("Duration (Seconds)")
    
    Juan["ApprovalDuration"].plot(kind="hist", bins=50, range=(0, 100), edgecolor="black", color="green", ax=axes[1])
    axes[1].set_title("Juan Mendez")
    axes[1].set_xlabel("Duration (Seconds)")
    
    Gary["ApprovalDuration"].plot(kind="hist", bins=50, range=(0, 100), edgecolor="black", color="hotpink", ax=axes[2])
    axes[2].set_title("Gary Arnold")
    axes[2].set_xlabel("Duration (Seconds)")
    
    plt.suptitle("Skewness Visualization (0 to 100 seconds)", fontsize=14, weight="bold")
    st.pyplot(fig)

with tab4:
    st.subheader("Chronological Timelines (Before/After June 2020 Pay Cut)")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    
    Matt.plot.scatter(x="APPROVAL_DATE", y="ApprovalDuration", color="blue", alpha=0.3, s=10, ax=ax1)
    ax1.axvline(pd.to_datetime("2020-06-01"), color="black", linestyle="--", linewidth=2)
    ax1.set_title("Matt Shawn Timeline")
    
    Juan.plot.scatter(x="APPROVAL_DATE", y="ApprovalDuration", color="green", alpha=0.3, s=10, ax=ax2)
    ax2.axvline(pd.to_datetime("2020-06-01"), color="black", linestyle="--", linewidth=2)
    ax2.set_title("Juan Mendez Timeline")
    
    Gary.plot.scatter(x="APPROVAL_DATE", y="ApprovalDuration", color="hotpink", alpha=0.3, s=10, ax=ax3)
    ax3.axvline(pd.to_datetime("2020-06-01"), color="black", linestyle="--", linewidth=2)
    ax3.set_title("Gary Arnold Timeline")
    
    plt.suptitle("Chronological Approval Duration", fontsize=14, weight="bold")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Gary Arnold Daily Earnings Analysis")
    DailyEarnings = BlockAnalysis[BlockAnalysis["PROVIDER_APPROVING_NAME"] == "Gary arnold"].copy()
    if not DailyEarnings.empty:
        DailyEarnings["Date"] = DailyEarnings["ShiftStart"].dt.date
        DailySummary = DailyEarnings.groupby("Date")["BlockEarnings"].sum().reset_index()
        DailySummary["Date"] = pd.to_datetime(DailySummary["Date"])

        fig_earn, ax_earn = plt.subplots(figsize=(12, 4))
        sns.scatterplot(data=DailySummary, x="Date", y="BlockEarnings", color="#FF1493", alpha=0.7, s=40, ax=ax_earn)
        ax_earn.axvline(pd.to_datetime("2020-06-01"), color="black", linestyle="--", linewidth=2, label="June 2020 Pay Cut")
        ax_earn.set_title("Gary Arnold Daily Earnings Before and After Pay Cut")
        ax_earn.legend()
        st.pyplot(fig_earn)

with tab5:
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Fatigue Index vs. High-Risk Ratio")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=BlockAnalysis, x="FatigueIndex", y="HighRiskRatio", color="#980EDD", alpha=0.6, ax=ax)
        ax.set_xlim(0, 10)
        ax.set_title("Fatigue Index vs. High-Risk Approvals")
        st.pyplot(fig)
    
    with col_r:
        st.subheader("Density Heatmap")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=BlockAnalysis, x="FatigueIndex", y="HighRiskRatio", bins=30, cbar=True, cmap="viridis", ax=ax)
        ax.set_title("Fatigue vs. High-Risk Density")
        st.pyplot(fig)

    st.subheader("Correlation Matrix")
    CaseCorr = BlockAnalysis[["FatigueIndex", "HighRiskRatio", "AverageDuration"]].dropna()
    if not CaseCorr.empty:
        CorrMatrix = CaseCorr.corr()
        st.dataframe(CorrMatrix.style.background_gradient(cmap='coolwarm', axis=None).format("{:.3f}"))

# --- 7. STATISTICAL BENCHMARKS & TABLES ---
st.markdown("---")
st.header("Statistical Benchmarks & Summary Data")

def TestBenchmarks(Speed):
    Speed = Speed.dropna()
    if len(Speed) < 2: return pd.Series(dtype=float)
    TrueMean = Speed.mean()
    
    t10, p10 = stats.ttest_1samp(Speed, popmean=600)
    stand10 = "Significantly Faster" if (TrueMean < 600 and p10 < 0.05) else "No Difference"
    t5, p5 = stats.ttest_1samp(Speed, popmean=300)
    stand5 = "Significantly Faster" if (TrueMean < 300 and p5 < 0.05) else "No Difference"
    t2, p2 = stats.ttest_1samp(Speed, popmean=120)
    stand2 = "Significantly Faster" if (TrueMean < 120 and p2 < 0.05) else "No Difference"
    t1, p1 = stats.ttest_1samp(Speed, popmean=60)
    stand1 = "Significantly Faster" if (TrueMean < 60 and p1 < 0.05) else "No Difference"

    return pd.Series({
        "Sample Size": len(Speed),
        "Actual Mean (s)": round(TrueMean, 2),
        "10 Min Exp": stand10,
        "5 Min Exp": stand5,
        "2 Min Exp": stand2,
        "1 Min Exp": stand1
    })

if not BlockAnalysis.empty:
    BenchmarkSummary = BlockAnalysis.groupby("PROVIDER_APPROVING_NAME")["AverageDuration"].apply(TestBenchmarks).unstack()
    st.dataframe(BenchmarkSummary)

st.subheader("Full Block Analysis Data Table")
st.dataframe(BlockAnalysis, hide_index=True)

# --- 8. DOWNLOAD CSV ---
csv = BlockAnalysis.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Cleaned Block Analysis CSV",
    data=csv,
    file_name='ClearCheckApprovalAnomaly.csv',
    mime='text/csv',
)