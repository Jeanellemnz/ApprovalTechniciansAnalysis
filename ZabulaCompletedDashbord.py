import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import glob
import os

# --- 1. PAGE CONFIG & HEADER ---
st.set_page_config(page_title="Technician Approval Dashboard", layout="wide")
st.title("Dental Mini Case: Technician Approval Analysis")

# --- 2. DATA LOADING & CACHING ---
# @st.cache_data prevents Streamlit from reloading/re-combining Excel files on every user interaction
@st.cache_data
def load_data():

    folder_path = "./data"  
    files = os.path.join(folder_path, "*.xlsx")
    filelist = glob.glob(files)

    if not filelist:
        st.error(f"No Excel files found in {folder_path}. Please add your data to the repository.")
        st.stop()

    dataframe1 = []
    for file in filelist:
        TechCases = pd.read_excel(file, engine="openpyxl")
        dataframe1.append(TechCases)
        # Clean column names
        TechCases.columns = TechCases.columns.str.strip().str.upper().str.replace(" ", "_")

    CompleteFile = pd.concat(dataframe1, ignore_index=True)
    
    # Pre-processing
    CompleteFile["APPROVAL_DATE"] = pd.to_datetime(CompleteFile["APPROVAL_DATE"])
    CompleteFile = CompleteFile.sort_values(by=["PROVIDER_APPROVING_NAME", "APPROVAL_DATE"]).reset_index(drop=True)
    CompleteFile["ApprovalTime"] = CompleteFile.groupby("PROVIDER_APPROVING_NAME")["APPROVAL_DATE"].diff()
    CompleteFile["ApprovalDuration"] = CompleteFile["ApprovalTime"].dt.total_seconds()
    CompleteFile = CompleteFile.drop(columns=["ApprovalTime"])
    
    return CompleteFile

# Load the data
CompleteFile = load_data()

# --- 3. DASHBOARD METRICS ---
st.header("Overview")
col1, col2 = st.columns(2)
col1.metric("Total Records", len(CompleteFile))
with col2:
    st.write("**Records per Technician:**")
    st.dataframe(CompleteFile["PROVIDER_APPROVING_NAME"].value_counts())

# --- 4. DATA TRANSFORMATIONS ---
# Creating blocks under the condition that if duration > 600s or NaN, create new block
DurationBlocks = (CompleteFile["ApprovalDuration"] > 600) | (CompleteFile["ApprovalDuration"].isna())
CompleteFile["BlockID"] = DurationBlocks.cumsum()

WorkingBlocksData = CompleteFile[CompleteFile["ApprovalDuration"] <= 600].copy()
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

# KPI Calculations
BlockAnalysis["HighRiskRatio"] = ((BlockAnalysis["HighRiskApproval"] / BlockAnalysis["TotalCases"]) * 100).round(2)
BlockAnalysis["BlockDensityCPM"] = (BlockAnalysis["TotalCases"] / ShiftDurationMinutes.clip(lower=1/60)).round(2)
BlockAnalysis["AverageDistortion"] = (BlockAnalysis["AverageDuration"] - BlockAnalysis["MedianDuration"]).round(2)
BlockAnalysis["LateNightApprovals"] = BlockAnalysis["ShiftStart"].dt.hour.between(0, 4)

BlockAnalysisGaps = CompleteFile.groupby("BlockID")["ApprovalDuration"].first().reset_index()
BlockAnalysisGaps = BlockAnalysisGaps.rename(columns={"ApprovalDuration": "ClaimedPrepTime"})
BlockAnalysis = BlockAnalysis.merge(BlockAnalysisGaps, on="BlockID", how="left")
BlockAnalysis["ClaimedApprovalAverage"] = (BlockAnalysis["ClaimedPrepTime"] / BlockAnalysis["TotalCases"]).round(2)
BlockAnalysis["FatigueIndex"] = (ShiftDurationSeconds / BlockAnalysis["ClaimedPrepTime"]).round(2)
BlockAnalysis["PaceFluctuation"] = BlockAnalysis["PaceFluctuation"].fillna(0)

# Pay Rate logic
BlockAnalysis["PayRate"] = np.where(BlockAnalysis["ShiftStart"] < "2020-06-01", 50, 17)
BlockAnalysis["BlockEarnings"] = BlockAnalysis["TotalCases"] * BlockAnalysis["PayRate"]

st.header("Block Analysis Data")
st.dataframe(BlockAnalysis, hide_index=True)


# --- 5. VISUALIZATIONS ---
st.header("Visualizations")

tab1, tab2, tab3, tab4 = st.tabs(["Duration Distributions", "Timeline & Pay Cut", "Fatigue Analysis", "Correlation"])

with tab1:
    st.subheader("Approval Duration Distribution (Breaks <= 10 mins)")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    CompleteFile[CompleteFile["ApprovalDuration"] <= 600]["ApprovalDuration"].plot(
        kind="hist", bins=75, edgecolor="black", color="#FF1493", ax=ax1
    )
    ax1.set_title("Approval Duration Distribution")
    ax1.set_xlabel("Duration (seconds)")
    st.pyplot(fig1)

    st.subheader("Distribution of Average Duration")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.histplot(BlockAnalysis["AverageDuration"], bins=100, color="#980EDD", kde=True, ax=ax2)
    ax2.set_xlim(0, 100)
    st.pyplot(fig2)

with tab2:
    st.subheader("Gary Arnold Daily Earnings Before/After Pay Cut")
    DailyEarnings = BlockAnalysis[BlockAnalysis["PROVIDER_APPROVING_NAME"] == "Gary arnold"].copy()
    DailyEarnings["Date"] = DailyEarnings["ShiftStart"].dt.date
    DailySummary = DailyEarnings.groupby("Date")["BlockEarnings"].sum().reset_index()
    DailySummary["Date"] = pd.to_datetime(DailySummary["Date"])

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    sns.scatterplot(data=DailySummary, x="Date", y="BlockEarnings", color="#FF1493", alpha=0.7, s=40, ax=ax3)
    ax3.axvline(pd.to_datetime("2020-06-01"), color="black", linestyle="--", linewidth=2, label="June 2020 Pay Cut")
    ax3.legend()
    st.pyplot(fig3)

    st.subheader("Gary Arnold Approval Duration Shift Over Time")
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    sns.scatterplot(data=DailyEarnings, x="ShiftStart", y="AverageDuration", hue="PayRate", palette="Set1", alpha=0.7, ax=ax4)
    ax4.axvline(pd.to_datetime("2020-06-01"), color="black", linestyle="--", linewidth=2, label="June 2020 Pay Cut")
    st.pyplot(fig4)

with tab3:
    st.subheader("Fatigue vs. High-Risk Approvals")
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.histplot(data=BlockAnalysis, x="FatigueIndex", y="HighRiskRatio", bins=30, cbar=True, cmap="viridis", ax=ax5)
    st.pyplot(fig5)

with tab4:
    st.subheader("Correlation Matrix")
    CaseCorr = BlockAnalysis[["FatigueIndex", "HighRiskRatio", "AverageDuration"]].dropna()
    CorrMatrix = CaseCorr.corr()
    st.dataframe(CorrMatrix.style.background_gradient(cmap='coolwarm', axis=None).format("{:.3f}"))

# --- 6. STATISTICAL TESTS ---
st.header("Statistical Benchmarks")

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

BenchmarkSummary = BlockAnalysis.groupby("PROVIDER_APPROVING_NAME")["AverageDuration"].apply(TestBenchmarks).unstack()
st.dataframe(BenchmarkSummary)

st.subheader("Welch's T-Test: Pre vs Post Pay Cut")
BeforeCut = BlockAnalysis[BlockAnalysis["PayRate"] == 50]["AverageDuration"].dropna()
AfterCut = BlockAnalysis[BlockAnalysis["PayRate"] == 17]["AverageDuration"].dropna()

TestResults = stats.ttest_ind(BeforeCut, AfterCut, equal_var=False)

col_a, col_b = st.columns(2)
col_a.metric("Avg Speed BEFORE Cut ($50)", f"{BeforeCut.mean():.2f} sec")
col_b.metric("Avg Speed AFTER Cut ($17)", f"{AfterCut.mean():.2f} sec")

if TestResults.pvalue < 0.05 and AfterCut.mean() < BeforeCut.mean():
    st.success("Result: Statistically significant change in average approval time after the pay cut.")
else:
    st.info("Result: No statistically significant change.")

# --- 7. EXPORT DATA ---
st.header("Export Cleaned Data")
csv = BlockAnalysis.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Cleaned Block Analysis CSV",
    data=csv,
    file_name='ClearCheckApprovalAnomaly.csv',
    mime='text/csv',
)