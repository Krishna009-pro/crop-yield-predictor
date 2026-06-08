# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GOOGLE COLAB — CROP YIELD PREDICTION USING MACHINE LEARNING           ║
# ║  Dataset : Kaggle — "Agriculture Crops Production in India"             ║
# ║  Link    : https://www.kaggle.com/datasets/srinivas1/                  ║
# ║             agricuture-crops-production-in-india                        ║
# ║  Author  : [Your Name]   |   Date: June 2026                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
# HOW TO USE THIS FILE IN GOOGLE COLAB:
#   Option A (Recommended):
#     1. Open https://colab.research.google.com
#     2. File → New Notebook
#     3. Copy each CELL block below into separate Colab cells (split at # ── CELL)
#
#   Option B:
#     1. Upload this .py file to Colab
#     2. Run: !python crop_yield_colab.py
#
# DATASET SETUP (2 options):
#   Option 1 — Kaggle API (automatic, Cell 1 handles this)
#   Option 2 — Manual: download crop_production.csv from Kaggle and upload to Colab
# =============================================================================


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 1: Install Libraries & Mount Drive (Run First)
# ══════════════════════════════════════════════════════════════════════════════

# Install any missing packages (most are pre-installed in Colab)
import subprocess
subprocess.run(["pip", "install", "xgboost", "seaborn", "--quiet"], check=False)

import os, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection  import cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.linear_model     import LinearRegression, Ridge, Lasso
from sklearn.tree             import DecisionTreeRegressor
from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics          import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection       import permutation_importance
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")
np.random.seed(42)

print("✅ All libraries imported successfully!")
print(f"   NumPy     : {np.__version__}")
print(f"   Pandas    : {pd.__version__}")
print(f"   XGBoost   : {xgb.__version__}")
print(f"   Seaborn   : {sns.__version__}")

# Global colour palettes
PAL_GREEN = ["#1B5E20","#2E7D32","#388E3C","#43A047","#66BB6A","#A5D6A7","#E8F5E9"]
PAL_MULTI = ["#2E7D32","#1565C0","#E65100","#6A1B9A","#AD1457","#00695C","#F57F17"]
BG = "#FAFAFA"

plt.rcParams.update({
    "figure.facecolor" : BG,
    "axes.facecolor"   : BG,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.titlesize"   : 14,
    "axes.labelsize"   : 12,
    "xtick.labelsize"  : 10,
    "ytick.labelsize"  : 10,
    "legend.fontsize"  : 10,
})
print("\n✅ Global settings configured.")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 2: Load Dataset (Kaggle API or Upload Manually)
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────
#  METHOD A: Kaggle API (paste your kaggle.json credentials below)
# ─────────────────────────────────────────────────────────────────
# Uncomment and fill in your Kaggle username & key, then run:
#
# import json
# kaggle_creds = {"username": "YOUR_KAGGLE_USERNAME", "key": "YOUR_KAGGLE_API_KEY"}
# os.makedirs("/root/.kaggle", exist_ok=True)
# with open("/root/.kaggle/kaggle.json", "w") as f:
#     json.dump(kaggle_creds, f)
# os.chmod("/root/.kaggle/kaggle.json", 0o600)
# !kaggle datasets download -d srinivas1/agricuture-crops-production-in-india --unzip
# df_raw = pd.read_csv("crop_production.csv")

# ─────────────────────────────────────────────────────────────────
#  METHOD B: Manual Upload
#  Run this cell → a file picker will appear → upload crop_production.csv
# ─────────────────────────────────────────────────────────────────
# from google.colab import files
# uploaded = files.upload()
# df_raw = pd.read_csv(list(uploaded.keys())[0])

# ─────────────────────────────────────────────────────────────────
#  METHOD C: Auto-generate synthetic dataset (no Kaggle needed)
#  This is the default — just run the cell!
# ─────────────────────────────────────────────────────────────────

def generate_dataset():
    states = {
        "Maharashtra"    : ["Latur","Osmanabad","Aurangabad","Nashik","Pune","Nagpur","Amravati"],
        "Punjab"         : ["Ludhiana","Amritsar","Patiala","Bathinda","Jalandhar"],
        "Uttar Pradesh"  : ["Varanasi","Lucknow","Agra","Kanpur","Allahabad"],
        "Rajasthan"      : ["Jaipur","Jodhpur","Udaipur","Kota","Ajmer"],
        "Andhra Pradesh" : ["Guntur","Krishna","East Godavari","Nellore","Kurnool"],
        "Karnataka"      : ["Belagavi","Dharwad","Bidar","Ballari","Hassan"],
        "Madhya Pradesh" : ["Indore","Bhopal","Gwalior","Vidisha","Sagar"],
        "Bihar"          : ["Patna","Muzaffarpur","Gaya","Bhagalpur","Darbhanga"],
        "Gujarat"        : ["Surat","Rajkot","Vadodara","Ahmedabad","Anand"],
        "West Bengal"    : ["Howrah","Bardhaman","Murshidabad","Nadia","Jalpaiguri"],
    }
    crops_info = {
        # crop: (base_yield_kg_ha, std_dev, [valid_seasons])
        "Rice"      : (2200, 600,   ["Kharif","Whole Year"]),
        "Wheat"     : (3100, 700,   ["Rabi"]),
        "Maize"     : (2400, 700,   ["Kharif","Rabi"]),
        "Soyabean"  : (900,  250,   ["Kharif"]),
        "Cotton"    : (450,  130,   ["Kharif"]),
        "Sugarcane" : (65000,12000, ["Whole Year","Kharif"]),
        "Tur"       : (650,  180,   ["Kharif"]),
        "Jowar"     : (900,  200,   ["Kharif","Rabi"]),
        "Bajra"     : (800,  220,   ["Kharif"]),
        "Groundnut" : (1300, 300,   ["Kharif","Rabi"]),
        "Mustard"   : (1050, 230,   ["Rabi"]),
        "Potato"    : (20000,4000,  ["Rabi"]),
        "Onion"     : (15000,3000,  ["Rabi","Kharif"]),
        "Tomato"    : (22000,5000,  ["Kharif","Rabi"]),
        "Gram"      : (850,  200,   ["Rabi"]),
    }
    rows = []
    for state, districts in states.items():
        for district in districts:
            for year in range(2001, 2023):
                for crop, (base_y, std_y, seasons) in crops_info.items():
                    season  = np.random.choice(seasons)
                    trend   = 1 + 0.008 * (year - 2001)       # ~0.8% annual yield gain
                    rf_mm   = max(100, np.random.normal(750, 300))
                    rf_eff  = np.clip(rf_mm / 750, 0.5, 1.5)
                    area    = max(100, np.random.lognormal(10, 1.5))
                    yield_  = max(50, np.random.normal(base_y * trend * rf_eff, std_y))
                    prod    = yield_ * area / 1000             # metric tonnes
                    rows.append({
                        "State_Name"     : state,
                        "District_Name"  : district,
                        "Crop_Year"      : year,
                        "Season"         : season,
                        "Crop"           : crop,
                        "Area"           : round(area, 2),
                        "Production"     : round(prod, 2),
                        "Annual_Rainfall": round(rf_mm, 1),
                        "Fertiliser"     : round(area * np.random.uniform(0.05, 0.2), 2),
                        "Pesticide"      : round(area * np.random.uniform(0.001, 0.01), 3),
                    })
    return pd.DataFrame(rows)

print("⏳ Generating synthetic dataset (mirrors Kaggle structure)...")
df_raw = generate_dataset()
print(f"✅ Dataset ready! Shape: {df_raw.shape}")
print(f"\nColumn names:\n{list(df_raw.columns)}")
print(f"\nFirst 5 rows:")
df_raw.head()


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 3: Dataset Understanding — Shape, Types, Missing Values
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("   DATASET UNDERSTANDING")
print("=" * 60)

print(f"\n📦 Shape     : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
print(f"📅 Year Range: {df_raw['Crop_Year'].min()} – {df_raw['Crop_Year'].max()}")
print(f"🌾 Unique Crops  : {df_raw['Crop'].nunique()}")
print(f"🗺️  Unique States : {df_raw['State_Name'].nunique()}")
print(f"📍 Districts     : {df_raw['District_Name'].nunique()}")

print(f"\n── Data Types ──────────────────────────────────────────")
print(df_raw.dtypes.to_string())

print(f"\n── Missing Values ──────────────────────────────────────")
miss = df_raw.isnull().sum()
miss_pct = (miss / len(df_raw) * 100).round(2)
miss_df = pd.DataFrame({"Missing Count": miss, "Missing %": miss_pct})
print(miss_df.to_string())

print(f"\n── Statistical Summary ─────────────────────────────────")
df_raw.describe().round(2)


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 4: Data Cleaning & Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

df = df_raw.copy()

# Step 1: Remove invalid rows
before = len(df)
df = df[df["Area"] > 0]
df = df[df["Production"] >= 0]
print(f"Removed {before - len(df):,} rows with Area≤0 or Production<0")

# Step 2: Derive target variable
df["Yield_kg_ha"] = (df["Production"] * 1000) / df["Area"]
df = df[df["Yield_kg_ha"] > 0]

# Step 3: Remove per-crop outliers (>99.5th percentile)
before = len(df)
df = df[df["Yield_kg_ha"] < df.groupby("Crop")["Yield_kg_ha"].transform(lambda x: x.quantile(0.995))]
print(f"Removed {before - len(df):,} outlier rows (>99.5th percentile per crop)")

# Step 4: Log-transform target
df["log_Yield"] = np.log10(df["Yield_kg_ha"])

# Step 5: Encode categoricals
cat_cols = ["State_Name", "District_Name", "Crop", "Season"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Step 6: Build feature matrix
feature_cols = [f"{c}_enc" for c in cat_cols] + \
               [c for c in ["Crop_Year","Area","Annual_Rainfall","Fertiliser","Pesticide"] if c in df.columns]

X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
y = df["log_Yield"]

print(f"\n✅ Cleaned dataset: {df.shape[0]:,} rows")
print(f"✅ Feature matrix : {X.shape}")
print(f"✅ Target (log₁₀ Yield) range: [{y.min():.3f}, {y.max():.3f}]")
print(f"\nFeatures used: {feature_cols}")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 5: EDA — Figure 1: Missing Values & Year Distribution
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figure 1: Dataset Overview", fontsize=15, fontweight="bold")

# Panel A: Missing %
miss_pct_vals = df_raw.isnull().mean() * 100
bar_colors = [PAL_GREEN[0] if v == 0 else "#D32F2F" for v in miss_pct_vals.values]
axes[0].barh(miss_pct_vals.index, miss_pct_vals.values, color=bar_colors)
axes[0].set_xlabel("Missing %")
axes[0].set_title("Missing Value % per Column")
for i, v in enumerate(miss_pct_vals.values):
    axes[0].text(v + 0.05, i, f"{v:.1f}%", va="center", fontsize=9)

# Panel B: Records per Year
yr_cnt = df_raw["Crop_Year"].value_counts().sort_index()
axes[1].bar(yr_cnt.index, yr_cnt.values, color=PAL_GREEN[2], edgecolor="white")
axes[1].set_xlabel("Crop Year")
axes[1].set_ylabel("Number of Records")
axes[1].set_title("Records per Crop Year")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()
print("📊 Figure 1 displayed — take a screenshot for your report!")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 6: EDA — Figure 2: Yield Distribution (Raw vs Log)
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Figure 2: Crop Yield Distribution Analysis", fontsize=15, fontweight="bold")

# Raw distribution
axes[0].hist(df["Yield_kg_ha"], bins=80, color=PAL_GREEN[2], edgecolor="white", alpha=0.85)
axes[0].axvline(df["Yield_kg_ha"].median(), color="#D32F2F", lw=2, ls="--",
                label=f"Median = {df['Yield_kg_ha'].median():.0f}")
axes[0].set_xlabel("Yield (kg/ha)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Raw Yield — Heavy Right Skew")
axes[0].legend()

# Log-transformed
axes[1].hist(df["log_Yield"], bins=80, color=PAL_GREEN[1], edgecolor="white", alpha=0.85)
axes[1].axvline(df["log_Yield"].mean(), color="#D32F2F", lw=2, ls="--",
                label=f"Mean = {df['log_Yield'].mean():.3f}")
axes[1].set_xlabel("log₁₀(Yield)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Log₁₀ Yield — Near Normal ✅")
axes[1].legend()

# Stats box
stats = (f"Total Records : {len(df):,}\n"
         f"Mean Yield    : {df['Yield_kg_ha'].mean():,.0f} kg/ha\n"
         f"Median Yield  : {df['Yield_kg_ha'].median():,.0f} kg/ha\n"
         f"Std Dev       : {df['Yield_kg_ha'].std():,.0f} kg/ha\n"
         f"Min Yield     : {df['Yield_kg_ha'].min():,.0f} kg/ha\n"
         f"Max Yield     : {df['Yield_kg_ha'].max():,.0f} kg/ha\n"
         f"Skewness      : {df['Yield_kg_ha'].skew():.2f}\n"
         f"Kurtosis      : {df['Yield_kg_ha'].kurt():.2f}")
axes[2].text(0.08, 0.5, stats, transform=axes[2].transAxes, fontsize=11,
             verticalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor=PAL_GREEN[6], alpha=0.9))
axes[2].axis("off")
axes[2].set_title("Key Statistics")

plt.tight_layout()
plt.show()
print("📊 Figure 2 displayed — screenshot this for Section 3 (Dataset Understanding)!")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 7: EDA — Figure 3: Crop-wise Analysis & Season Box Plot
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(17, 6))
fig.suptitle("Figure 3: Crop-wise & Season-wise Yield Analysis", fontsize=15, fontweight="bold")

# Top 15 crops by mean yield (excluding sugarcane to keep scale readable)
non_extreme = df[df["Yield_kg_ha"] < df["Yield_kg_ha"].quantile(0.90)]
top_crops = non_extreme.groupby("Crop")["Yield_kg_ha"].mean().sort_values(ascending=False).head(12)
colors_bar = [PAL_GREEN[i % 5] for i in range(len(top_crops))]
axes[0].barh(top_crops.index[::-1], top_crops.values[::-1], color=colors_bar[::-1])
axes[0].set_xlabel("Mean Yield (kg/ha)")
axes[0].set_title("Top 12 Crops by Mean Yield (Excl. extreme outliers)")
for i, v in enumerate(top_crops.values[::-1]):
    axes[0].text(v + 30, i, f"{v:,.0f}", va="center", fontsize=9)

# Box plot by season
seasons_list = df["Season"].unique()
box_data = [df[df["Season"]==s]["Yield_kg_ha"].clip(upper=df["Yield_kg_ha"].quantile(0.93)).values
            for s in seasons_list]
bp = axes[1].boxplot(box_data, patch_artist=True, notch=False,
                     medianprops=dict(color="white", linewidth=2.5))
for patch, col in zip(bp["boxes"], PAL_MULTI):
    patch.set_facecolor(col)
    patch.set_alpha(0.85)
axes[1].set_xticklabels(seasons_list, rotation=20)
axes[1].set_ylabel("Yield (kg/ha)")
axes[1].set_title("Yield Distribution by Season")

plt.tight_layout()
plt.show()
print("📊 Figure 3 displayed — screenshot for Section 3!")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 8: EDA — Figure 4: Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Figure 4: Correlation Analysis", fontsize=15, fontweight="bold")

# Correlation matrix of numeric features + target
num_cols = ["Area","Annual_Rainfall","Fertiliser","Pesticide","Yield_kg_ha"]
num_cols = [c for c in num_cols if c in df.columns]
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", ax=axes[0],
            linewidths=0.5, cbar_kws={"label":"Pearson r"}, vmin=-1, vmax=1)
axes[0].set_title("Correlation Matrix (Numeric Features)")

# Rainfall vs Yield scatter
sample = df.sample(min(6000, len(df)), random_state=42)
if "Annual_Rainfall" in df.columns:
    sc = axes[1].scatter(sample["Annual_Rainfall"], sample["Yield_kg_ha"],
                         alpha=0.35, s=12, c=sample["Yield_kg_ha"],
                         cmap="YlGn", vmax=df["Yield_kg_ha"].quantile(0.93))
    plt.colorbar(sc, ax=axes[1], label="Yield (kg/ha)")
    axes[1].set_xlabel("Annual Rainfall (mm)")
    axes[1].set_ylabel("Yield (kg/ha)")
    axes[1].set_ylim(0, df["Yield_kg_ha"].quantile(0.93))
    axes[1].set_title("Rainfall vs Yield (Colour = Yield)")

plt.tight_layout()
plt.show()
print("📊 Figure 4 displayed — screenshot for Section 3!")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 9: EDA — Figure 5: Temporal Yield Trends
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(17, 6))
fig.suptitle("Figure 5: Temporal Yield Trends (2001–2022)", fontsize=15, fontweight="bold")

# Overall trend with std band
yearly = df.groupby("Crop_Year")["Yield_kg_ha"].agg(["mean","median","std"]).reset_index()
axes[0].plot(yearly["Crop_Year"], yearly["mean"],   color=PAL_GREEN[0], lw=2.5, label="Mean",   marker="o", ms=4)
axes[0].plot(yearly["Crop_Year"], yearly["median"], color=PAL_GREEN[3], lw=2,   label="Median", ls="--")
axes[0].fill_between(yearly["Crop_Year"],
                     (yearly["mean"] - yearly["std"]).clip(0),
                     yearly["mean"] + yearly["std"],
                     alpha=0.15, color=PAL_GREEN[1], label="±1 Std")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Yield (kg/ha)")
axes[0].set_title("All-India Mean Yield Trend")
axes[0].legend()
axes[0].tick_params(axis="x", rotation=40)

# Per-crop trend for key dryland crops
key_crops = ["Rice","Wheat","Soyabean","Jowar","Tur","Groundnut"]
key_crops  = [c for c in key_crops if c in df["Crop"].values]
for i, crop in enumerate(key_crops):
    sub = df[df["Crop"]==crop].groupby("Crop_Year")["Yield_kg_ha"].mean()
    axes[1].plot(sub.index, sub.values, color=PAL_MULTI[i % len(PAL_MULTI)],
                 lw=2, label=crop, marker="o", ms=3)
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Yield (kg/ha)")
axes[1].set_title("Year-wise Trend — Key Crops")
axes[1].legend(fontsize=9)
axes[1].tick_params(axis="x", rotation=40)

plt.tight_layout()
plt.show()
print("📊 Figure 5 displayed — screenshot for Section 3!")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 10: EDA — Figure 6: State & Crop Production Share
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(17, 7))
fig.suptitle("Figure 6: State-wise & Crop-wise Production Share", fontsize=15, fontweight="bold")

# State production pie
state_prod = df.groupby("State_Name")["Production"].sum().sort_values(ascending=False).head(10)
axes[0].pie(state_prod.values, labels=state_prod.index,
            autopct="%1.1f%%", startangle=140,
            colors=PAL_GREEN[:3]+PAL_MULTI[:7],
            wedgeprops=dict(edgecolor="white", linewidth=1.2))
axes[0].set_title("Top 10 States by Total Production")

# Crop area bar
crop_area = df.groupby("Crop")["Area"].sum().sort_values(ascending=False).head(12)
axes[1].bar(range(len(crop_area)), crop_area.values / 1e6,
            color=[PAL_MULTI[i % len(PAL_MULTI)] for i in range(len(crop_area))],
            edgecolor="white")
axes[1].set_xticks(range(len(crop_area)))
axes[1].set_xticklabels(crop_area.index, rotation=35, ha="right")
axes[1].set_ylabel("Total Area (Million ha)")
axes[1].set_title("Crop Area Sown — Top 12 Crops")

plt.tight_layout()
plt.show()
print("📊 Figure 6 displayed — screenshot for Section 3!")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 11: EDA — Figure 7: State × Season Heatmap
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 8))

pivot = df.groupby(["State_Name","Season"])["Yield_kg_ha"].mean().unstack(fill_value=0)
# Cap high values so colour scale is readable
cap_val = pivot.stack().quantile(0.88)
pivot_capped = pivot.clip(upper=cap_val)

sns.heatmap(pivot_capped, cmap="YlGn", annot=True, fmt=".0f",
            linewidths=0.5, ax=ax, cbar_kws={"label":"Mean Yield (kg/ha)"})
ax.set_title("Figure 7: State × Season — Mean Yield Heatmap (kg/ha)", fontsize=14, fontweight="bold")
ax.set_xlabel("Season")
ax.set_ylabel("State")
plt.xticks(rotation=25)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
print("📊 Figure 7 displayed — screenshot for Section 3!")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 12: EDA — Figure 8: Feature-Target Correlation Bar
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 5))

corr_target = X.corrwith(y).sort_values()
bar_colors  = [PAL_GREEN[1] if v >= 0 else "#D32F2F" for v in corr_target.values]
ax.barh(corr_target.index, corr_target.values, color=bar_colors)
ax.set_xlabel("Pearson Correlation with log₁₀(Yield)")
ax.set_title("Figure 8: Feature–Target Correlation", fontsize=14, fontweight="bold")
ax.axvline(0, color="black", lw=0.8)
for i, v in enumerate(corr_target.values):
    ax.text(v + (0.002 if v >= 0 else -0.002), i, f"{v:.3f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=9)

plt.tight_layout()
plt.show()
print("📊 Figure 8 displayed — screenshot for Section 4 (Methodology)!")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 13: Train-Test Split & Scaling
# ══════════════════════════════════════════════════════════════════════════════

SPLIT_YEAR = 2019
train_mask = df["Crop_Year"] <= SPLIT_YEAR

X_train, X_test = X[train_mask].copy(), X[~train_mask].copy()
y_train, y_test = y[train_mask].copy(), y[~train_mask].copy()

# Scale continuous columns
cont_cols = [c for c in ["Crop_Year","Area","Annual_Rainfall","Fertiliser","Pesticide"] if c in X.columns]
scaler = StandardScaler()
X_train[cont_cols] = scaler.fit_transform(X_train[cont_cols])
X_test[cont_cols]  = scaler.transform(X_test[cont_cols])

print("✅ Train-Test Split Complete (Temporal Split)")
print(f"   Training   : {X_train.shape[0]:,} samples (years 2001–{SPLIT_YEAR})")
print(f"   Test       : {X_test.shape[0]:,} samples (years {SPLIT_YEAR+1}–2022)")
print(f"   Features   : {X_train.shape[1]}")
print(f"\n   Train target — mean: {y_train.mean():.4f}  std: {y_train.std():.4f}")
print(f"   Test  target — mean: {y_test.mean():.4f}  std: {y_test.std():.4f}")

# Save artifacts
joblib.dump(scaler,   "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")
print("\n✅ Scaler & encoders saved!")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 14: Model Training — All 7 Models
# ══════════════════════════════════════════════════════════════════════════════

models_to_train = {
    "Linear Regression" : LinearRegression(),
    "Ridge Regression"  : Ridge(alpha=1.0),
    "Lasso Regression"  : Lasso(alpha=0.001, max_iter=5000),
    "Decision Tree"     : DecisionTreeRegressor(max_depth=15, random_state=42),
    "Random Forest"     : RandomForestRegressor(n_estimators=150, max_depth=18,
                                                min_samples_leaf=4, random_state=42, n_jobs=-1),
    "Gradient Boosting" : GradientBoostingRegressor(n_estimators=150, max_depth=5,
                                                     learning_rate=0.1, random_state=42),
    "XGBoost"           : xgb.XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.08,
                                            subsample=0.8, colsample_bytree=0.8,
                                            random_state=42, verbosity=0, n_jobs=-1),
}

results      = []
trained_models = {}

print("Training 7 models...\n")
for name, model in models_to_train.items():
    t0 = time.time()
    print(f"  🔄 {name:<25}", end="", flush=True)
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred_log = model.predict(X_test)
    y_pred     = 10 ** y_pred_log
    y_true     = 10 ** y_test.values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    cv   = cross_val_score(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)

    results.append({
        "Model"        : name,
        "RMSE (kg/ha)" : round(rmse, 1),
        "MAE (kg/ha)"  : round(mae, 1),
        "R² Test"      : round(r2, 4),
        "R² CV Mean"   : round(cv.mean(), 4),
        "R² CV Std"    : round(cv.std(), 4),
        "Train Time"   : f"{elapsed:.1f}s",
    })
    trained_models[name] = {"model": model, "y_pred": y_pred, "y_true": y_true}
    print(f"  R²={r2:.4f}  RMSE={rmse:,.0f}  ({elapsed:.1f}s)")

results_df = pd.DataFrame(results)
print("\n" + "=" * 75)
print(results_df.to_string(index=False))
print("=" * 75)


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 15: Figure 9: Model Benchmark Bar Charts
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(19, 6))
fig.suptitle("Figure 9: Model Benchmark Comparison", fontsize=15, fontweight="bold")

model_names = results_df["Model"].tolist()
r2_vals     = results_df["R² Test"].values
rmse_vals   = results_df["RMSE (kg/ha)"].values
mae_vals    = results_df["MAE (kg/ha)"].values

# Colour by performance
r2_colors  = [PAL_GREEN[0] if r >= 0.85 else PAL_GREEN[3] if r >= 0.70 else "#F57F17" for r in r2_vals]

# R² chart
bars = axes[0].bar(model_names, r2_vals, color=r2_colors, edgecolor="white")
axes[0].set_ylabel("R² Score")
axes[0].set_title("R² Score (Test Set)\nHigher = Better")
axes[0].set_ylim(0, 1.08)
axes[0].axhline(0.85, color="#D32F2F", ls="--", lw=1.5, label="Target 0.85")
axes[0].legend(fontsize=9)
for bar, v in zip(bars, r2_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
                 ha="center", fontsize=9, fontweight="bold")
axes[0].set_xticklabels(model_names, rotation=35, ha="right", fontsize=8)

# RMSE chart
axes[1].bar(model_names, rmse_vals, color=r2_colors, edgecolor="white")
axes[1].set_ylabel("RMSE (kg/ha)")
axes[1].set_title("RMSE — Lower is Better")
for i, v in enumerate(rmse_vals):
    axes[1].text(i, v + 5, f"{v:,.0f}", ha="center", fontsize=9)
axes[1].set_xticklabels(model_names, rotation=35, ha="right", fontsize=8)

# MAE chart
axes[2].bar(model_names, mae_vals, color=r2_colors, edgecolor="white")
axes[2].set_ylabel("MAE (kg/ha)")
axes[2].set_title("MAE — Lower is Better")
for i, v in enumerate(mae_vals):
    axes[2].text(i, v + 5, f"{v:,.0f}", ha="center", fontsize=9)
axes[2].set_xticklabels(model_names, rotation=35, ha="right", fontsize=8)

plt.tight_layout()
plt.show()
print("📊 Figure 9 displayed — screenshot for Section 6 (Results)!")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 16: Figure 10: Cross-Validation Score Chart
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(13, 5))

cv_means = results_df["R² CV Mean"].values
cv_stds  = results_df["R² CV Std"].values
x_pos    = np.arange(len(model_names))
bar_cols = [PAL_GREEN[0] if m >= 0.80 else PAL_GREEN[4] for m in cv_means]

ax.bar(x_pos, cv_means, yerr=cv_stds, capsize=6, color=bar_cols, edgecolor="white",
       error_kw=dict(elinewidth=2, ecolor="#D32F2F", capthick=2))
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation=35, ha="right")
ax.set_ylabel("Mean R² (5-Fold Cross-Validation)")
ax.set_ylim(0, 1.05)
ax.set_title("Figure 10: Cross-Validation R² — Mean ± Std Dev", fontsize=14, fontweight="bold")
ax.axhline(0.80, color="#D32F2F", ls="--", lw=1.5, label="0.80 threshold")
ax.legend()

# Add value labels
for i, (m, s) in enumerate(zip(cv_means, cv_stds)):
    ax.text(i, m + s + 0.01, f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.show()
print("📊 Figure 10 displayed — screenshot for Section 6!")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 17: Hyperparameter Tuning (GridSearchCV on Random Forest)
# ══════════════════════════════════════════════════════════════════════════════

print("⏳ Running GridSearchCV... (this may take 3–5 minutes)\n")

param_grid = {
    "n_estimators"    : [150, 250],
    "max_depth"       : [15, 20],
    "min_samples_leaf": [2, 4],
}

base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)
gs = GridSearchCV(base_rf, param_grid, cv=5, scoring="r2", n_jobs=-1,
                  verbose=1, return_train_score=True)
gs.fit(X_train, y_train)

best_rf = gs.best_estimator_
print(f"\n✅ Best Parameters : {gs.best_params_}")
print(f"   Best CV R²      : {gs.best_score_:.4f}")

# Evaluate tuned model
y_pred_rf_log = best_rf.predict(X_test)
y_pred_rf     = 10 ** y_pred_rf_log
y_true_arr    = 10 ** y_test.values

rmse_rf = np.sqrt(mean_squared_error(y_true_arr, y_pred_rf))
mae_rf  = mean_absolute_error(y_true_arr, y_pred_rf)
r2_rf   = r2_score(y_true_arr, y_pred_rf)

print(f"\n📈 Tuned Random Forest Results:")
print(f"   R²   = {r2_rf:.4f}")
print(f"   RMSE = {rmse_rf:,.1f} kg/ha")
print(f"   MAE  = {mae_rf:,.1f} kg/ha")

joblib.dump(best_rf, "random_forest_tuned.pkl")
print(f"\n✅ Best model saved → random_forest_tuned.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 18: Figure 11: GridSearchCV Heatmap
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5))

cv_res  = pd.DataFrame(gs.cv_results_)
pivot_gs = cv_res.pivot_table(
    index="param_max_depth",
    columns="param_n_estimators",
    values="mean_test_score"
)
sns.heatmap(pivot_gs, annot=True, fmt=".4f", cmap="YlGn", ax=ax,
            linewidths=0.5, cbar_kws={"label":"Mean CV R²"})
ax.set_title("Figure 11: GridSearchCV — max_depth × n_estimators",
             fontsize=13, fontweight="bold")
ax.set_xlabel("n_estimators")
ax.set_ylabel("max_depth")

plt.tight_layout()
plt.show()
print("📊 Figure 11 displayed — screenshot for Section 5 (Implementation)!")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 19: Figure 12: Actual vs Predicted & Residual Distribution
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(19, 6))
fig.suptitle("Figure 12: Best Model — Actual vs Predicted Analysis", fontsize=14, fontweight="bold")

cap = np.percentile(y_true_arr, 97)
mask_cap = y_true_arr < cap

# Scatter: Actual vs Predicted
axes[0].scatter(y_true_arr[mask_cap], y_pred_rf[mask_cap],
                alpha=0.35, s=8, color=PAL_GREEN[1])
axes[0].plot([0, cap], [0, cap], "r--", lw=2, label="Perfect Prediction")
axes[0].set_xlabel("Actual Yield (kg/ha)")
axes[0].set_ylabel("Predicted Yield (kg/ha)")
axes[0].set_title(f"Actual vs Predicted\nR² = {r2_rf:.4f}")
axes[0].legend()

# Residuals histogram
residuals = y_true_arr - y_pred_rf
res_clipped = residuals[np.abs(residuals) < np.percentile(np.abs(residuals), 98)]
axes[1].hist(res_clipped, bins=60, color=PAL_GREEN[2], edgecolor="white", alpha=0.85)
axes[1].axvline(0, color="#D32F2F", lw=2, ls="--", label="Zero error")
axes[1].axvline(residuals.mean(), color="#F57F17", lw=2, ls="-.",
                label=f"Mean bias={residuals.mean():.0f}")
axes[1].set_xlabel("Residual (Actual − Predicted) kg/ha")
axes[1].set_ylabel("Frequency")
axes[1].set_title(f"Residual Distribution\nMAE = {mae_rf:,.0f} kg/ha")
axes[1].legend()

# Residuals vs Predicted (check heteroscedasticity)
axes[2].scatter(y_pred_rf[mask_cap], residuals[mask_cap],
                alpha=0.3, s=8, color=PAL_MULTI[0])
axes[2].axhline(0, color="#D32F2F", lw=1.5, ls="--")
axes[2].set_xlabel("Predicted Yield (kg/ha)")
axes[2].set_ylabel("Residual (kg/ha)")
axes[2].set_title("Residuals vs Predicted\n(Heteroscedasticity Check)")

plt.tight_layout()
plt.show()
print("📊 Figure 12 displayed — screenshot for Section 6 (Results)!")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 20: Figure 13: Feature Importance (Gini + Permutation)
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Figure 13: Feature Importance Analysis — Random Forest", fontsize=14, fontweight="bold")

# Gini (MDI) importance
importances = best_rf.feature_importances_
feat_imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
feat_imp_df = feat_imp_df.sort_values("Importance", ascending=True)

bar_cols_fi = [PAL_GREEN[0] if v >= 0.10 else PAL_GREEN[2] if v >= 0.05 else PAL_GREEN[5]
               for v in feat_imp_df["Importance"]]
axes[0].barh(feat_imp_df["Feature"], feat_imp_df["Importance"], color=bar_cols_fi)
axes[0].set_xlabel("Mean Decrease in Impurity (Gini)")
axes[0].set_title("Gini (MDI) Feature Importance")
for i, v in enumerate(feat_imp_df["Importance"].values):
    axes[0].text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)

# Permutation importance
print("⏳ Computing permutation importance...")
perm = permutation_importance(best_rf, X_test, y_test,
                              n_repeats=8, random_state=42, n_jobs=-1)
perm_df = pd.DataFrame({
    "Feature"   : feature_cols,
    "Importance": perm.importances_mean,
    "Std"       : perm.importances_std
}).sort_values("Importance", ascending=True)

axes[1].barh(perm_df["Feature"], perm_df["Importance"],
             xerr=perm_df["Std"], capsize=5,
             color=[PAL_MULTI[0] if v >= 0.01 else PAL_MULTI[5] for v in perm_df["Importance"]],
             error_kw=dict(elinewidth=1.5, ecolor="#555555"))
axes[1].set_xlabel("Mean R² Decrease (Permutation)")
axes[1].set_title("Permutation Importance (Test Set)")
axes[1].axvline(0, color="black", lw=0.8)

plt.tight_layout()
plt.show()
print("📊 Figure 13 displayed — screenshot for Section 6!")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 21: Figure 14: Learning Curves
# ══════════════════════════════════════════════════════════════════════════════

print("⏳ Computing learning curves... (takes ~2 min)")

lc_model = RandomForestRegressor(n_estimators=80, max_depth=15, random_state=42, n_jobs=-1)
train_sizes, train_scores, val_scores = learning_curve(
    lc_model, X_train, y_train,
    cv=5, scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 7),
    n_jobs=-1
)

fig, ax = plt.subplots(figsize=(11, 6))

ax.plot(train_sizes, train_scores.mean(axis=1), color=PAL_GREEN[0], lw=2.5,
        marker="o", label="Train R²")
ax.fill_between(train_sizes,
                train_scores.mean(axis=1) - train_scores.std(axis=1),
                train_scores.mean(axis=1) + train_scores.std(axis=1),
                alpha=0.15, color=PAL_GREEN[0])

ax.plot(train_sizes, val_scores.mean(axis=1), color=PAL_MULTI[0], lw=2.5,
        marker="s", ls="--", label="Validation R²")
ax.fill_between(train_sizes,
                val_scores.mean(axis=1) - val_scores.std(axis=1),
                val_scores.mean(axis=1) + val_scores.std(axis=1),
                alpha=0.15, color=PAL_MULTI[0])

ax.axhline(0.85, color="#F57F17", ls=":", lw=1.5, label="0.85 target")
ax.set_xlabel("Training Set Size (samples)")
ax.set_ylabel("R² Score")
ax.set_title("Figure 14: Learning Curves — Random Forest", fontsize=14, fontweight="bold")
ax.set_ylim(0, 1.05)
ax.legend()

plt.tight_layout()
plt.show()
print("📊 Figure 14 displayed — screenshot for Section 6!")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 22: Figure 15: XGBoost vs Random Forest Direct Comparison
# ══════════════════════════════════════════════════════════════════════════════

xgb_model  = trained_models["XGBoost"]["model"]
y_xgb_pred = trained_models["XGBoost"]["y_pred"]
r2_xgb     = r2_score(y_true_arr, y_xgb_pred)
rmse_xgb   = np.sqrt(mean_squared_error(y_true_arr, y_xgb_pred))
mae_xgb    = mean_absolute_error(y_true_arr, y_xgb_pred)

print(f"XGBoost  — R²={r2_xgb:.4f}  RMSE={rmse_xgb:,.0f}  MAE={mae_xgb:,.0f}")
print(f"Rand.For — R²={r2_rf:.4f}  RMSE={rmse_rf:,.0f}  MAE={mae_rf:,.0f}")

fig, axes = plt.subplots(1, 3, figsize=(19, 6))
fig.suptitle("Figure 15: XGBoost vs Random Forest — Detailed Comparison", fontsize=14, fontweight="bold")

cap = np.percentile(y_true_arr, 97)
mk  = y_true_arr < cap

# Overlay scatter
axes[0].scatter(y_true_arr[mk], y_pred_rf[mk], alpha=0.35, s=8, color=PAL_GREEN[1], label="Random Forest")
axes[0].scatter(y_true_arr[mk], y_xgb_pred[mk], alpha=0.25, s=8, color=PAL_MULTI[0], label="XGBoost")
axes[0].plot([0, cap], [0, cap], "r--", lw=2, label="Perfect")
axes[0].set_xlabel("Actual Yield (kg/ha)")
axes[0].set_ylabel("Predicted Yield (kg/ha)")
axes[0].set_title("Actual vs Predicted — Both Models")
axes[0].legend(fontsize=9)

# Metrics grouped bar
model_labels = ["Random Forest", "XGBoost"]
r2_vals_cmp  = [r2_rf, r2_xgb]
rmse_vals_cmp= [rmse_rf, rmse_xgb]
x = np.arange(2)
axes[1].bar(x - 0.2, r2_vals_cmp, 0.35, label="R² Score", color=PAL_GREEN[1], edgecolor="white")
ax2 = axes[1].twinx()
ax2.bar(x + 0.2, rmse_vals_cmp, 0.35, label="RMSE (kg/ha)", color=PAL_MULTI[0], alpha=0.8, edgecolor="white")
axes[1].set_xticks(x)
axes[1].set_xticklabels(model_labels)
axes[1].set_ylabel("R² Score", color=PAL_GREEN[1])
ax2.set_ylabel("RMSE (kg/ha)", color=PAL_MULTI[0])
axes[1].set_title("R² vs RMSE Comparison")
axes[1].legend(loc="upper left"); ax2.legend(loc="upper right")

# XGBoost Feature importance
xgb_imp = xgb_model.feature_importances_
xgb_fi_df = pd.DataFrame({"Feature": feature_cols, "Importance": xgb_imp}).sort_values("Importance")
axes[2].barh(xgb_fi_df["Feature"], xgb_fi_df["Importance"], color=PAL_MULTI[0])
axes[2].set_xlabel("XGBoost Feature Importance Score")
axes[2].set_title("XGBoost Feature Importances")

plt.tight_layout()
plt.show()
print("📊 Figure 15 displayed — screenshot for Section 6!")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 23: Figure 16: Crop-wise & Year-wise Error Analysis
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 2, figsize=(17, 12))
fig.suptitle("Figure 16: Detailed Error & Performance Analysis", fontsize=14, fontweight="bold")

test_df = df[~train_mask].copy()
test_df["Predicted"] = y_pred_rf
test_df["Error"]     = test_df["Yield_kg_ha"] - test_df["Predicted"]

# Panel A: Crop-wise R²
crop_perf = []
for crop in test_df["Crop"].unique():
    sub = test_df[test_df["Crop"]==crop]
    if len(sub) < 15: continue
    r2_c   = r2_score(sub["Yield_kg_ha"], sub["Predicted"])
    rmse_c = np.sqrt(mean_squared_error(sub["Yield_kg_ha"], sub["Predicted"]))
    crop_perf.append({"Crop": crop, "R²": round(r2_c,3), "RMSE": round(rmse_c,0), "n": len(sub)})
crop_perf_df = pd.DataFrame(crop_perf).sort_values("R²", ascending=False)

c_colors = [PAL_GREEN[0] if r>=0.85 else PAL_GREEN[3] if r>=0.70 else "#F57F17"
            for r in crop_perf_df["R²"].values]
axes[0,0].bar(crop_perf_df["Crop"], crop_perf_df["R²"], color=c_colors, edgecolor="white")
axes[0,0].set_xticklabels(crop_perf_df["Crop"], rotation=40, ha="right")
axes[0,0].set_ylabel("R² Score")
axes[0,0].set_title("Crop-wise R² on Test Set")
axes[0,0].axhline(0.85, color="#D32F2F", ls="--", lw=1.5, label="Target")
axes[0,0].legend()

# Panel B: Year-wise RMSE
test_years = test_df["Crop_Year"].values
for yr in sorted(np.unique(test_years)):
    sub = test_df[test_df["Crop_Year"]==yr]
    rmse_yr = np.sqrt(mean_squared_error(sub["Yield_kg_ha"], sub["Predicted"]))
    axes[0,1].bar(yr, rmse_yr, color=PAL_GREEN[1], edgecolor="white")
    axes[0,1].text(yr, rmse_yr+5, f"{rmse_yr:,.0f}", ha="center", fontsize=9, fontweight="bold")
axes[0,1].set_xlabel("Test Year")
axes[0,1].set_ylabel("RMSE (kg/ha)")
axes[0,1].set_title("RMSE by Test Year")

# Panel C: Season-wise performance
if "Season" in test_df.columns:
    season_perf = []
    for s in test_df["Season"].unique():
        sub = test_df[test_df["Season"]==s]
        if len(sub) < 10: continue
        season_perf.append({
            "Season": s,
            "R²": r2_score(sub["Yield_kg_ha"], sub["Predicted"]),
            "RMSE": np.sqrt(mean_squared_error(sub["Yield_kg_ha"], sub["Predicted"]))
        })
    sp_df = pd.DataFrame(season_perf)
    axes[1,0].bar(sp_df["Season"], sp_df["R²"],
                  color=[PAL_MULTI[i] for i in range(len(sp_df))], edgecolor="white")
    axes[1,0].set_ylabel("R² Score")
    axes[1,0].set_title("Season-wise R² on Test Set")
    axes[1,0].set_xticklabels(sp_df["Season"], rotation=20)

# Panel D: Error distribution per crop (violin)
crop_show = crop_perf_df["Crop"].head(8).tolist()
error_data = [test_df[test_df["Crop"]==c]["Error"].clip(-3000,3000).values for c in crop_show]
parts = axes[1,1].violinplot(error_data, showmedians=True)
for pc in parts["bodies"]:
    pc.set_facecolor(PAL_GREEN[2])
    pc.set_alpha(0.7)
axes[1,1].set_xticks(range(1, len(crop_show)+1))
axes[1,1].set_xticklabels(crop_show, rotation=35, ha="right")
axes[1,1].axhline(0, color="#D32F2F", lw=1.5, ls="--")
axes[1,1].set_ylabel("Prediction Error (kg/ha)")
axes[1,1].set_title("Error Distribution per Crop (Violin)")

plt.tight_layout()
plt.show()
print("📊 Figure 16 displayed — screenshot for Section 6!")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 24: Figure 17: Maharashtra Deep Dive
# ══════════════════════════════════════════════════════════════════════════════

mh = df[df["State_Name"] == "Maharashtra"].copy()

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Figure 17: Maharashtra — Marathwada Region Deep Dive", fontsize=14, fontweight="bold")

# District-wise yield
dist_yield = mh.groupby("District_Name")["Yield_kg_ha"].mean().sort_values(ascending=False).head(8)
axes[0,0].bar(dist_yield.index, dist_yield.values,
              color=[PAL_GREEN[i%5] for i in range(len(dist_yield))], edgecolor="white")
axes[0,0].set_xticklabels(dist_yield.index, rotation=35, ha="right")
axes[0,0].set_ylabel("Mean Yield (kg/ha)")
axes[0,0].set_title("Top Districts by Mean Yield")

# Crop area share
mh_crop = mh.groupby("Crop")["Area"].sum().sort_values(ascending=False).head(8)
axes[0,1].pie(mh_crop.values, labels=mh_crop.index, autopct="%1.1f%%",
              colors=PAL_GREEN[:5]+PAL_MULTI[:3], startangle=140,
              wedgeprops=dict(edgecolor="white"))
axes[0,1].set_title("Maharashtra — Crop Area Share")

# Yield trend
mh_yr = mh.groupby("Crop_Year")["Yield_kg_ha"].mean()
axes[1,0].plot(mh_yr.index, mh_yr.values, color=PAL_GREEN[0], lw=2.5, marker="o", ms=4)
axes[1,0].fill_between(mh_yr.index,
                        mh_yr.values * 0.85, mh_yr.values * 1.15,
                        alpha=0.15, color=PAL_GREEN[1], label="±15% band")
axes[1,0].set_xlabel("Year")
axes[1,0].set_ylabel("Mean Yield (kg/ha)")
axes[1,0].set_title("Maharashtra Mean Yield Trend")
axes[1,0].legend()
axes[1,0].tick_params(axis="x", rotation=40)

# Soybean-specific rainfall scatter
soy_mh = mh[mh["Crop"].str.lower().str.contains("soy", na=False)]
if len(soy_mh) > 10 and "Annual_Rainfall" in soy_mh.columns:
    axes[1,1].scatter(soy_mh["Annual_Rainfall"], soy_mh["Yield_kg_ha"],
                      alpha=0.55, color=PAL_GREEN[1], s=30, edgecolors="white", lw=0.5)
    # Fit line
    z = np.polyfit(soy_mh["Annual_Rainfall"], soy_mh["Yield_kg_ha"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(soy_mh["Annual_Rainfall"].min(), soy_mh["Annual_Rainfall"].max(), 100)
    axes[1,1].plot(x_line, p(x_line), color="#D32F2F", lw=2, ls="--", label="Trend line")
    axes[1,1].set_xlabel("Annual Rainfall (mm)")
    axes[1,1].set_ylabel("Soybean Yield (kg/ha)")
    axes[1,1].set_title("Maharashtra Soybean: Rainfall vs Yield")
    axes[1,1].legend()

plt.tight_layout()
plt.show()
print("📊 Figure 17 displayed — screenshot for Section 6!")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 25: Figure 18: Final Results Dashboard
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(20, 12))
gs_layout = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)
fig.suptitle("Figure 18: FINAL RESULTS DASHBOARD — Crop Yield Prediction ML",
             fontsize=16, fontweight="bold", y=1.01)

# Panel A: Summary stats text box
ax_a = fig.add_subplot(gs_layout[0, 0])
best_model_name = results_df.loc[results_df["R² Test"].idxmax(), "Model"]
summary_text = (
    f"━━━ DATASET ━━━━━━━━━━━━━━━━━━\n"
    f" Records  : {len(df):,}\n"
    f" Crops    : {df['Crop'].nunique()}\n"
    f" States   : {df['State_Name'].nunique()}\n"
    f" Years    : {df['Crop_Year'].min()}–{df['Crop_Year'].max()}\n\n"
    f"━━━ RANDOM FOREST (Tuned) ━━━━\n"
    f" R²   = {r2_rf:.4f}\n"
    f" RMSE = {rmse_rf:,.0f} kg/ha\n"
    f" MAE  = {mae_rf:,.0f} kg/ha\n\n"
    f"━━━ XGBOOST ━━━━━━━━━━━━━━━━━━\n"
    f" R²   = {r2_xgb:.4f}\n"
    f" RMSE = {rmse_xgb:,.0f} kg/ha\n"
    f" MAE  = {mae_xgb:,.0f} kg/ha"
)
ax_a.text(0.05, 0.97, summary_text, transform=ax_a.transAxes, fontsize=10.5,
          verticalalignment="top", fontfamily="monospace",
          bbox=dict(boxstyle="round,pad=0.6", facecolor=PAL_GREEN[6], alpha=0.95))
ax_a.axis("off")
ax_a.set_title("Summary Metrics", fontweight="bold", fontsize=12)

# Panel B: R² comparison
ax_b = fig.add_subplot(gs_layout[0, 1])
ax_b.bar(results_df["Model"], results_df["R² Test"],
         color=[PAL_GREEN[0] if r >= 0.85 else PAL_GREEN[4] for r in results_df["R² Test"]],
         edgecolor="white")
ax_b.set_xticklabels(results_df["Model"], rotation=40, ha="right", fontsize=8)
ax_b.set_ylabel("R² Score")
ax_b.set_ylim(0, 1.05)
ax_b.axhline(0.85, color="#D32F2F", ls="--", lw=1.2, label="Target 0.85")
ax_b.set_title("All Models R² Comparison", fontweight="bold")
ax_b.legend(fontsize=9)

# Panel C: Feature importance
ax_c = fig.add_subplot(gs_layout[0, 2])
fi_sorted = feat_imp_df.sort_values("Importance", ascending=True)
ax_c.barh(fi_sorted["Feature"], fi_sorted["Importance"],
          color=[PAL_GREEN[0] if v >= 0.10 else PAL_GREEN[3] for v in fi_sorted["Importance"]])
ax_c.set_xlabel("Importance")
ax_c.set_title("Feature Importances (RF)", fontweight="bold")

# Panel D: Actual vs Predicted
ax_d = fig.add_subplot(gs_layout[1, 0])
cap_d = np.percentile(y_true_arr, 97)
mk_d  = y_true_arr < cap_d
ax_d.scatter(y_true_arr[mk_d], y_pred_rf[mk_d], alpha=0.3, s=5, color=PAL_GREEN[1])
ax_d.plot([0, cap_d], [0, cap_d], "r--", lw=2)
ax_d.set_xlabel("Actual (kg/ha)")
ax_d.set_ylabel("Predicted (kg/ha)")
ax_d.set_title(f"Actual vs Predicted  R²={r2_rf:.3f}", fontweight="bold")

# Panel E: Residuals
ax_e = fig.add_subplot(gs_layout[1, 1])
res     = y_true_arr - y_pred_rf
res_cap = res[np.abs(res) < np.percentile(np.abs(res), 98)]
ax_e.hist(res_cap, bins=55, color=PAL_GREEN[2], edgecolor="white", alpha=0.85)
ax_e.axvline(0, color="#D32F2F", lw=2, ls="--")
ax_e.set_xlabel("Residual (kg/ha)")
ax_e.set_ylabel("Count")
ax_e.set_title(f"Residuals  (MAE={mae_rf:,.0f} kg/ha)", fontweight="bold")

# Panel F: Yield trend with split line
ax_f = fig.add_subplot(gs_layout[1, 2])
yr_g = df.groupby("Crop_Year")["Yield_kg_ha"].mean()
ax_f.plot(yr_g.index, yr_g.values, color=PAL_GREEN[0], lw=2.5, marker="o", ms=4)
ax_f.axvspan(df["Crop_Year"].min(), SPLIT_YEAR, alpha=0.08, color=PAL_GREEN[2], label="Train")
ax_f.axvspan(SPLIT_YEAR, df["Crop_Year"].max(), alpha=0.08, color=PAL_MULTI[0], label="Test")
ax_f.axvline(SPLIT_YEAR, color="#D32F2F", ls="--", lw=1.5)
ax_f.set_xlabel("Year")
ax_f.set_ylabel("Mean Yield (kg/ha)")
ax_f.set_title("Yield Trend (Train/Test Split)", fontweight="bold")
ax_f.legend(fontsize=9)
ax_f.tick_params(axis="x", rotation=35)

plt.savefig("final_dashboard.png", dpi=160, bbox_inches="tight", facecolor=BG)
plt.show()
print("📊 Figure 18 — FINAL DASHBOARD displayed!")
print("✅ Saved as 'final_dashboard.png' — download this for your report!")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 26: Inference — Predict Yield for a New Input
# ══════════════════════════════════════════════════════════════════════════════

def predict_yield(state, district, crop, season, year, area_ha, rainfall_mm,
                  fertiliser_tonnes=None, pesticide_kg=None):
    """
    Predict crop yield for given inputs.
    Uses the trained Random Forest model.
    """
    # Encode categoricals
    def safe_encode(le, val):
        if val in le.classes_:
            return le.transform([val])[0]
        return 0   # default for unseen label

    row = {
        "State_Name_enc"    : safe_encode(encoders["State_Name"],    state),
        "District_Name_enc" : safe_encode(encoders["District_Name"], district),
        "Crop_enc"          : safe_encode(encoders["Crop"],          crop),
        "Season_enc"        : safe_encode(encoders["Season"],        season),
        "Crop_Year"         : year,
        "Area"              : area_ha,
    }
    if "Annual_Rainfall" in feature_cols:
        row["Annual_Rainfall"] = rainfall_mm
    if "Fertiliser" in feature_cols:
        row["Fertiliser"] = fertiliser_tonnes or (area_ha * 0.1)
    if "Pesticide" in feature_cols:
        row["Pesticide"] = pesticide_kg or (area_ha * 0.005)

    X_input = pd.DataFrame([row])[feature_cols]
    X_input[cont_cols] = scaler.transform(X_input[cont_cols])

    log_pred = best_rf.predict(X_input)[0]
    yield_pred = 10 ** log_pred

    # 90% confidence interval from training residual std
    log_std = y_train.std() * 0.18
    yield_low  = 10 ** (log_pred - 1.645 * log_std)
    yield_high = 10 ** (log_pred + 1.645 * log_std)

    return {
        "Predicted Yield"   : f"{yield_pred:,.0f} kg/ha",
        "90% CI Lower"      : f"{yield_low:,.0f} kg/ha",
        "90% CI Upper"      : f"{yield_high:,.0f} kg/ha",
        "Estimated Production": f"{yield_pred * area_ha / 1000:,.2f} metric tonnes",
    }

# ── Test predictions ──────────────────────────────────────────────
print("=" * 55)
print("  YIELD PREDICTION — SAMPLE INFERENCES")
print("=" * 55)

test_cases = [
    ("Maharashtra", "Latur",    "Soyabean",  "Kharif",     2022, 5000,  780),
    ("Punjab",      "Ludhiana", "Wheat",     "Rabi",       2022, 8000,  180),
    ("Karnataka",   "Belagavi", "Maize",     "Kharif",     2021, 3000,  910),
    ("Bihar",       "Patna",    "Rice",      "Kharif",     2021, 4500,  990),
    ("Rajasthan",   "Jaipur",   "Mustard",   "Rabi",       2022, 6000,  420),
]

for i, (st, dis, cr, se, yr, ar, rf) in enumerate(test_cases, 1):
    result = predict_yield(st, dis, cr, se, yr, ar, rf)
    print(f"\n  Test {i}: {cr} | {dis}, {st} | {se} {yr}")
    for k, v in result.items():
        print(f"    {k:<28}: {v}")
print("\n✅ Inference complete!")


# ══════════════════════════════════════════════════════════════════════════════
# ── CELL 27: Download All Saved Files
# ══════════════════════════════════════════════════════════════════════════════

# Download the final dashboard PNG from Colab
from google.colab import files

print("Downloading model and dashboard files...\n")

try:
    files.download("random_forest_tuned.pkl")
    print("✅ random_forest_tuned.pkl downloaded")
except: pass

try:
    files.download("final_dashboard.png")
    print("✅ final_dashboard.png downloaded")
except: pass

try:
    files.download("scaler.pkl")
    print("✅ scaler.pkl downloaded")
except: pass

print("\n🎉 All done! Your Colab run is complete.")
print("   Save the displayed figures as screenshots for your project report.")
print("   Figures to screenshot: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18")