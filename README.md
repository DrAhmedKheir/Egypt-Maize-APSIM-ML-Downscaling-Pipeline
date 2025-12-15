1️⃣ Environment Setup
rm(list=ls())
options(stringsAsFactors = FALSE)

🔹 Ensures a clean R session
🔹 Avoids accidental reuse of variables from previous runs
2️⃣ Packages
pkgs <- c("apsimx","data.table","dplyr","sf","foreach","doParallel",
          "ranger","FNN","ggplot2","DBI","RSQLite","rnaturalearth",
          "jsonlite","httr")

Why each package is needed
Package	Role
apsimx	Run APSIM NG & edit APSIMX files
DBI, RSQLite	Read APSIM .db outputs
data.table, dplyr	Fast data manipulation
sf, rnaturalearth	Spatial grids & Egypt boundary
foreach, doParallel	Parallel APSIM runs
ranger	Random Forest ML
FNN	k-nearest-neighbor interpolation
httr, jsonlite	SoilGrids API
ggplot2	Visualization
3️⃣ User Settings (Paths & Controls)
base_dir <- "D:/APSIMICARDATraining/AnotherFullScript"
work_dir <- file.path(base_dir, "EG_Maize_CMIP6_FULL")
coarse_dir <- file.path(work_dir, "coarse")

Folder Logic
EG_Maize_CMIP6_FULL/
 ├─ coarse/
 │   ├─ EG_0001/
 │   │   ├─ MaizeFull.apsimx
 │   │   ├─ EG_0001.met
 │   │   └─ MaizeFull.db   (created by APSIM)
 ├─ outputs_master/
 └─ _cache/


This structure is critical — APSIM expects the .apsimx and .met files inside each site folder.

4️⃣ Helper Functions
Logging
msg("Running APSIM…")


Adds timestamps so participants can see progress.

Numeric Safety
to_na()
impute_median()


Guarantees:

No crashes from Inf, NaN

ML models never fail due to missing data

5️⃣ Indexing Coarse APSIM Sites
site_dirs <- list.dirs(coarse_dir, recursive=FALSE)


✔ Automatically discovers all EG_#### folders
✔ Builds a run index that controls the entire pipeline

This design allows:

10 sites or 1,000 sites

No hard-coded filenames

6️⃣ APSIM Report Configuration (Key Stability Step)
report_vars <- c(
 "[Clock].Today as Date",
 "[Maize].Grain.Total.Wt*10 as Yield",
 "[Maize].Leaf.LAI as LAI",
 "[Maize].AboveGround.Wt as Biomass"
)

Why this works

Uses existing Report node

No JSON hacking

Event = EndOfDay (universally safe)

Aliases columns so DB tables are predictable

This is the main reason the original script was stable.

7️⃣ Running APSIM Safely
apsimx(file, src.dir, value="none", silent=TRUE)


Important concepts to explain to participants:

APSIM always writes to SQLite

value="none" avoids fragile R-side parsing

Database is read manually afterward (more robust)

8️⃣ Robust Database Reading
dbListTables(con)


Strategy:

Ignore internal tables (_Messages, _Simulations)

Select the Report table

Extract Date, Yield, LAI, Biomass

This makes the script version-independent across APSIM NG releases.

9️⃣ Climate Feature Extraction (.met)
What is extracted?
Variable	Meaning
DD35	Heat stress above 35 °C
HDW	Hot–Dry–Windy events
tmax_mean, tmin_mean	Seasonal temperature
rain_sum	Seasonal rainfall
radn_mean	Radiation
wind_p95	Extreme wind

✔ Features are season-filtered
✔ Scaled per year, so multi-year runs are comparable

🔟 Soil Data (ISRIC SoilGrids)

The script queries SoilGrids v2 API for:

Sand, clay, silt

Soil organic carbon

Bulk density

pH

CEC

Depths:

0–5 cm
5–15 cm
15–30 cm


Then computes a weighted 0–30 cm composite.

Why this is good practice

Physically meaningful soil depth

Cached → API called only once per site

Never crashes if SoilGrids fails

1️⃣1️⃣ Parallel APSIM Execution
foreach(i = 1:nrow(run_index)) %dopar%


✔ Each EG site runs independently
✔ Failures are logged, not fatal
✔ Produces run_log.csv for diagnostics

This is production-grade batch simulation design.

1️⃣2️⃣ Coarse APSIM Output Summary

For each site:

Max Yield (t/ha)

Max LAI

Max Biomass

Result:

coarse_apsim_summary.csv


This is the training dataset for ML.

1️⃣3️⃣ Egypt 0.1° Grid Creation
st_make_grid(cellsize = 0.1)


✔ Uses real Egypt boundary
✔ Generates ~3,000 grid cells
✔ Each cell becomes a prediction target

1️⃣4️⃣ Machine Learning Downscaling
Model
ranger::ranger()


Predictors

Coordinates

Climate summaries

Soil properties

Target

APSIM Yield (t/ha)

Fallback:

If too few points → IDW interpolation

This guarantees the script never crashes.

1️⃣5️⃣ Uncertainty & Failure Probability

Bootstrap Random Forest:

Produces Yield mean

Standard deviation

Probability Yield < threshold

This step converts yield maps into risk maps.

1️⃣6️⃣ CMIP6 Climate Scenarios (Fallback)

Simple stress-based yield penalties using:

DD35

HDW

Scenario ΔT

Outputs:

yield_cmip6_scenarios.csv


Perfect for teaching impact concepts before full CMIP pipelines.

1️⃣7️⃣ Exploratory Data Analysis (EDA)

Automatically saves:

Yield maps

Histograms & density plots

RF importance

Correlation heatmaps

Failure probability maps

Participants see results immediately.
