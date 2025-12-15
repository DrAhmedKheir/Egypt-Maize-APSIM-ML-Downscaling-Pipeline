################################################################################
# Egypt maize | APSIM NextGen + ML downscaling (0.1°) + Climate (MET) + Soil (ISRIC SoilGrids) + CMIP6 (fallback) + EDA
# FULL updated robust master script
#
# Predictors in ML include:
#   - Coordinates: lon, lat
#   - Climate summaries from .met: tmax/tmin/radn/rain/rh/windspeed (+ DD35, HDW)
#   - Soil from ISRIC SoilGrids (queried at coarse sites, then interpolated to 0.1°)
#
# Expected folders:
#   D:/APSIMICARDATraining/AnotherFullScript/EG_Maize_CMIP6_FULL/coarse/EG_0001/
#     - MaizeFull.apsimx
#     - EG_0001.met (or any .met)
#     - (after run) MaizeFull.db (or any .db)
#
# Outputs:
#   .../EG_Maize_CMIP6_FULL/outputs_master/
################################################################################

rm(list=ls())
options(stringsAsFactors = FALSE)

# ------------------------------- PACKAGES ---------------------------------------
pkgs <- c(
  "apsimx","data.table","dplyr","sf","foreach","doParallel","ranger","FNN",
  "ggplot2","DBI","RSQLite","rnaturalearth","jsonlite","httr"
)
to_install <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
if(length(to_install) > 0) install.packages(to_install, dependencies = TRUE)

suppressPackageStartupMessages({
  library(apsimx)
  library(data.table)
  library(dplyr)
  library(sf)
  library(foreach)
  library(doParallel)
  library(ranger)
  library(FNN)
  library(ggplot2)
  library(DBI)
  library(RSQLite)
  library(rnaturalearth)
  library(jsonlite)
  library(httr)
})

# ------------------------------- USER SETTINGS ---------------------------------
base_dir       <- "D:/APSIMICARDATraining/AnotherFullScript"
work_dir       <- file.path(base_dir, "EG_Maize_CMIP6_FULL")
coarse_dir     <- file.path(work_dir, "coarse")
out_dir        <- file.path(work_dir, "outputs_master")
cache_dir      <- file.path(work_dir, "_cache")

apsim_template <- "MaizeFull.apsimx"
db_name_pref   <- "MaizeFull.db"

# If APSIM exe not on PATH, uncomment:
# apsimx_options(exe.path = "C:/Program Files/APSIM2025/bin/Models.exe")

# Fine grid resolution for downscaling
res_fine <- 0.1

# Fail threshold (t/ha)
yield_fail_threshold <- 3.0

# Cores
n_cores <- max(1, parallel::detectCores() - 1)

# Climate feature season months (change if you want)
season_months <- c(3,4,5,6,7,8)

# CMIP6 fallback deltas (replace later with real deltas if you have them)
cmip6 <- data.frame(
  scenario = c("SSP245","SSP585"),
  dT = c(2.1, 3.8)
)

# Uncertainty / probability-of-failure bootstrap settings
do_bootstrap_uncertainty <- TRUE
n_boot <- 50   # 30–80 typical

# SoilGrids settings
do_soilgrids <- TRUE
soil_depths_cm <- c("0-5","5-15","15-30")  # will compute weighted 0–30cm
soil_props <- c("sand","clay","silt","soc","bdod","phh2o","cec")  # SoilGrids v2 properties

dir.create(out_dir,   showWarnings = FALSE, recursive = TRUE)
dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)

# ------------------------------- HELPERS ----------------------------------------
msg <- function(...) cat(sprintf("[%s] ", format(Sys.time(), "%H:%M:%S")),
                         sprintf(...), "\n")

to_na <- function(x){
  x <- suppressWarnings(as.numeric(x))
  x[!is.finite(x)] <- NA_real_
  x
}

impute_median <- function(x){
  x <- to_na(x)
  m <- median(x, na.rm=TRUE)
  if(!is.finite(m)) return(x)
  x[is.na(x)] <- m
  x
}

xy_matrix <- function(lon, lat){
  m <- cbind(as.numeric(lon), as.numeric(lat))
  storage.mode(m) <- "double"
  colnames(m) <- c("lon","lat")
  m
}

# IDW kNN interpolation
idw_knn <- function(src_xy, src_val, trg_xy, k = 8, power = 2){
  src_val <- to_na(src_val)
  if(all(is.na(src_val))) stop("All NA values in interpolation source")
  kn <- FNN::get.knnx(src_xy, trg_xy, k = min(k, nrow(src_xy)))
  d  <- kn$nn.dist
  d[d == 0] <- 1e-12
  w  <- 1/(d^power)
  v  <- matrix(src_val[kn$nn.index], nrow = nrow(trg_xy))
  rowSums(w * v, na.rm = TRUE) / rowSums(w, na.rm = TRUE)
}

safe_ggsave <- function(path, plot, w=8, h=6, dpi=150){
  try(ggsave(path, plot, width=w, height=h, dpi=dpi), silent=TRUE)
}

# --------------------------- 1) INDEX COARSE SITES ------------------------------
msg("Indexing coarse site folders...")

site_dirs <- list.dirs(coarse_dir, recursive = FALSE, full.names = TRUE)
site_dirs <- site_dirs[grepl("^EG_[0-9]{4}$", basename(site_dirs))]

run_index <- data.frame(
  site_id   = basename(site_dirs),
  site_dir  = site_dirs,
  apsimx_fp = file.path(site_dirs, apsim_template),
  stringsAsFactors = FALSE
)
run_index <- run_index[file.exists(run_index$apsimx_fp), , drop=FALSE]
if(nrow(run_index) == 0) stop("No site folders found with ", apsim_template, " under: ", coarse_dir)

msg("Found %d site(s). Example: %s", nrow(run_index), run_index$apsimx_fp[1])

# --------------------------- 2) REPORT SETTINGS (stable aliases) ----------------
report_vars <- c(
  "[Clock].Today as Date",
  "[Maize].Grain.Total.Wt*10 as Yield",   # g/m2 -> kg/ha (x10)
  "[Maize].Leaf.LAI as LAI",
  "[Maize].AboveGround.Wt as Biomass"
)

ensure_report <- function(site_dir, apsimx_name){
  apsimx::edit_apsimx(
    file      = apsimx_name,
    src.dir   = site_dir,
    node      = "Report",
    parm      = "EventNames",
    value     = "EndOfDay",
    overwrite = TRUE
  )
  apsimx::edit_apsimx(
    file      = apsimx_name,
    src.dir   = site_dir,
    node      = "Report",
    parm      = "VariableNames",
    value     = report_vars,
    overwrite = TRUE
  )
  invisible(TRUE)
}

# --------------------------- 3) RUN APSIM (safe) --------------------------------
run_apsim_safely <- function(site_dir, apsimx_name){
  apsimx::apsimx(
    file    = apsimx_name,
    src.dir = site_dir,
    value   = "none",
    silent  = TRUE
  )
  invisible(TRUE)
}

# --------------------------- 4) ROBUST DB READ (SQLite) -------------------------
read_report_from_db <- function(site_dir, db_name = db_name_pref){
  db_fp <- file.path(site_dir, db_name)
  if(!file.exists(db_fp)){
    dbs <- list.files(site_dir, pattern="\\.db$", full.names=TRUE)
    if(length(dbs) == 0) return(NULL)
    db_fp <- dbs[1]
  }
  
  con <- DBI::dbConnect(RSQLite::SQLite(), db_fp)
  on.exit(DBI::dbDisconnect(con), add=TRUE)
  
  tabs <- DBI::dbListTables(con)
  if(length(tabs) == 0) return(NULL)
  
  rep_tabs <- setdiff(tabs, grep("^_", tabs, value = TRUE))
  pick <- rep_tabs[grepl("^Report", rep_tabs, ignore.case = TRUE)]
  if(length(pick) == 0) pick <- rep_tabs
  if(length(pick) == 0) return(NULL)
  
  df <- DBI::dbReadTable(con, pick[1])
  if(nrow(df) == 0) return(NULL)
  
  if("Date" %in% names(df)){
    try(df$Date <- as.Date(sub(" .*", "", df$Date)), silent=TRUE)
  }
  df
}

# --------------------------- 5) COORDS (from .met header) -----------------------
parse_met_lonlat <- function(site_dir){
  mf <- list.files(site_dir, pattern="\\.met$", full.names=TRUE)
  if(length(mf) == 0) return(c(lon=NA_real_, lat=NA_real_))
  x <- readLines(mf[1], warn = FALSE)
  
  lat_line <- grep("latitude",  x, ignore.case = TRUE, value = TRUE)[1]
  lon_line <- grep("longitude", x, ignore.case = TRUE, value = TRUE)[1]
  
  grab_num <- function(s){
    if(is.na(s)) return(NA_real_)
    suppressWarnings(as.numeric(sub(".*?(-?[0-9]+\\.?[0-9]*).*", "\\1", s)))
  }
  
  c(lon=grab_num(lon_line), lat=grab_num(lat_line))
}

# --------------------------- 6) MET READER (ROBUST) -----------------------------
read_met_data <- function(site_dir){
  mf <- list.files(site_dir, pattern="\\.met$", full.names=TRUE)
  if(length(mf) == 0) return(NULL)
  
  x <- readLines(mf[1], warn = FALSE)
  
  i0 <- grep("^\\s*year\\s+day\\b", x, ignore.case = TRUE)
  if(length(i0) == 0) return(NULL)
  
  hdr  <- tolower(trimws(x[i0[1]]))
  cols <- strsplit(gsub("\\s+", " ", hdr), " ")[[1]]
  
  start_line <- i0[1] + 2
  if(start_line > length(x)) return(NULL)
  
  txt <- paste(x[start_line:length(x)], collapse = "\n")
  
  dt <- tryCatch(
    data.table::fread(txt, header = FALSE, fill = TRUE, showProgress = FALSE),
    error = function(e) NULL
  )
  if(is.null(dt) || nrow(dt) == 0) return(NULL)
  if(ncol(dt) < length(cols)) return(NULL)
  
  names(dt)[1:length(cols)] <- cols
  for(nm in intersect(c("year","day","maxt","mint","radn","rain","rh","windspeed"), names(dt))){
    dt[[nm]] <- suppressWarnings(as.numeric(dt[[nm]]))
  }
  dt
}

calc_DD35 <- function(tmax) sum(pmax(tmax - 35, 0), na.rm = TRUE)
calc_HDW  <- function(tmax, tmin){
  tmean <- (tmax + tmin)/2
  sum(tmax > 35 & tmean > 30, na.rm = TRUE)
}

# climate features from met (season filtered + per-year scaling)
compute_climate_features_site <- function(site_id, site_dir, season_months = NULL){
  met <- read_met_data(site_dir)
  if(is.null(met)) return(NULL)
  if(!all(c("year","day") %in% names(met))) return(NULL)
  
  # build dates to filter months
  d <- as.Date(met$day - 1, origin = paste0(met$year, "-01-01"))
  if(!is.null(season_months)){
    keep <- as.integer(format(d, "%m")) %in% season_months
    met <- met[keep, , drop=FALSE]
  }
  if(nrow(met) == 0) return(NULL
                            
  )
  
  # per-year scaling basis
  ny <- length(unique(met$year))
  if(!is.finite(ny) || ny <= 0) ny <- 1
  
  tmax <- met$maxt
  tmin <- met$mint
  radn <- met$radn
  rain <- met$rain
  rh   <- met$rh
  wind <- met$windspeed
  
  out <- data.frame(
    site_id = site_id,
    
    # heat stress (scaled per year)
    DD35 = calc_DD35(tmax) / ny,
    HDW  = calc_HDW(tmax, tmin) / ny,
    
    # climate summaries (season)
    tmax_mean = mean(tmax, na.rm=TRUE),
    tmin_mean = mean(tmin, na.rm=TRUE),
    tmean_mean = mean((tmax+tmin)/2, na.rm=TRUE),
    tmax_p95  = as.numeric(quantile(tmax, 0.95, na.rm=TRUE)),
    tmin_p05  = as.numeric(quantile(tmin, 0.05, na.rm=TRUE)),
    
    radn_mean = mean(radn, na.rm=TRUE),
    rain_sum  = sum(rain, na.rm=TRUE) / ny,         # per year (seasonal)
    rain_p95  = as.numeric(quantile(rain, 0.95, na.rm=TRUE)),
    rh_mean   = mean(rh, na.rm=TRUE),
    wind_mean = mean(wind, na.rm=TRUE),
    wind_p95  = as.numeric(quantile(wind, 0.95, na.rm=TRUE)),
    
    stringsAsFactors = FALSE
  )
  out
}

# --------------------------- 7) SOIL (ISRIC SoilGrids v2 API) -------------------
# --------------------------- 7) SOIL (ISRIC SoilGrids v2 API) -------------------
# Always returns EXACTLY 1 ROW (with NA if request fails), so cbind/rbind never breaks.

soilgrids_point <- function(lon, lat,
                            props = soil_props,
                            depths = soil_depths_cm,
                            timeout_sec = 60){
  
  base_url <- "https://rest.isric.org/soilgrids/v2.0/properties/query"
  
  # expected output columns (always)
  out_cols <- paste0(props, "_0_30")
  
  # depth weights for 0-30 cm (0-5,5-15,15-30) => 5,10,15
  w <- c("0-5"=5, "5-15"=10, "15-30"=15)
  w <- w[depths]
  w <- w / sum(w)
  
  # default (fail-safe): one row of NA
  out_fail <- as.data.frame(as.list(setNames(rep(NA_real_, length(out_cols)), out_cols)))
  
  # helper: extract mean for prop+depth
  get_mean <- function(js, prop, depth){
    p <- js$properties[[prop]]
    if(is.null(p) || is.null(p$depths)) return(NA_real_)
    
    d0 <- as.numeric(strsplit(depth, "-")[[1]][1])
    d1 <- as.numeric(strsplit(depth, "-")[[1]][2])
    
    for(dd in p$depths){
      top <- dd$range$top_depth
      bot <- dd$range$bottom_depth
      if(isTRUE(all.equal(top, d0)) && isTRUE(all.equal(bot, d1))){
        return(to_na(dd$values$mean))
      }
    }
    NA_real_
  }
  
  q <- list(
    lon = lon,
    lat = lat,
    property = props,
    depth = depths,
    value = "mean"
  )
  
  r <- tryCatch(
    httr::GET(base_url, query = q, httr::timeout(timeout_sec)),
    error = function(e) NULL
  )
  if(is.null(r)) return(out_fail)
  if(httr::status_code(r) != 200) return(out_fail)
  
  txt <- tryCatch(httr::content(r, as="text", encoding="UTF-8"), error=function(e) NULL)
  if(is.null(txt) || !nzchar(txt)) return(out_fail)
  
  js <- tryCatch(jsonlite::fromJSON(txt, simplifyVector = FALSE), error=function(e) NULL)
  if(is.null(js) || is.null(js$properties)) return(out_fail)
  
  out <- list()
  for(p in props){
    vals <- sapply(depths, function(d) get_mean(js, p, d))
    out[[paste0(p, "_0_30")]] <- sum(vals * w, na.rm = TRUE)
  }
  
  # ensure all expected columns exist + one row
  out_df <- as.data.frame(out)
  for(cc in out_cols){
    if(!(cc %in% names(out_df))) out_df[[cc]] <- NA_real_
  }
  out_df <- out_df[, out_cols, drop=FALSE]
  if(nrow(out_df) == 0) out_df <- out_fail
  
  out_df
}

get_soil_features_for_sites <- function(site_tbl, cache_dir){
  
  soil_cache_fp <- file.path(cache_dir, "soilgrids_coarse_sites.csv")
  
  # expected columns for all sites
  out_cols <- c("site_id", paste0(soil_props, "_0_30"))
  
  # if cache exists and contains all sites, reuse
  if(file.exists(soil_cache_fp)){
    sc <- suppressWarnings(data.table::fread(soil_cache_fp))
    if("site_id" %in% names(sc) && all(site_tbl$site_id %in% sc$site_id)){
      # guarantee full column set
      for(cc in out_cols){
        if(!(cc %in% names(sc))) sc[[cc]] <- NA_real_
      }
      sc <- sc[, out_cols, with=FALSE]
      return(as.data.frame(sc))
    }
  }
  
  msg("Querying SoilGrids for coarse sites (cached)...")
  
  res <- lapply(1:nrow(site_tbl), function(i){
    sid <- site_tbl$site_id[i]
    lon <- to_na(site_tbl$lon[i])
    lat <- to_na(site_tbl$lat[i])
    
    fp <- file.path(cache_dir, paste0("soil_", sid, ".rds"))
    
    # default row (all NA)
    z_fail <- as.data.frame(as.list(setNames(rep(NA_real_, length(out_cols)-1), out_cols[-1])))
    
    if(!is.finite(lon) || !is.finite(lat)){
      return(cbind(data.frame(site_id=sid, stringsAsFactors = FALSE), z_fail))
    }
    
    # load per-site cache if exists
    if(file.exists(fp)){
      z <- tryCatch(readRDS(fp), error=function(e) NULL)
      if(is.data.frame(z) && nrow(z) == 1){
        # ensure columns
        for(cc in out_cols[-1]){
          if(!(cc %in% names(z))) z[[cc]] <- NA_real_
        }
        z <- z[, out_cols[-1], drop=FALSE]
        return(cbind(data.frame(site_id=sid, stringsAsFactors = FALSE), z))
      }
    }
    
    # query
    z <- tryCatch(soilgrids_point(lon, lat), error=function(e) NULL)
    if(!is.data.frame(z) || nrow(z) != 1){
      z <- z_fail
    } else {
      for(cc in out_cols[-1]){
        if(!(cc %in% names(z))) z[[cc]] <- NA_real_
      }
      z <- z[, out_cols[-1], drop=FALSE]
    }
    
    saveRDS(z, fp)
    cbind(data.frame(site_id=sid, stringsAsFactors = FALSE), z)
  })
  
  sc <- data.table::rbindlist(res, fill=TRUE)
  
  # guarantee columns + order
  for(cc in out_cols){
    if(!(cc %in% names(sc))) sc[[cc]] <- NA_real_
  }
  sc <- sc[, out_cols, with=FALSE]
  
  data.table::fwrite(sc, soil_cache_fp)
  as.data.frame(sc)
}

# --------------------------- 8) PARALLEL: EDIT + RUN + CHECK --------------------
msg("Running APSIM in parallel on %d core(s)...", n_cores)

cl <- makeCluster(n_cores)
registerDoParallel(cl)

run_log <- foreach(i = 1:nrow(run_index),
                   .packages = c("apsimx","DBI","RSQLite"),
                   .errorhandling = "pass") %dopar% {
                     
                     site_id   <- run_index$site_id[i]
                     site_dir  <- run_index$site_dir[i]
                     apsimx_nm <- basename(run_index$apsimx_fp[i])
                     
                     ok_edit <- tryCatch({ ensure_report(site_dir, apsimx_nm); TRUE },
                                         error=function(e) list(ok=FALSE, step="edit_report", msg=conditionMessage(e)))
                     if(is.list(ok_edit) && identical(ok_edit$ok, FALSE)){
                       return(data.frame(site_id=site_id, site_dir=site_dir, ok=FALSE, step=ok_edit$step, msg=ok_edit$msg))
                     }
                     
                     ok_run <- tryCatch({ run_apsim_safely(site_dir, apsimx_nm); TRUE },
                                        error=function(e) list(ok=FALSE, step="run_apsim", msg=conditionMessage(e)))
                     if(is.list(ok_run) && identical(ok_run$ok, FALSE)){
                       return(data.frame(site_id=site_id, site_dir=site_dir, ok=FALSE, step=ok_run$step, msg=ok_run$msg))
                     }
                     
                     df <- tryCatch(read_report_from_db(site_dir), error=function(e) NULL)
                     if(is.null(df) || nrow(df) == 0){
                       return(data.frame(site_id=site_id, site_dir=site_dir, ok=FALSE, step="read_db",
                                         msg="DB exists but no non-empty Report table found."))
                     }
                     
                     data.frame(site_id=site_id, site_dir=site_dir, ok=TRUE, step="done", msg="")
                   }

stopCluster(cl)

run_log <- data.table::rbindlist(run_log, fill=TRUE)
fwrite(run_log, file.path(out_dir, "run_log.csv"))
msg("Run success: %d / %d", sum(run_log$ok), nrow(run_log))

# --------------------------- 9) COLLECT COARSE APSIM OUTPUTS --------------------
msg("Collecting coarse outputs (max Yield/LAI/Biomass)...")

coarse_out <- lapply(1:nrow(run_index), function(i){
  site_id  <- run_index$site_id[i]
  site_dir <- run_index$site_dir[i]
  ok <- run_log$ok[match(site_id, run_log$site_id)]
  if(!isTRUE(ok)) return(NULL)
  
  df <- read_report_from_db(site_dir)
  if(is.null(df)) return(NULL)
  
  get_col <- function(nm){
    if(!(nm %in% names(df))) return(rep(NA_real_, nrow(df)))
    to_na(df[[nm]])
  }
  
  Yield   <- get_col("Yield")     # kg/ha (because g/m2*10)
  LAI     <- get_col("LAI")
  Biomass <- get_col("Biomass")
  
  Yield_t_ha <- Yield / 1000
  
  data.frame(
    site_id = site_id,
    Yield_max_t_ha = suppressWarnings(max(Yield_t_ha, na.rm=TRUE)),
    LAI_max        = suppressWarnings(max(LAI, na.rm=TRUE)),
    Biomass_max    = suppressWarnings(max(Biomass, na.rm=TRUE)),
    stringsAsFactors = FALSE
  )
})

coarse_out <- bind_rows(coarse_out)
coarse_out$Yield_max_t_ha[!is.finite(coarse_out$Yield_max_t_ha)] <- NA_real_
coarse_out$LAI_max[!is.finite(coarse_out$LAI_max)] <- NA_real_
coarse_out$Biomass_max[!is.finite(coarse_out$Biomass_max)] <- NA_real_

fwrite(coarse_out, file.path(out_dir, "coarse_apsim_summary.csv"))
msg("Coarse summary rows: %d", nrow(coarse_out))

# --------------------------- 10) COORDS + CLIMATE FEATURES ----------------------
msg("Extracting site lon/lat + climate features from MET...")

coords <- t(sapply(run_index$site_dir, parse_met_lonlat))
coords <- as.data.frame(coords)
coords$site_id <- run_index$site_id
coords$lon <- to_na(coords$lon); coords$lat <- to_na(coords$lat)

clim_feat <- bind_rows(lapply(1:nrow(run_index), function(i){
  sid <- run_index$site_id[i]
  sdir <- run_index$site_dir[i]
  ok <- run_log$ok[match(sid, run_log$site_id)]
  if(!isTRUE(ok)) return(NULL)
  compute_climate_features_site(sid, sdir, season_months = season_months)
}))

coarse_site <- coarse_out %>%
  left_join(coords, by="site_id") %>%
  left_join(clim_feat, by="site_id")

# --------------------------- 11) SOIL FEATURES (ISRIC SoilGrids) ----------------
if(do_soilgrids){
  msg("Adding soil predictors (SoilGrids)...")
  soil_sites_in <- coarse_site %>%
    select(site_id, lon, lat) %>%
    filter(is.finite(lon), is.finite(lat))
  soil_feat <- get_soil_features_for_sites(soil_sites_in, cache_dir = cache_dir)
  coarse_site <- coarse_site %>% left_join(soil_feat, by="site_id")
}

fwrite(coarse_site, file.path(out_dir, "coarse_site_features_all.csv"))

# --------------------------- 12) BUILD EGYPT 0.1° GRID --------------------------
msg("Building Egypt 0.1° grid...")

egy0_sf <- rnaturalearth::ne_countries(scale="medium", returnclass="sf") |>
  filter(admin == "Egypt") |>
  st_transform(4326)

grid_fine <- egy0_sf |>
  st_make_grid(cellsize = res_fine, what = "centers") |>
  st_as_sf() |>
  st_intersection(egy0_sf)

stn_fine <- st_coordinates(grid_fine) |> as.data.frame()
names(stn_fine) <- c("lon","lat")
stn_fine$cell_id <- sprintf("EGF_%06d", seq_len(nrow(stn_fine)))

fine <- data.frame(
  cell_id = stn_fine$cell_id,
  lon = stn_fine$lon,
  lat = stn_fine$lat
)

# --------------------------- 13) ML TRAINING (coords + climate + soil) ----------
msg("Training ML model for yield downscaling (coords + climate + soil)...")

train_df <- coarse_site %>%
  filter(is.finite(Yield_max_t_ha), is.finite(lon), is.finite(lat))

if(nrow(train_df) < 10){
  msg("WARNING: Too few training points (%d). Will downscale by IDW Yield only.", nrow(train_df))
}

# Candidate predictors: coordinates + climate + soil (automatically kept if enough finite values)
cand <- c(
  "lon","lat",
  # climate summaries
  "DD35","HDW","tmax_mean","tmin_mean","tmean_mean","tmax_p95","tmin_p05",
  "radn_mean","rain_sum","rain_p95","rh_mean","wind_mean","wind_p95",
  # soil (0-30cm)
  paste0(soil_props, "_0_30")
)

cand <- cand[cand %in% names(train_df)]
ok_pred <- cand[sapply(cand, function(v){
  sum(is.finite(to_na(train_df[[v]]))) >= max(10, floor(0.5*nrow(train_df)))
})]

# Always keep lon/lat
ok_pred <- unique(c("lon","lat", setdiff(ok_pred, c("lon","lat"))))
msg("Predictors used (%d): %s", length(ok_pred), paste(ok_pred, collapse=", "))

# Impute in training
for(v in ok_pred) train_df[[v]] <- impute_median(train_df[[v]])

# Matrices for interpolation to fine
src_xy <- xy_matrix(train_df$lon, train_df$lat)
trg_xy <- xy_matrix(fine$lon, fine$lat)

# Interpolate all non-coordinate predictors to fine grid
for(v in setdiff(ok_pred, c("lon","lat"))){
  fine[[v]] <- tryCatch(idw_knn(src_xy, train_df[[v]], trg_xy), error=function(e) NA_real_)
  fine[[v]] <- impute_median(fine[[v]])
}

# Train + predict
if(nrow(train_df) >= 10){
  ml_df <- train_df[, c("Yield_max_t_ha", ok_pred)] |> as.data.frame()
  
  set.seed(123)
  rf_model <- ranger(
    Yield_max_t_ha ~ .,
    data = ml_df,
    num.trees = 900,
    importance = "permutation",
    replace = TRUE,
    sample.fraction = 1.0,
    seed = 123,
    num.threads = max(1, min(8, n_cores))
  )
  saveRDS(rf_model, file.path(out_dir, "rf_model_yield.rds"))
  
  pred_data <- fine[, ok_pred, drop=FALSE]
  for(v in ok_pred) pred_data[[v]] <- impute_median(pred_data[[v]])
  fine$Yield_pred_t_ha <- as.numeric(predict(rf_model, data=pred_data)$predictions)
} else {
  fine$Yield_pred_t_ha <- idw_knn(src_xy, train_df$Yield_max_t_ha, trg_xy)
}

fine$Fail <- fine$Yield_pred_t_ha < yield_fail_threshold
fwrite(fine, file.path(out_dir, "yield_baseline_0p1deg.csv"))
msg("Saved baseline 0.1° yield.")

# --------------------------- 14) UNCERTAINTY + FAIL PROBABILITY -----------------
if(do_bootstrap_uncertainty && nrow(train_df) >= 10){
  msg("Bootstrap uncertainty (n_boot=%d) ...", n_boot)
  
  ml_df <- train_df[, c("Yield_max_t_ha", ok_pred)] |> as.data.frame()
  pred_data <- fine[, ok_pred, drop=FALSE]
  for(v in ok_pred) pred_data[[v]] <- impute_median(pred_data[[v]])
  
  preds <- matrix(NA_real_, nrow=nrow(fine), ncol=n_boot)
  
  set.seed(999)
  for(b in 1:n_boot){
    rb <- ranger(
      Yield_max_t_ha ~ .,
      data = ml_df,
      num.trees = 450,
      replace = TRUE,
      sample.fraction = 1.0,
      seed = 1000 + b,
      num.threads = max(1, min(6, n_cores))
    )
    preds[, b] <- as.numeric(predict(rb, data=pred_data)$predictions)
  }
  
  fine$Yield_mean_t_ha <- rowMeans(preds, na.rm=TRUE)
  fine$Yield_sd_t_ha   <- apply(preds, 1, sd, na.rm=TRUE)
  fine$P_fail          <- rowMeans(preds < yield_fail_threshold, na.rm=TRUE)
  
  fwrite(fine, file.path(out_dir, "yield_baseline_0p1deg_with_uncertainty.csv"))
  msg("Saved baseline + uncertainty + P_fail.")
}

# --------------------------- 15) CMIP6 IMPACTS (fallback) -----------------------
# Uses baseline heat-stress predictors if available (DD35/HDW interpolated), else uses zeros.
msg("Computing CMIP6 scenario impacts (fallback)...")

base_DD35 <- if("DD35" %in% names(fine)) to_na(fine$DD35) else rep(0, nrow(fine))
base_HDW  <- if("HDW"  %in% names(fine)) to_na(fine$HDW)  else rep(0, nrow(fine))
base_DD35 <- impute_median(base_DD35)
base_HDW  <- impute_median(base_HDW)

yield_cc <- merge(fine, cmip6, by=NULL) %>%
  mutate(
    DD35_f = base_DD35 + dT * 30,
    HDW_f  = base_HDW  + dT * 10,
    Yield_CC_t_ha = Yield_pred_t_ha - 0.015*(DD35_f/30) - 0.05*(HDW_f/10),
    Fail_CC = Yield_CC_t_ha < yield_fail_threshold
  )

fwrite(yield_cc, file.path(out_dir, "yield_cmip6_scenarios.csv"))
msg("Saved CMIP6 scenario yield impacts.")

# --------------------------- 16) EDA (maps, distributions, heatmaps, pairs) -----
msg("Saving EDA outputs...")

# 16.1 Baseline yield map (fine)
p_map <- ggplot(fine, aes(lon, lat, color = Yield_pred_t_ha)) +
  geom_point(size=0.25) +
  geom_sf(data=egy0_sf, fill=NA, linewidth=0.3, inherit.aes = FALSE) +
  theme_classic() +
  labs(title="Baseline maize yield (0.1°) - ML (coords + climate + soil)", color="t/ha")
safe_ggsave(file.path(out_dir, "EDA_map_yield_baseline.png"), p_map, 8, 6)

# 16.2 Histogram + density (fine)
p_hist <- ggplot(fine, aes(Yield_pred_t_ha)) +
  geom_histogram(bins=60) +
  theme_classic() +
  labs(title="Yield histogram (downscaled 0.1°)", x="t/ha", y="count")
safe_ggsave(file.path(out_dir, "EDA_hist_yield_baseline.png"), p_hist, 8, 5)

p_den <- ggplot(fine, aes(Yield_pred_t_ha)) +
  geom_density() +
  theme_classic() +
  labs(title="Yield density (downscaled 0.1°)", x="t/ha", y="density")
safe_ggsave(file.path(out_dir, "EDA_density_yield_baseline.png"), p_den, 8, 5)

# 16.3 Training points
p_train <- ggplot(train_df, aes(lon, lat, color = Yield_max_t_ha)) +
  geom_point(size=2) +
  theme_classic() +
  labs(title="Coarse APSIM yield (training points)", color="t/ha")
safe_ggsave(file.path(out_dir, "EDA_coarse_training_points.png"), p_train, 7, 5)

# 16.4 RF importance
if(file.exists(file.path(out_dir, "rf_model_yield.rds"))){
  rf <- readRDS(file.path(out_dir, "rf_model_yield.rds"))
  imp <- data.frame(var = names(rf$variable.importance),
                    importance = as.numeric(rf$variable.importance),
                    stringsAsFactors = FALSE) %>%
    arrange(desc(importance))
  p_imp <- ggplot(imp, aes(reorder(var, importance), importance)) +
    geom_col() +
    coord_flip() +
    theme_classic() +
    labs(title="RF permutation importance", x="", y="importance")
  safe_ggsave(file.path(out_dir, "EDA_rf_importance.png"), p_imp, 8, 6)
  fwrite(imp, file.path(out_dir, "EDA_rf_importance_table.csv"))
}

# 16.5 Correlation heatmap (coarse predictors + yield)
eda_cols <- c("Yield_max_t_ha", ok_pred)
eda_cols <- eda_cols[eda_cols %in% names(coarse_site)]
eda_df <- coarse_site[, eda_cols, drop=FALSE] |> as.data.frame()
for(nm in names(eda_df)) eda_df[[nm]] <- to_na(eda_df[[nm]])
eda_df <- eda_df[, sapply(eda_df, function(x) sum(is.finite(x)) >= 8), drop=FALSE]

if(ncol(eda_df) >= 3){
  C <- cor(eda_df, use="pairwise.complete.obs")
  Cdt <- as.data.table(as.table(C))
  setnames(Cdt, c("Var1","Var2","Correlation"))
  p_cor <- ggplot(Cdt, aes(Var1, Var2, fill=Correlation)) +
    geom_tile() +
    theme_classic() +
    theme(axis.text.x = element_text(angle=45, hjust=1)) +
    labs(title="Correlation heatmap (coarse: yield + predictors)", x="", y="")
  safe_ggsave(file.path(out_dir, "EDA_correlation_heatmap_coarse.png"), p_cor, 10, 8)
}

# 16.6 P_fail map + Yield SD map (if uncertainty)
if("P_fail" %in% names(fine)){
  p_pf <- ggplot(fine, aes(lon, lat, color = P_fail)) +
    geom_point(size=0.25) +
    geom_sf(data=egy0_sf, fill=NA, linewidth=0.3, inherit.aes = FALSE) +
    theme_classic() +
    labs(title=paste0("Probability of yield failure (Yield < ", yield_fail_threshold, " t/ha)"),
         color="P_fail")
  safe_ggsave(file.path(out_dir, "EDA_map_P_fail.png"), p_pf, 8, 6)
}
if("Yield_sd_t_ha" %in% names(fine)){
  p_sd <- ggplot(fine, aes(lon, lat, color = Yield_sd_t_ha)) +
    geom_point(size=0.25) +
    geom_sf(data=egy0_sf, fill=NA, linewidth=0.3, inherit.aes = FALSE) +
    theme_classic() +
    labs(title="Prediction uncertainty (SD, t/ha)", color="SD")
  safe_ggsave(file.path(out_dir, "EDA_map_uncertainty_SD.png"), p_sd, 8, 6)
}

# 16.7 Safe pairs scatter matrix (coarse)
pair_vars <- unique(c("Yield_max_t_ha", ok_pred))
pair_vars <- pair_vars[pair_vars %in% names(coarse_site)]
pair_df <- coarse_site[, pair_vars, drop=FALSE] |> as.data.frame()
for(nm in names(pair_df)) pair_df[[nm]] <- to_na(pair_df[[nm]])

# keep only columns with enough data + variation
good_col <- sapply(pair_df, function(x){
  x <- x[is.finite(x)]
  length(x) >= 10 && sd(x) > 0
})
pair_df <- pair_df[, good_col, drop=FALSE]
pair_df <- pair_df[complete.cases(pair_df), , drop=FALSE]

if(nrow(pair_df) >= 10 && ncol(pair_df) >= 2){
  png(file.path(out_dir, "EDA_pairs_scatter_matrix.png"), width=1400, height=1100, res=160)
  pairs(pair_df, main="Pairwise scatter matrix (coarse: yield + predictors)")
  dev.off()
}

# --------------------------- DONE ------------------------------------------------
msg("ALL DONE ✅")
msg("Outputs saved to: %s", out_dir)
################################################################################
