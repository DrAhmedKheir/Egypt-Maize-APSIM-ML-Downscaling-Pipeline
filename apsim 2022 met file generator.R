
################################################################################
## Author: Ahmed Kheir
## Purpose: To create .met files from NASA power for APSIM workshop 2025
## Required information: Start Date, End Date, Latitude, and Longitude 
################################################################################

## Packages to run script

# Code to install r packages (only needs to be run the first time)
install.packages(c("tidyverse", "nasapower"))

# Code to activate r packages (needs to be run EVERY time)
library(tidyverse)
library(nasapower)



################################################################################
## Assign required path, file name, location, and time span
## These are the only steps required to change in order to create a met file!

# 1 Assigns path where .met files will be stored
path <- setwd("C:/Users/mebaum/Desktop/met files/APSIM 2022 workshop")  

# 2 Name of .met file
filename <- "/name.met"                         

# 3 Defines location (Longitude, Latitude)
location <- c(-82.89, 39.92)                  

# 4 Start date, End date ("yyyy-mm-dd")
start_end_date <- c("1984-01-01", "2021-12-31")    


# 5 Highlight and run all script below to create met file 
################################################################################

### MAKES LAT, TAV, AND AMP HEADERS
header <- get_power(community = "ag",
                    lonlat = location,                               ## LONGITUDE,  LATITUDE
                    pars = c("T2M_MAX",                              ## Mean daily max temp at 2 Meters above ground level (C)
                             "T2M_MIN",                              ## Mean daily max temp at 2 Meters above ground level (C)
                             "PRECTOTCORR",                          ## daily cumulative precipitation (mm)
                             "ALLSKY_SFC_SW_DWN"),                   ## sky shortwave downward irradiance (MJ/m^2/day)
                    dates = start_end_date,                          ## START DATE, END DATE ("yyyy-mm-dd")
                    temporal_api = "daily") %>%          
  mutate(Latitude = LAT, year = YEAR, day = DOY,
         radn = ALLSKY_SFC_SW_DWN, maxt = T2M_MAX, mint = T2M_MIN) %>%  ## puts variables to APSIM friendly names
  mutate(dailyt = (maxt + mint)/2)%>%                                   ## computes mean daily temp
  group_by(year) %>%                 
  mutate(tav = mean(dailyt)) %>%                                        ## computes mean annual temp
  ungroup()%>%
  group_by(MM)%>%
  mutate(avgMM = mean(dailyt))%>%                                       ## computes mean monthly temp 
  group_by(year)%>%
  mutate(amp = max(avgMM)- min(avgMM)) %>%                              ## computes amplitude in monthly temp 
  group_by(Latitude)%>%
  summarise(tav = mean(tav),
            amp = mean(amp))%>% 
  select(Latitude, tav, amp) %>%                                        ## removes unnecessary rows
  gather(variable, value, Latitude,tav, amp)%>%
  mutate(variable = ifelse(variable == "Latitude", "Latitude = ",
                           ifelse(variable == "tav", "tav = ",
                                  ifelse(variable == "amp", "amp = ", variable)))) ## makes row names readable for APSIM


## CREATES HEADER TO BE SAVED IN FILE
myfile <- paste0(path, filename)                     

## WRITES HEADER FILE 
write.table(data.frame(header), myfile,              
            quote = F,
            row.names = F,
            col.names = F)



## COMPUTES DAILY MAXT MINT RADN AND PRECIPITATION TO BE SAVED IN .met FILE
metdata <- get_power(community = "ag",
                     lonlat = location,                               ## LONGITUDE,  LATITUDE
                     pars = c("PRECTOTCORR",                          ## Precipitation (mm day-1) 
                              "ALLSKY_SFC_SW_DWN",                    ## sky shortwave downward irradiance (MJ/m^2/day)
                              "T2M_MIN",                              ## Mean daily min temp at 2 Meters (C)
                              "T2M_MAX"),                             ## Mean daily max temp at 2 Meters (C) 
                     dates = start_end_date,                          ## START DATE, END DATE ("yyyy-mm-dd")
                     temporal_api = "daily") %>%          
  mutate(Latitude = LAT, year = YEAR, day = DOY,
         radn = ALLSKY_SFC_SW_DWN, maxt = T2M_MAX, 
         mint = T2M_MIN,
         rain = PRECTOTCORR,
         radn = ifelse(radn == -999, round(abs(mean(radn)), digits = 2), radn),            ## renames all not available data 
         rain = ifelse(rain == -999, "?", rain),
         maxt = ifelse(maxt == -999, "?", maxt),
         mint = ifelse(mint == -999, "?", mint)) %>%
  select(year, day, radn, maxt, mint, rain) %>%
  mutate(year = factor(year),
         day = factor(day),
         radn = factor(radn),
         maxt = factor(maxt),
         mint = factor(mint),
         rain = factor(rain)) %>%                 
  add_row(year = "()",                                     ## writes row with column units 
          day ="()",
          radn = "(MJ/m^2)",
          maxt = "(oC)",
          mint = "(oC)",
          rain = "(mm)",
          .before = 1)


## COMBINES HEADER AND NASS POWER DATA INTO ONE MET FILE 
write.table(metdata, myfile, 
            append = T,
            sep = '     ',    
            quote = F,             
            row.names = F)         
