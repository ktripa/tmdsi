#devtools::install_github("cszang/awc")
library(awc)
#get_awc(list(lon = 6.1, lat = 50.75))



# load latlon
latlon <- read.table("E:/Datasets/CPC/Monthly/latlon.txt", sep="\t", header=FALSE)
colnames(latlon) <- c("lon", "lat")

N_GRIDS <- nrow(latlon)
awc_all  <- numeric(N_GRIDS)

for (g in 1:N_GRIDS) {
  if (g %% 5000 == 0) cat(sprintf("%d / %d\n", g, N_GRIDS))
  val <- get_awc(list(lon=latlon$lon[g], lat=latlon$lat[g]))
  awc_all[g] <- ifelse(is.null(val), NA, val)
}

# save — same format: lon, lat, awc_mm
out <- data.frame(lon=latlon$lon, lat=latlon$lat, awc_mm=awc_all)
write.table(out,
            "E:/Datasets/CPC/Monthly/awc_cpc.txt",
            sep="\t", row.names=FALSE, quote=FALSE)

cat("AWC saved.\n")
cat(sprintf("AWC range: %.0f to %.0f mm\n", min(awc_all, na.rm=TRUE), max(awc_all, na.rm=TRUE)))
