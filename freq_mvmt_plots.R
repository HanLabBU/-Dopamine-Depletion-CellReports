data <- read.csv("D:\\6OHDA\\submission2\\lfpForR.csv")
freq = 'highGamma'
mvmt1 = 'rot'
mvmt2 = 'lowRot'
df2 <- reshape(data[,c('sess','Period',freq,'mvmt')], 
               idvar=c("sess", "Period"), timevar="mvmt",v.names=freq, 
               direction="wide", sep="_")
p <- ggpaired(df2, cond1 = paste(freq,mvmt1,sep="_"), 
              cond2 = paste(freq,mvmt2,sep="_"), color = "condition", 
              palette = "jco",line.color = "gray", line.size = 0.4,
              facet.by = "Period", short.panel.labs = FALSE)
p