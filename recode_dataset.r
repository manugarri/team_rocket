
rm(list=ls())
setwd("~/Dropbox/SpaceApps2016")

dataset <- read.csv("MPCORB_PHA_MERGE.csv")
dataset$Desn <- as.numeric(levels(dataset$Desn))[dataset$Desn]
levels(dataset$Epoch) <- 1:length(levels(dataset$Epoch))
dataset$Epoch <- as.numeric(levels(dataset$Epoch))[dataset$Epoch]
dataset$zero <- as.numeric(levels(dataset$zero))[dataset$zero]
levels(dataset$Reference) <- 1:length(levels(dataset$Reference))
dataset$Reference <- as.numeric(dataset$Reference)
levels(dataset$Arc) <- 1:length(levels(dataset$Arc))
dataset$Arc <- as.numeric(dataset$Arc)
levels(dataset$Computer) <- 1:length(dataset$Computer)
dataset$Computer <- as.numeric(dataset$Computer)
levels(dataset$zeros_id) <- 1:length(dataset$zeros_id)
dataset$zeros_id <- as.numeric(dataset$zeros_id)
levels(dataset$Perts) <- 1:length(dataset$Perts)
dataset$Perts <- as.numeric(dataset$Perts)
levels(dataset$number) <- 1:length(dataset$number)
dataset$number <- as.numeric(dataset$number)
levels(dataset$Name) <- 1:length(dataset$Name)
dataset$Name <- as.numeric(dataset$Name)
dataset$NEO_flag <- as.numeric(dataset$NEO_flag)

resumen <- summary(dataset)
dataset <- dataset[complete.cases(dataset),]
dataset <- dataset[,c("H","G","Epoch","M","Peri","Node","Incl","e","n","a","X.Obs","X.Opp","Arc","rms","NEO_flag")]
cor(dataset,method=c("spearman"))

# mod <- glm(NEO_flag~H+G+Epoch+M+Peri+Node+Incl+e+n+a+X.Obs+X.Opp+Arc+rms-1,data=dataset,binomial(link = "logit"))
# R <- (mod$null.deviance-mod$deviance)/mod$null.deviance

mod <- glm(NEO_flag~H+G+M+Incl+e+n+a+X.Obs+Arc+rms-1,data=dataset,family=poisson(link ="log"))
R <- (mod$null.deviance-mod$deviance)/mod$null.deviance
summary(mod)
table(round(exp(mod$linear.predictors),0))

dataset_0 <- dataset[dataset$NEO_flag==0,]
dataset_1 <- dataset[dataset$NEO_flag==1,]
dataset_0 <- dataset_0[sample(c(1:dim(dataset_0)[1]),dim(dataset_1)[1],replace=F),]
dataset_fin <- as.data.frame(rbind(dataset_0,dataset_1))
# 
# mod <- glm(NEO_flag~H+G+M+Incl+e+n+a+X.Obs+Arc+rms-1,data=dataset_fin,family=poisson(link ="log"))
# R <- (mod$null.deviance-mod$deviance)/mod$null.deviance
# summary(mod)
# 
mod <- glm(NEO_flag~H+G+Epoch+M+Peri+Node+Incl+e+n+a+X.Obs+X.Opp+Arc+rms-1,data=dataset_fin,family=poisson(link = "log"))
R <- (mod$null.deviance-mod$deviance)/mod$null.deviance
summary(mod)
# 
# write.csv2(dataset_fin,"Adrian_balanceado.csv")

dataset_fin$log_H <- log(dataset_fin$H)
dataset_fin$log_X.Obs <- log(dataset_fin$X.Obs)
dataset_fin$log_X.Opp <- log(dataset_fin$X.Opp)
dataset_fin$log_Arc <- log(dataset_fin$Arc)
mod <- glm(NEO_flag~H+M+Peri+Node+Incl+e+n+a+log_Arc-1,data=dataset_fin,family=gaussian(link = "identity"))
R <- (mod$null.deviance-mod$deviance)/mod$null.deviance
summary(mod)


claudia <- dataset[1:10,c("M","Peri","Node","Incl","e","a","NEO_flag")]
neo <- dataset[dataset$NEO_flag==1,c("M","Peri","Node","Incl","e","a","NEO_flag")]
claudia <- rbind(claudia,neo[c(1,2),])
colnames(claudia) <- c("Mean_anomaly","Pheriasis","Log_ascending","inclination","Semi_mayor_axis","Excentricity","NEO_flag")
write.csv2(claudia,"10_ejemplos.csv")
