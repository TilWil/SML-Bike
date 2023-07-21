# K-Means

source("Data preparation.R")

library(data.table)
library(ggplot2)
library(factoextra)
library(NbClust)

# baseline linear model
lm_bikes <- lm(cnt ~ season + yr + mnth^2 + hr + holiday + weekday + workingday +
                 weathersit + temp + atemp + hum + windspeed + (mnth*temp),data = data) # check excluding interaction term
summary(lm_bikes)

# Clustering

set.seed(12345)
dt_kmeans <- data  #[,c("season","mnth", "holiday", "weekday", "workingday")]
dt_kmeans <- na.omit(dt_kmeans)
dt_kmeans_scaled <- data.table(scale(dt_kmeans))

model_kmeans <- kmeans(dt_kmeans, centers = 2)
model_kmeans_scaled <- kmeans(dt_kmeans_scaled, centers = 2, nstart=5)

ggplot(data=dt_kmeans, aes(x=cnt, y=season, color=as.factor(model_kmeans$cluster))) +
  geom_point() +
  geom_point(data=data.frame(model_kmeans$centers),
             shape="circle filled",
             size=7, 
             color="#000000",
             fill=c("#bbffff","#001d46"),
             alpha=0.75) +
  guides(
    color="none") +
  scale_color_manual(values=c("#bbffff","#001d46")) +
  labs(
    x="hr",
    y= "temp") +
  theme_minimal() +
  theme(axis.line = element_line(color = "#000000"))

# doesnt reveal too much differentiation between two clusters - very close to each others 
# plus tricky to compute for more clusters

fviz_cluster(object = kmeans(dt_kmeans, centers = 2), 
             data = dt_kmeans,
             geom = "point",
             shape = "circle",
             show.clust.cent = TRUE,
             stand = FALSE,
             main = "") +
  theme_minimal() +
  guides(
    color="none",
    fill="none") +
  scale_color_manual(values=c("#001d46","#bbffff")) +
  scale_fill_manual(values=c("#001d46","#bbffff")) +
  labs(
    x="hr",
    y= "temp") +
  theme_minimal() +
  theme(axis.line = element_line(color = "#000000"))

# 4 centers
fviz_cluster(object = kmeans(dt_kmeans, centers = 4), 
             data = dt_kmeans,
             geom = "point",
             shape = "circle",
             show.clust.cent = TRUE,
             stand = FALSE,
             main = "") +
  theme_minimal() +
  guides(
    color = "none",
    fill = "none") +
  scale_color_manual(values = c("#001d46", "#bbffff", "#00FF00", "#00AA00")) +
  scale_fill_manual(values = c("#001d46", "#bbffff", "#00FF00", "#00AA00")) +
  labs(
    x = "x",
    y = "y") +
  theme_minimal() +
  theme(axis.line = element_line(color = "#000000"))

# next
model_kmeans_scaled$tot.withinss

k <- 10
dt_elbow_small <- data.table("k" = 1:k, "WSS" = NA)

for (i in 1:k) {
  dt_elbow_small$k[i] <- i
  dt_elbow_small$WSS[i] <- kmeans(dt_kmeans_scaled, centers = i)$tot.withinss
}

k <- 80
dt_elbow <- data.table("k" = 1:k, "WSS" = NA)

for (i in 1:k) {
  dt_elbow$k[i] <- i
  dt_elbow$WSS[i] <- kmeans(dt_kmeans_scaled, centers = i)$tot.withinss
}

TWSS_elbow <- ggplot(data=dt_elbow, aes(x=k, y=WSS)) +
  geom_line() +
  labs(
    x="k",
    y= "Total WSS") +
  theme_minimal() +
  theme(axis.line = element_line(color = "#000000"))


TWSS_elbow_small <- ggplot(data=dt_elbow_small, aes(x=k, y=WSS)) +
  geom_line() +
  labs(
    x="k",
    y= "Total WSS") +
  theme_minimal() +
  theme(axis.line = element_line(color = "#000000"))


