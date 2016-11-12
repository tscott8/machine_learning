library(datasets)
library(cluster)
dat = state.x77
stateData = data.frame(dat)

##Hier

# dat <- stateData[ ,!(names(stateData) %in% c("Area"))]
#dat <- stateData[c("Frost")]
#distance = dist(as.matrix(scale(dat)))
#hc = hclust(distance)
#plot(hc)

##KMEANS
scaledStateData <- scale(stateData)
stateClusters <- kmeans(scaledStateData, 3)
summary(stateClusters)
clusplot(scaledStateData, stateClusters$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)

table = NULL
for (i in 1:25) {
  temp <- kmeans(scaledStateData, i)
  table[i] = temp$withinss
}
plot(table)
# 5 seems to be the best elbow point
stateClusters <- kmeans(scaledStateData, 5)
clusplot(scaledStateData, stateClusters$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)
sort(stateClusters$cluster)
