install.packages('C:\\Users\\tybug\\Downloads\\RLastFM_0.1-5.tar.gz', repos = NULL, type="source")

library (e1071)
library (datasets)
library(cluster)

songs = read.csv("~/Git/recommender_system/datasets/EvolutionPop.csv")
songData = data.frame(songs)

##Hier

#songs <- songData[ ,!(names(songData) %in% c("artist_name", "artist_name_clean", "track_name", "quarter", "first_entry"))]
songs <- songData[c("recording_id","artist_name","artist_name_clean","track_name","first_entry","quarter","year","fiveyear","decade","era")]
#distance = dist(as.matrix(scale(songs)))
#hc = hclust(distance)
#plot(hc)

##KMEANS
#scaledSongData <- scale(songs)
#songClusters <- kmodes(songs, 5)
#summary(songClusters)
#clusplot(scaledSongData, songClusters$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)