library(e1071)
library(igraph)
library(sna)

# Reading an edge list
mommyEdgesTwitter = read.csv("~/Git/machine_learning/datasets/MommyMentionMap.csv", header = FALSE)
mommyEdgesBlog = read.csv("~/Git/machine_learning/datasets/MommyBlogLinks.csv", header = FALSE)

twitterGraph = graph.data.frame(mommyEdgesTwitter)
blogGraph = graph.data.frame(mommyEdgesBlog)

# plot twitterGraph"
layout1 = layout.fruchterman.reingold(twitterGraph)
layout2 = layout.auto(twitterGraph)
plot(twitterGraph, layout=layout2)

# Compute density
graph.density(twitterGraph)

# Compute centralization measures for all nodes in the graph
dctwit = centralization.degree(twitterGraph)
cctwit = centralization.closeness(twitterGraph)
bctwit = centralization.betweenness(twitterGraph)

# get the max node names
View(dctwit)
dctwitmax = V(twitterGraph)$name[423]
View(cctwit)
cctwitmax = V(twitterGraph)$name[72]
View(bctwit)
bctwitmax = V(twitterGraph)$name[260]


# plot blogGraph
layout3 = layout.fruchterman.reingold(blogGraph)
layout4 = layout.auto(blogGraph)
plot(blogGraph, layout=layout4)

# Compute density
graph.density(blogGraph)

# Compute centralization measures for all nodes in the graph
dcblog = centralization.degree(blogGraph)
ccblog = centralization.closeness(blogGraph)
bcblog = centralization.betweenness(blogGraph)

# get the max node names
View(dcblog)
dcblogmax = V(blogGraph)$name[79]
View(ccblog)
ccblogmax = V(blogGraph)$name[356]
View(bcblog)
bcblogmax = V(blogGraph)$name[79]
