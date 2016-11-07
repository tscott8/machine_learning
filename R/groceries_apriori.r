library(arules)
library(arulesViz)
library(datasets)

data(Groceries)
#itemFrequencyPlot(Groceries,topN=20,type="absolute")
rules <- apriori(Groceries, parameter = list(supp = 0.001, conf = 0.8))
#rules<-apriori(data=Groceries, parameter=list(supp=0.001,conf = 0.15,minlen=2), 
#              appearance = list(default="rhs",lhs="whole milk"),
#             control = list(verbose=F))

#rules<-sort(rules, by="confidence", decreasing=TRUE)
subset.matrix <- is.subset(rules, rules)
subset.matrix[lower.tri(subset.matrix, diag=T)] <- NA
redundant <- colSums(subset.matrix, na.rm=T) >= 1
rules.pruned <- rules[!redundant]
rules<-rules.pruned
options(digits=2)
inspect(rules[1:5])
summary(rules)
#plot(rules,method="graph",interactive=TRUE,shading=NA)
