# "Enhanced Fuzzy C-Means (FCM) Clustering Algorithm Implementation in Python"

This is an implementation of the Fuzzy C-Means (FCM) clustering algorithm in Python; .py and .ipynb file extension. The FCM algorithm is an extension of the traditional K-Means clustering algorithm that allows each data point to belong to multiple clusters with different degrees of membership.
### Requirements
* pandas
* numpy
* matplotlib
* scikit-learn (for evaluating the quality of the clusters)
### Usage
Usage on .ipynb file extension can be seen in [testingFCM.ipynb](https://github.com/agung-madani/fuzzy-cmeans-clustering-algorithm/blob/main/testingFCM.ipynb)

Usage on .py file extension can be seen in [tes.py](https://github.com/agung-madani/fuzzy-cmeans-clustering-algorithm/blob/main/test.py)
### Parameters 
Parameters for fuzzycmeans function inside [fcm.ipynb](https://github.com/agung-madani/fuzzy-cmeans-clustering-algorithm/blob/main/fcm.ipynb) or [fcm.py](https://github.com/agung-madani/fuzzy-cmeans-clustering-algorithm/blob/main/fcm.py) :<br>
`df`: a dataframe representing the data before clustered.<br>
`df_sampl`: a dataframe representing the data for iteration.<br>
`noc`: the number of clusters to be formed.<br>
`powerOf`: a parameter that controls the "fuzziness" of the membership degrees.<br>
`mi`: the maximum number of iterations to run the algorithm.<br>
`tsee`: a stopping criterion for the algorithm. If the difference between the objective function values in two consecutive iterations is less than tsee, the algorithm will stop.<br>
`p0`: initial objective function.<br>
`ia`: the current initial number.
### Evaluation
To evaluate the quality of the clusters I am using the silhouette score and the Davies-Bouldin index. The silhouette score is a measure of how similar the data points within a cluster are to each other, and how different they are from the data points in other clusters. A higher silhouette score indicates a better quality of clusters. The Davies-Bouldin index is a measure of the compactness and separation of the clusters. A lower Davies-Bouldin index indicates a better quality of clusters.
