# import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score,davies_bouldin_score
from statistics import mean

# The iteration
def fuzzycmeans(df,df_sampl,noc,powerOf,mi,tsee,p0,ia):
    
    if ia == 1:
#     Define a random value as Cluster membership:
        sampl = np.random.uniform(low=0, high=0.2, size=(len(df_sampl),noc))
        for row in sampl:
            if sum(row) != 1:
                x = 1-(sum(row) - row[row == row.max()])
                row[row == row.max()] = x
        df_sampl = pd.DataFrame(sampl)
        df_sampl = pd.DataFrame(np.array(df_sampl))
    
#     Cluster center calculation: Cluster membership value raised to the power
    sampl_powerOf = np.power(df_sampl,powerOf)
    
#     Cluster center calculation: Multiply the data value with the power of membership value
    df_sampl_powerOf = pd.DataFrame(sampl_powerOf)
    df_sampl_powerOf.loc['total'] = df_sampl_powerOf.sum()
    
    total_param = []
    for i in range(noc):
        globals()["dataxsampl" + str(i)] = df.multiply(np.array(sampl_powerOf[i]), axis="index")
        globals()["dataxsampl" + str(i)].loc['total'] = globals()["dataxsampl" + str(i)].sum()
        total_param.append(globals()["dataxsampl" + str(i)].loc['total'])
    
#     Calculate the Cluster center based on the total value that has been obtained
    total_miu = np.array(df_sampl_powerOf.loc['total'])
    total_miu = pd.DataFrame(total_miu)
    
    total_param = pd.DataFrame(np.array(total_param))
    
    center_cluster = total_param.divide(total_miu[0], axis="index")
    
#     Perform calculations Objective Function
    for i in range(noc):
        globals()["param_clus" + str(i)] = []
        for j in range(len(df)):
            temp = df.loc[j] - center_cluster.loc[i]
            temp = np.power(temp,powerOf)
            globals()["param_clus" + str(i)].append(temp)
    for i in range(noc):
        globals()["param_clus" + str(i)] = pd.DataFrame(np.array(globals()["param_clus" + str(i)]))
        globals()["param_clus" + str(i)] = globals()["param_clus" + str(i)].sum(axis=1)

    param_clus = pd.DataFrame(globals()["param_clus" + str(0)])

    for i in range(noc-1):
        param_clus[i+1] = pd.DataFrame(globals()["param_clus" + str(i+1)])
    
    miu_squared = df_sampl_powerOf.drop(index=['total'])
    objective_function = miu_squared.multiply(param_clus)
    
#     Perform calculations Objective Function if ia > 1
    if ia > 1:
        globals()["iterasike" + str(ia)] = (objective_function.sum()).sum()
        difference = abs(globals()["iterasike" + str(ia-1)] - globals()["iterasike" + str(ia)])
    
#     Perform calculations Objective Function if ia == 1
    if ia == 1:
        globals()["iterasike" + str(ia)] = (objective_function.sum()).sum()
        difference = abs(globals()["iterasike" + str(ia)] - p0)
    
#     Display Iteration- , Objective Function: | difference: 
    print(f'Iteration - {ia}\nObjective Function: {globals()["iterasike" + str(ia)]}| difference: {difference}')
    
#     Perform the calculation of the U partition matrix
    matrix_partition_u = (param_clus**(-1))
    matrix_partition_u['LT'] = matrix_partition_u.sum(axis=1)
    new_membership_data = matrix_partition_u.divide(matrix_partition_u['LT'], axis="index")
    new_membership_data = new_membership_data.drop(columns=['LT']) 
        
#     If Initial iteration = Max iteration or The difference between the objective function and the previous objective function <= the smallest expected error
    if ia == mi or difference <= tsee:
        
#         Display Iteration Cluster Center-
        print('\n\n')
        print(f'Iteration Cluster Center - {ia}\n',center_cluster)
        print('\n\n')
    
#     Making Tables Conclusion Degree of membership of each data in each cluster with FCM.
        df_sampl['selected clusters'] = df_sampl.max(axis=1)

        x = []
        for i in range(noc):
            globals()["x" + str(i+1)] = df_sampl[i].values.tolist()
            x.append(globals()["x" + str(i+1)])

        x0 = df_sampl['selected clusters'].values.tolist()

        cluster_h = []

        for i in x0:
            for j in range(noc):
                if i in x[j]:
                    cluster_h.append(j+1)

        df_sampl['cluster'] = cluster_h

        print(df_sampl,'\n\n')

        for i in range(noc):
            globals()["Cluster_" + str(i+1)] = []
            for count,j in enumerate(cluster_h):
                if j == i+1:
                    globals()["Cluster_" + str(i+1)].append(count+1)

        for i in range(noc):
            print(f'Cluster {i+1} = Data-',globals()["Cluster_" + str(i+1)])    

#     Making graphic illustrations before clustering and after clustering
        x = []
        x = list(df.sum(axis=1))
        data_ke_ = []
        for i in range(len(df)):
            data_ke_.append(i+1)
        y = []
        y = list(data_ke_)

        print('\nBefore Clustering\n')
        plt.plot(x,y,'o')
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.show()

        print('\n\n')
        print('After Clustering\n')

        for i in range(noc):
            centroids_x = []
            centroids_y = []
            x1 = []
            y2 = []
            for j in range(len(df)):
                if j+1 in globals()["Cluster_" + str(i+1)]:
                    x1.append(x[j])
                    y2.append(y[j])
            centroids_x.append(mean(x1))
            centroids_y.append(mean(y2))
            plt.scatter(centroids_x, centroids_y, marker = "x", s=150, linewidths = 5, zorder = 10, c= 'black')
            plt.text(centroids_x[0], centroids_y[0], f"Centroid {i+1}", fontsize=12, color='black', style='italic')
            plt.plot(x1,y2,'o',label = f'Cluster {i+1}')
        plt.legend(bbox_to_anchor = (1.0, 1), loc = 'upper left')
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.show()

        X = np.array(df)
#     The value of the silhouette coefficient score uses the library
        score = silhouette_score(X, cluster_h, metric='euclidean')
        print(f'\nSilhouette Coefficient score : {score}')

#     The value of the Davies Bouldin score uses the library   
        score2 = davies_bouldin_score(X, cluster_h)
        print(f'Davies Bouldin score: {score2}')
    
#     If Absolute The difference between the objective function and the previous objective function > the smallest expected error
    elif difference > tsee:
#     Iteration plus 1
        ia += 1
        fuzzycmeans(df,new_membership_data,noc,powerOf,mi,tsee,p0,ia) 


