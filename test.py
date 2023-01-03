from fcm import fuzzycmeans
import pandas as pd

# Datasets taken from https://www.kaggle.com/datasets/rakeshrau/social-network-ads

data = pd.read_csv('Social_Network_Ads.csv')
data.head(10)

global df
df = data
# Replace Female & Male with 0 & 1
df['Gender'].replace(['Female', 'Male'], [0,1], inplace=True)
# Drop unnessecary columns
df = df.drop(['User ID'], axis=1)
# Replace NaN values with 0
df.replace(float('nan'), 0, inplace=True)

# Change columns name into number
columnnames = {}
count=0
for i in df.columns:
    columnnames[i] = count
    count += 1
df.rename(columns = columnnames, inplace = True)

fuzzycmeans(df,df,3,2,100,0.00001,0,1)

# The Ouput Will be:
"""
Iteration - 1
Objective Function: 344759993900.6355| difference: 344759993900.6355
Iteration - 2
Objective Function: 139045895877.29285| difference: 205714098023.34265
Iteration - 3
Objective Function: 110295924438.74577| difference: 28749971438.547073
Iteration - 4
Objective Function: 83454608565.41852| difference: 26841315873.327255
Iteration - 5
Objective Function: 59815592206.889366| difference: 23639016358.529152
Iteration - 6
Objective Function: 48456818093.49752| difference: 11358774113.391846
Iteration - 7
Objective Function: 45811737311.55294| difference: 2645080781.94458
Iteration - 8
Objective Function: 45304506689.46083| difference: 507230622.0921097
Iteration - 9
Objective Function: 45213352691.84299| difference: 91153997.61784363
Iteration - 10
Objective Function: 45196511107.82819| difference: 16841584.014793396
Iteration - 11
Objective Function: 45192967753.571014| difference: 3543354.2571792603
Iteration - 12
Objective Function: 45192039328.391716| difference: 928425.1792984009
Iteration - 13
Objective Function: 45191734661.58248| difference: 304666.80923461914
Iteration - 14
Objective Function: 45191618538.307045| difference: 116123.27543640137
Iteration - 15
Objective Function: 45191570913.2442| difference: 47625.062843322754
Iteration - 16
Objective Function: 45191550778.59813| difference: 20134.646072387695
Iteration - 17
Objective Function: 45191542165.98197| difference: 8612.616157531738
Iteration - 18
Objective Function: 45191538465.741615| difference: 3700.2403564453125
Iteration - 19
Objective Function: 45191536873.4178| difference: 1592.3238143920898
Iteration - 20
Objective Function: 45191536187.77968| difference: 685.6381225585938
Iteration - 21
Objective Function: 45191535892.48462| difference: 295.29505920410156
Iteration - 22
Objective Function: 45191535765.29428| difference: 127.19033813476562
Iteration - 23
Objective Function: 45191535710.50871| difference: 54.78556823730469
Iteration - 24
Objective Function: 45191535686.91025| difference: 23.598464965820312
Iteration - 25
Objective Function: 45191535676.74534| difference: 10.164909362792969
Iteration - 26
Objective Function: 45191535672.36685| difference: 4.378486633300781
Iteration - 27
Objective Function: 45191535670.480835| difference: 1.886016845703125
Iteration - 28
Objective Function: 45191535669.668434| difference: 0.8124008178710938
Iteration - 29
Objective Function: 45191535669.3185| difference: 0.34993743896484375
Iteration - 30
Objective Function: 45191535669.16776| difference: 0.15073394775390625
Iteration - 31
Objective Function: 45191535669.10284| difference: 0.0649261474609375
Iteration - 32
Objective Function: 45191535669.074875| difference: 0.02796173095703125
Iteration - 33
Objective Function: 45191535669.06282| difference: 0.012054443359375
Iteration - 34
Objective Function: 45191535669.05763| difference: 0.00518798828125
Iteration - 35
Objective Function: 45191535669.0554| difference: 0.00223541259765625
Iteration - 36
Objective Function: 45191535669.05443| difference: 0.00096893310546875
Iteration - 37
Objective Function: 45191535669.05402| difference: 0.00040435791015625
Iteration - 38
Objective Function: 45191535669.05385| difference: 0.00017547607421875
Iteration - 39
Objective Function: 45191535669.05377| difference: 7.62939453125e-05
Iteration - 40
Objective Function: 45191535669.053734| difference: 3.814697265625e-05
Iteration - 41
Objective Function: 45191535669.05373| difference: 7.62939453125e-06



Iteration Cluster Center - 41
           0          1              2         3
0  0.378891  42.250706  127941.200878  0.842990
1  0.539276  35.845495   75211.335134  0.194012
2  0.474458  37.833125   33612.980390  0.346812



            0         1         2  selected clusters  cluster
0    0.016574  0.062255  0.921171           0.921171        3
1    0.014772  0.056462  0.928766           0.928766        3
2    0.011132  0.077406  0.911462           0.911462        3
3    0.039408  0.597991  0.362601           0.597991        2
4    0.000231  0.999423  0.000346           0.999423        2
..        ...       ...       ...                ...      ...
395  0.006850  0.044241  0.948909           0.948909        3
396  0.009726  0.039293  0.950980           0.950980        3
397  0.014772  0.056462  0.928766           0.928766        3
398  0.000042  0.000211  0.999747           0.999747        3
399  0.000671  0.003690  0.995639           0.995639        3

[400 rows x 5 columns] 


Cluster 1 = Data- [8, 32, 43, 49, 60, 64, 74, 76, 86, 92, 98, 104, 138, 160, 169, 172, 173, 183, 203, 207, 208, 209, 212, 216, 220, 223, 224, 227, 228, 231, 233, 235, 240, 241, 244, 246, 248, 253, 254, 260, 262, 263, 266, 269, 271, 274, 285, 288, 291, 298, 300, 303, 307, 308, 309, 314, 317, 321, 325, 329, 330, 332, 337, 340, 341, 345, 348, 351, 361, 365, 374, 383]
Cluster 2 = Data- [4, 5, 6, 7, 10, 11, 13, 15, 16, 31, 35, 39, 45, 47, 50, 53, 55, 56, 58, 62, 63, 65, 66, 68, 69, 70, 71, 79, 81, 84, 85, 87, 88, 89, 91, 95, 99, 101, 102, 103, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 119, 120, 121, 122, 123, 126, 127, 130, 131, 133, 134, 135, 136, 137, 139, 141, 142, 143, 144, 146, 147, 149, 150, 153, 157, 158, 161, 162, 165, 166, 167, 168, 171, 175, 182, 185, 186, 187, 189, 191, 194, 195, 197, 199, 202, 204, 205, 211, 214, 217, 218, 219, 221, 222, 225, 229, 230, 234, 236, 237, 238, 239, 242, 243, 245, 250, 256, 257, 258, 259, 261, 264, 265, 267, 268, 270, 272, 276, 277, 278, 281, 282, 283, 286, 287, 289, 290, 292, 294, 295, 296, 297, 299, 302, 304, 305, 311, 312, 315, 316, 318, 320, 322, 326, 327, 328, 333, 334, 335, 338, 339, 342, 343, 346, 347, 349, 350, 352, 353, 354, 355, 357, 358, 359, 364, 368, 369, 372, 373, 375, 377, 379, 381, 386, 388, 395]
Cluster 3 = Data- [1, 2, 3, 9, 12, 14, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 36, 37, 38, 40, 41, 42, 44, 46, 48, 51, 52, 54, 57, 59, 61, 67, 72, 73, 75, 77, 78, 80, 82, 83, 90, 93, 94, 96, 97, 100, 105, 107, 118, 124, 125, 128, 129, 132, 140, 145, 148, 151, 152, 154, 155, 156, 159, 163, 164, 170, 174, 176, 177, 178, 179, 180, 181, 184, 188, 190, 192, 193, 196, 198, 200, 201, 206, 210, 213, 215, 226, 232, 247, 249, 251, 252, 255, 273, 275, 279, 280, 284, 293, 301, 306, 310, 313, 319, 323, 324, 331, 336, 344, 356, 360, 362, 363, 366, 367, 370, 371, 376, 378, 380, 382, 384, 385, 387, 389, 390, 391, 392, 393, 394, 396, 397, 398, 399, 400]

Before Clustering

*

After Clustering

*

Silhouette Coefficient score : 0.6018615999609797
Davies Bouldin score: 0.47921954479607093 """
