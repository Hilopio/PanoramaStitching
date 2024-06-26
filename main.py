from stitch_invoke import *


panorama_size = (4, 5)
refer_unit = (2, 3)
postfix = '_ShiftN'
postfix = '_undistorced'

metadir = 'A12dist'
metadir = 'kedr3_dist'
metadir = 'kedr2_dist'
metadir = 'distortions'
metadir = ''


# panorama_size = (4, 5)
# refer_unit = (2, 3)
# directory = 'kedr3'
# output = 'kedr3_pano.jpg'

# panorama_size = (4, 5)
# refer_unit = (2, 3)
# directory = 'kedr1'
# output = 'kedr1_pano_without_interpolation.jpg'

# panorama_size = (5, 5)
# refer_unit = (3, 3)
# directory = 'A12.6.1'
# output= 'A12.6.1_pano.jpg'

# panorama_size = (5, 5)
# refer_unit = (3, 3)
# directory = 'A12.6.1_cropped'
# output= 'A12.6.1_cropped_H.jpg'


dirs = [
    #   "dist0", "dist005", 
    # "distortions/dist010", 
     "kedr1_undist"
    #   "dist012", "dist013", "dist014", "dist015", "dist016", "dist018", "dist025"
    #   
    #   "A12_orig",
    #   "A12_dist0", 
    #   "A12_dist005", 
    #  "A12dist/A12_dist010", 
    #   "A12_dist012", 
    #   "A12_dist013", 
    #   "A12_dist014", "A12_dist015", "A12_dist016", "A12_dist018", "A12_dist025"
    #   "kedr2_dist0", "kedr2_dist005", 
    #   "kedr2_dist/kedr2_dist010", 
    # "kedr2_dist012", "kedr2_dist013", "kedr2_dist014", "kedr2_dist015", "kedr2_dist016", "kedr2_dist018",
    #   "kedr2_dist007", "kedr2_dist008", "kedr2_dist009", "kedr2_dist011"
    #   "kedr2_orig"
    #   "kedr3_orig"
    # "kedr1",
    # "kedr1_filtered01",
    # "kedr1_filtered02",
    # "kedr1_filtered04",
    # "kedr1_filtered07",
    # "kedr1_filtered1",
    # 'dd23016dist',
        ]
scores = []
for dir in dirs:
    score = mosaicing(panorama_size, refer_unit,   dir, f"{dir}_pano.jpg") #metadir + "/" +
    scores.append(score)
    print(f"{dir} : {score}")

for i in range(len(dirs)):
    print(f"{dirs[i]} : {scores[i]}")

with open('dirs.pickle', 'wb') as handle:
        pickle.dump(dirs, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('scores.pickle', 'wb') as handle:
    pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)



'''
params:
75 best matches
homography on 25 matches

                    kedr1

distortion      with rotation           without rotation

0     :             -                   1.3304032667327756
-0.05 :             -                   0.41038508081226965
-0.10 :         0.3933702950364938      0.4142090761013085
-0.12 :         0.5485545816893012      0.5477899036859918
-0.13 :         0.6828184739141374      0.6675188221975663
-0.14 :         0.7815692956075807      0.8582110431454292
-0.15 :         0.8713786117757686      0.8700134223238022
-0.16 :         0.9129433225670888      0.9009865537212709
-0.18 :         1.2065400089531422      1.2477639826530242
-0.25 :         1.882537508970623       1.7126685989374366


A12.6

distortion  without rotation
orig  :     1.1155394784160904
0     :     1.0835382236170603
-0.05 :     0.7617165718505642
-0.10 :     0.3088768724968765
-0.12 :     0.4074266573379796
-0.13 :     0.4854850813350033
-0.14 :     0.5575223128481389
-0.15 :     0.6359568142891784
-0.16 :     0.7442166569442773
-0.18 :     0.8930894839921779
-0.25 :     1.351725514002936

kedr2

distortion  without rotation

orig : 1.6192762960692344
0 : 1.612374286004417
-0.05 : 1.1044350437364103
-0.07 : 0.720088621783183
-0.08 : 0.5787855439720618
-0.09 : 0.4320205951709878
-0.10 : 0.31728484609943564
-0.11 : 0.3409024413631671
-0.12 : 0.45031783808721976
-0.13 : 0.6340134760608827
-0.14 : 0.7723863857349158
-0.15 : 0.8366102378969631
-0.16 : 1.0567237353832857
-0.18 : 1.3310699410558646


kedr1

sigma       rmse
orig :      1.3881129421419278
0.1  :      1.2553658532670289
0.2  :      1.3333426993852795
0.4  :      1.3139537199716802
0.7  :      1.4380808792902733
1.0  :      1.28823343782483


custom undist
initial errors =  10.691392079369141
optimized errors =  0.48955347054691506


shiftn undist
initial errors =  6.772879402718617
optimized errors =  0.4001528785614424




'''