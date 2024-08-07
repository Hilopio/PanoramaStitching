from stitch_invoke import *
from Classes import Panorama

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
     "../dataset/kedr1_undist"
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
