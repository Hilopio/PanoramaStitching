import cv2
import numpy as np
import time
from stitch_functions import *
from stitch_pipline import *
import pickle

# panorama_size = (4, 5)
# refer_unit = (2, 3)
# directory = 'kedr3'
# output = 'kedr3_pano.jpg'

panorama_size = (4, 5)
refer_unit = (2, 3)
directory = 'CuM1'
output = 'CuM1_pano.jpg'

# panorama_size = (5, 5)
# refer_unit = (3, 3)
# directory = 'A12.6.1'
# output= 'A12.6.1_pano.jpg'

# panorama_size = (5, 5)
# refer_unit = (3, 3)
# directory = 'A12.6.1_cropped'
# output= 'A12.6.1_cropped_H.jpg'


start_time = time.time()

print("finding homographies...")
hhs, vhs, vinliers, hinliers = find_all_homographies(panorama_size, refer_unit, directory)
np.save('horizhomos.npy', hhs)
np.save('vertichomos.npy', vhs)
with open('horizinliers.pickle', 'wb') as handle:
    pickle.dump(hinliers, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('verticinliers.pickle', 'wb') as handle:
    pickle.dump(vinliers, handle, protocol=pickle.HIGHEST_PROTOCOL)


print("transforming...")
hhs = np.load('horizhomos.npy')
vhs = np.load('vertichomos.npy')
with open('horizinliers.pickle', 'rb') as handle:
    hinliers = pickle.load(handle)
with open('verticinliers.pickle', 'rb') as handle:
    vinliers = pickle.load(handle)
initial_transforms = get_averaged_transforms(panorama_size, refer_unit, hhs, vhs)
np.save('initial_transforms.npy', initial_transforms)


print("optimizing...")
initial_transforms = np.load('initial_transforms.npy')
final_transforms = optimization(initial_transforms, panorama_size, refer_unit, vinliers, hinliers)
np.save('final_transforms.npy', initial_transforms)

print("stitching...")
ans = transform_and_stitch(panorama_size, refer_unit, directory, final_transforms)

print("drawing...")
final_pic = ans.astype('uint8') 
final_bgr = cv2.cvtColor(final_pic, cv2.COLOR_RGB2BGR)
cv2.imwrite(output, final_bgr)
end_time = time.time()


all_time = end_time - start_time
print(f"completed! time spent: {round(all_time) // 60} min {all_time % 60} sec")