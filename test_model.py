import time
from CombineModel_jt import CombineModel
import cv2
import numpy as np
import jittor as jt

jt.flags.use_cuda = 1

# Model for face/eye1/eye2/nose/mouth
combine_model = CombineModel()

print('start')

# Specify the path to a single image
one_image_path = './test/image.jpg'

# Parameters for the CombineModel
params = [
    [0.80, 0.63, 1.0, 0.88, 0.93, 1]
]

# Index for parameter selection
i = 0

# Process the single image
print('Input file:', one_image_path)
mat_img = cv2.imread(one_image_path)
mat_img = cv2.resize(mat_img, (512, 512), interpolation=cv2.INTER_CUBIC)
mat_img = cv2.cvtColor(mat_img, cv2.COLOR_RGB2BGR)
sketch = (mat_img).astype(np.uint8)

combine_model.sex = params[i][5]
combine_model.part_weight['eye1'] = params[i][0]
combine_model.part_weight['eye2'] = params[i][1]
combine_model.part_weight['nose'] = params[i][2]
combine_model.part_weight['mouth'] = params[i][3]
combine_model.part_weight[''] = params[i][4]

combine_model.predict_shadow(mat_img)

output_file = 'ori.jpg'
print('Output file:', output_file)
cv2.imwrite(output_file, cv2.cvtColor(combine_model.generated, cv2.COLOR_BGR2RGB))
jt.gc()
