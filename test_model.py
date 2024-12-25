import sys
import cv2
import numpy as np
from CombineModel_jt import CombineModel
import jittor as jt

jt.flags.use_cuda = 1

# Get paths from command line arguments (input image and output image)
if len(sys.argv) != 3:
    print("Usage: python3 test_model.py <input_image_path> <output_image_path>")
    sys.exit(1)

input_image_path = sys.argv[1]
output_image_path = sys.argv[2]

# Parameters for CombineModel (you can adjust these as needed)
params = [
    [0.80, 0.63, 1.0, 0.88, 0.93, 1]  # You can change these parameters if needed
]

combine_model = CombineModel()

# Load and process the image
print(f"Processing input image: {input_image_path}")
mat_img = cv2.imread(input_image_path)
if mat_img is None:
    print(f"Error: Could not open or find the image: {input_image_path}")
    sys.exit(1)

mat_img = cv2.resize(mat_img, (512, 512), interpolation=cv2.INTER_CUBIC)
mat_img = cv2.cvtColor(mat_img, cv2.COLOR_RGB2BGR)

# Set model parameters for face/eye1/eye2/nose/mouth
combine_model.sex = params[0][5]
combine_model.part_weight['eye1'] = params[0][0]
combine_model.part_weight['eye2'] = params[0][1]
combine_model.part_weight['nose'] = params[0][2]
combine_model.part_weight['mouth'] = params[0][3]
combine_model.part_weight[''] = params[0][4]

# Perform model prediction
combine_model.predict_shadow(mat_img)

# Save the output image
print(f"Saving generated image to: {output_image_path}")
cv2.imwrite(output_image_path, cv2.cvtColor(combine_model.generated, cv2.COLOR_BGR2RGB))

# Perform garbage collection to free memory
jt.gc()
print("Model prediction completed successfully.")
