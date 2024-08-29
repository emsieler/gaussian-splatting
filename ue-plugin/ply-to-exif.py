# Em Sieler
# 09 August 2024

from ..scene import GaussianModel
import argparse
import sys
import os
import numpy as np
from PIL import Image
import math

def main():
    #parse command line args (filepath)
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', type=str, required=True, help="The PLY filepath")
    args = parser.parse_args()

    if not args.p:
        print("No filepath provided")
        sys.exit(1)

    ply_path = args.p 
    if not os.path.isfile(ply_path):
        print(f"Error: The filepath '{ply_path}' does not exist.")
        sys.exit(1)
    
    model = GaussianModel()
    model.load_ply_to_exif(ply_path)

print(f"Successfully loaded PLY file from: {ply_path}")

### 
###
###

##this needs to move to GaussianModel class but for simplicity im gonna keep it here for now

def load_ply_to_exif(self, path, use_train_test_exp = False):
    plydata = PlyData.read(path)
    if use_train_test_exp:
        exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
        if os.path.exists(exposure_file):
            with open(exposure_file, "r") as f:
                exposures = json.load(f)
            self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
            print(f"Pretrained exposures loaded.")
        else:
            print(f"No exposure to be loaded at {exposure_file}")
            self.pretrained_exposures = None

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)

#array stacked across columns (dim 1)
#3 columns, x, y, z, len = num_gaussians

##Niagara has limit of 2M particles
    gaussian_count = xyz.shape[0] 
    MAX_GAUSSIANS = 2000000

    if gaussian_count > MAX_GAUSSIANS:
        print(f"Error: Number of Gaussians ({num_gaussians}) exceeds the  maximum limit of {MAX_GAUSSIANS}. Exiting.")
        sys.exit(1)
    
    #create EXIF
    img_size = math.ceil(math.sqrt(num_gaussians))
    print(f"Initializing an image with size: {image_size}x{image_size}")

    data = np.random.randint(0, 256, (img_size, img_size, 4), dtype=np.uint8)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

## add my code here
    #save as image (RGBA PNG)
    image = Image.fromarray(data, 'RGBA')
    image.save("output.png", "PNG") #add support for custom filename arg

## calls main method when script called from cmd line
if __name__ == "__main__":
    main()
