# Em Sieler
# 09 August 2024

from ..scene import GaussianModel
import argparse
import sys
import os

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

#initialize empty GaussianModel and load model
model = GaussianModel()
model.load_ply(ply_path)

print(f"Successfully loaded PLY file from: {ply_path}")


