import os
import shutil
import argparse
from PIL import Image, ImageSequence

import torch
from torch import nn
from torch.nn import functional as F

from utils import *
from model import *




parser = argparse.ArgumentParser(description='MakeGIF')
parser.add_argument('--ModelWeightPath', required=True, help='Pretrained Model Weight Path for Inference')
parser.add_argument('--GIFPath',required=True, help='GIF_Image_path')
parser.add_argument('--OutputPath', required=True, help='OutputFolder')

def main():
    args = parser.parse_args()
    
    model = InceptionResnetV1()
    
    for param in model.parameters(): 
        param.requires_grad = True
    model.last_linear1 = nn.Linear(512, 120, bias=True)        
    
    model.load_state_dict(torch.load(args.ModelWeightPath))
    model.cuda()

    

    im = Image.open(args.GIFPath) # GIF 경로 
    frame_list = []
    index=0
    for frame in ImageSequence.Iterator(im):
        frame.save("frame%d.png" % index)
        index+=1
    frame_file_list = [i for i in os.listdir() if 'png' in i]
    frame_list = [Image.open(i) for i in frame_file_list]

    
    os.makedirs(args.OutputPath, exist_ok=True)
    make_gif(model, frame_list,120,300,500,600, args.OutputPath)
    generate_gif(frame_list, args.OutputPath)
    
    [os.remove(i) for i in os.listdir() if 'png' in i]


if __name__ == "__main__":
    main()