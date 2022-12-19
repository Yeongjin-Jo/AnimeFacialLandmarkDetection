import os
from PIL import Image
import matplotlib.pyplot as plt
from IPython.display import Image as Img
import torchvision.transforms.functional as TF

def visualize_image_(frame_list, image, landmarks,i,x,y,dh,dw, outputPath):
    plt.figure(figsize = (5, 5))
    image = (image - image.min())/(image.max() - image.min())

    landmarks = landmarks.view(-1, 2)
    landmarks = landmarks + 0.5
    landmarks[:,0] = landmarks[:,0]*(dh/128) + x
    landmarks[:,1] = landmarks[:,1]*(dw/128) + y
    tt = frame_list[i]
    plt.imshow(tt)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s = 25, c = 'dodgerblue')
    plt.axis('off')
    plt.savefig(f'./{outputPath}/{i}.png')

def make_gif(model, frame_list, x,y,h,w, outputPath):
    for i in range(len(frame_list)):
        img = frame_list[i]
        org_size = img.size
        img = img.crop((x,y, h,w))
        img = img.resize((128, 128))
        img = TF.to_grayscale(img)
        image = img
        image = TF.to_tensor(image)

        image = (image - image.min())/(image.max() - image.min())
        image = (2 * image) - 1
        image = image.reshape(1,1,128,128)
        model.eval()
        land = model(image.cuda())
        dh = h -x 
        dw = w - y
        visualize_image_(frame_list, image.reshape(1,128,128), land.detach().cpu(),i,x,y,dh,dw, outputPath)

def generate_gif(frame_list, path):
    img_list = os.listdir(path)
    img_list = [f'{path}/{i}' for i in range(len(frame_list))]
    images = [Image.open(f'{path}/{i}.png') for i in range(len(frame_list))]
    im = images[0]
    im.save('out.gif', save_all=True, append_images=images[1:],loop=0xff, duration=50)
    return Img(url='out.gif')
        