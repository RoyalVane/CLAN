import torch
import numpy as np
from PIL import Image

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def tensor2im(image_tensor, imtype=np.uint8):
    #print(image_tensor)
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
#    mean = np.array([0.485, 0.456, 0.406])
#    std = np.array([0.229, 0.224, 0.225])
#    image_numpy = (std * image_numpy + mean) * 255
    image_numpy = image_numpy + IMG_MEAN
    return image_numpy.astype(imtype)

def onedim_tensor2im(image_tensor, imtype=np.uint8, dataset = 'Pascal'):
    if dataset == 'Pascal':
        palette_idx = np.array([[0, 0, 0], [255, 0, 0], [155, 100, 0], [128, 128, 0], [0, 128, 128], [0, 100, 155], [0, 0, 255]])#Pascal
    result = np.zeros(shape = (image_tensor.size(2), image_tensor.size(3), 3))
    #image_numpy = image_tensor[0].cpu().float().numpy()
    for i in range(image_tensor.size(2)):
        for j in range(image_tensor.size(3)):
            #result[i][j] = palette_idx[np.argmax(image_numpy[:,i,j]) + 1]
            if image_tensor.data[0][0][i][j] > 0.8:
                result[i][j] = palette_idx[1]
            elif 0.65 < image_tensor.data[0][0][i][j] < 0.8:
                result[i][j] = palette_idx[2]
            elif 0.5 < image_tensor.data[0][0][i][j] < 0.65:
                result[i][j] = palette_idx[3]
            elif 0.35 < image_tensor.data[0][0][i][j] < 0.5:
                result[i][j] = palette_idx[4]
            elif 0.2 < image_tensor.data[0][0][i][j] < 0.35:
                result[i][j] = palette_idx[5]
            else:
                result[i][j] = palette_idx[6]
    return result.astype(imtype)

def onedim_superpixel2im(image_tensor, imtype=np.uint8, dataset = 'Pascal'):
    if dataset == 'Pascal':
        palette_idx = np.array([[0, 0, 0], [255, 0, 0], [155, 100, 0], [128, 128, 0], [0, 128, 128], [0, 100, 155], [0, 0, 255]])#Pascal
    result = np.zeros(shape = (image_tensor.size(1), image_tensor.size(2), 3))
    image_tensor = torch.div(image_tensor, 100)
    #image_numpy = image_tensor[0].cpu().float().numpy()
    for i in range(image_tensor.size(1)):
        for j in range(image_tensor.size(2)):
            #result[i][j] = palette_idx[np.argmax(image_numpy[:,i,j]) + 1]
            if image_tensor[0][i][j] > 0.8:
                result[i][j] = palette_idx[1]
            elif 0.65 < image_tensor[0][i][j] < 0.8:
                result[i][j] = palette_idx[2]
            elif 0.5 < image_tensor[0][i][j] < 0.65:
                result[i][j] = palette_idx[3]
            elif 0.35 < image_tensor[0][i][j] < 0.5:
                result[i][j] = palette_idx[4]
            elif 0.2 < image_tensor[0][i][j] < 0.35:
                result[i][j] = palette_idx[5]
            else:
                result[i][j] = palette_idx[6]
    return result.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
    
