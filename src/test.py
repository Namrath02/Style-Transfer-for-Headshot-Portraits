import os
import pickle

import cv2
import numpy as np
from skimage.io import imread, imsave, imshow , show
from face_alignment import FaceAlignment, LandmarksType
from image_morpher import ImageMorpher
import matplotlib.pyplot as plt

def laplacian_pyr(style, input, new_style, new_img, new_h, new_w):
    h,w,c = input.shape
    laplace_style = []
    laplace_input = []

    new_h, new_w = int(new_h / 2), int(new_w / 2)
    new_style = cv2.pyrDown(new_style, np.zeros((new_h, new_w, c)))
    new_img = cv2.pyrDown(new_img, np.zeros((new_h, new_w, c)))

    laplace_style.append(style - cv2.resize(new_style, (w, h)))
    laplace_input.append(input - cv2.resize(new_img, (w, h)))

    pre_style = new_style
    pre_input = new_img

    for i in range(6):
        new_h, new_w = int(new_h / 2), int(new_w / 2)
        new_style = cv2.pyrDown(new_style, np.zeros((new_h, new_w, c)))
        new_img = cv2.pyrDown(new_img, np.zeros((new_h, new_w, c)))
        
        
        temp_style = cv2.resize(pre_style, (w, h)) - cv2.resize(new_style, (w, h))
        temp_input = cv2.resize(pre_input, (w, h)) - cv2.resize(new_img, (w, h))
        laplace_style.append(temp_style)
        laplace_input.append(temp_input)

        pre_style = new_style
        pre_input = new_img
    
    return new_style , new_img, laplace_style, laplace_input, new_h, new_w

def local_energy(laplace_style, laplace_img, new_h, new_w, w, h, c):
    energy_style = []
    energy_input = []
    
    for i in range(7):
        new_style_ener = cv2.pyrDown(laplace_style[i] ** 2, (new_h, new_w, c))
        new_input_ener = cv2.pyrDown(laplace_img[i] ** 2, (new_h, new_w, c))

        for j in range(i - 1):
            new_style_ener = cv2.pyrDown(new_style_ener, (new_h, new_w, c))
            new_input_ener = cv2.pyrDown(new_input_ener, (new_h, new_w, c))

        energy_style.append(cv2.resize(np.sqrt(new_style_ener), (w, h)))
        energy_input.append(cv2.resize(np.sqrt(new_input_ener), (w, h)))

    return energy_style, energy_input

def compute_value(energy_style, energy_img, img_pyr, style_pyr, residue_style, h,w,c):
    eps = 0.01 ** 2
    gain_max = 2.8
    gain_min = 0.005
    output = np.zeros((h, w, c))
    for i in range(7):
        gain = np.sqrt(np.divide(energy_style[i], (energy_img[i] + eps)))
        gain[gain <= gain_min] = 1
        gain[gain > gain_max] = gain_max
        output += np.multiply(img_pyr[i], gain)
    output += residue_style

    return output

def local_matching(target, img, tmask, imask,  vx , vy):
    height , width, _ = img.shape
    h,w,c = img.shape
    new_style = np.copy(target)
    new_img = np.copy(img)
    new_style[tmask == 0] = 0
    new_img[imask == 0] = 0

    # laplacian stack construction
    new_style , new_img , style_pyr , img_pyr, height, width = laplacian_pyr(target, img, new_style, new_img, height, width)
    residue_style = cv2.resize(new_style, (w,h))
    residue_img = cv2.resize(new_img, (w,h))
    

    #energy local 
    energy_style, energy_img = local_energy(style_pyr, img_pyr, height, width,w,h,c)

    #post op warping style stacks
    for i in range(len(energy_style)):
        style_pyr[i] = style_pyr[i][vy, vx]
        energy_style[i] = energy_style[i][vy, vx]
    
    # gain map 
    return compute_value(energy_style, energy_img, img_pyr, style_pyr, residue_style ,h,w,c)


#/////////////

def dense_matching(style, input, style_lm, input_lm):
        im = ImageMorpher()
        morphed_img = im.run(style, input, style_lm, input_lm)

        # Match better with Dense SIFT

        return morphed_img

# /////////////////


def replace_bkg(matched, style, input, style_mask, input_mask, vx, vy):
        
        temp = np.zeros(input.shape, dtype=np.uint8)
        temp[style_mask == 0] = style[style_mask == 0]
        temp[input_mask == 255] = 0
        # xy = (255 - style_mask).astype(np.uint8)
        # bkg = cv.inpaint(temp, xy[:, :, 0], 10, cv.INPAINT_TELEA)
        # imsave('output/bkg.jpg', bkg.astype(int))
        # TODO: Extrapolate background
        xy = np.logical_not(input_mask.astype(bool))
        matched[xy] = 0
        output = temp + matched
        output[output > 255] = 255
        output[output <= 0] = 0
        output = output.astype(int)
        # imsave('output/temp.jpg', output)
        # imsave('output/temp.jpg', style.astype(int))
        
        return matched

# /////////////////
def circular_arc_detection(image):
   
    iris_center = (100, 150)
    iris_radius = 30

    return iris_center, iris_radius

def k_means_segmentation(image):
    iris_segment = image[:, :50]
    highlight_segment = image[:, 50:100]
    pupil_segment = image[:, 100:]

    return iris_segment, highlight_segment, pupil_segment

def alpha_matting(reflection_mask, image):

    refined_mask = reflection_mask

    return refined_mask

def detect_highlights(iris_region):
    threshold = 60
    highlight_mask = (iris_region > threshold).astype(np.uint8) * 255

    return highlight_mask

def inpaint(input_img, highlight_mask):
    inpainted_img = input_img.copy()
    common_region = min(input_img.shape[0], highlight_mask.shape[0])
    for i in range(common_region):
        for j in range(input_img.shape[1]):
            if highlight_mask[i, j].any():  
                inpainted_img[i, j] = inpainted_img[i - 1, j]

    return inpainted_img

def compose_highlights(example_img, inpainted_img):
    
    output_img = 0.5 * inpainted_img + 0.5 * example_img

    return output_img.astype(np.uint8)


def eye_highlight(matched, style, input, style_mask, input_mask, vx, vy):
    input_image = matched
    example_image = style

    iris_center, iris_radius = circular_arc_detection(example_image)
    iris_segment, highlight_segment, _ = k_means_segmentation(example_image)

    reflection_mask = alpha_matting(iris_segment, example_image)
    highlight_mask = detect_highlights(iris_segment)

    inpainted_img = inpaint(input_image, highlight_mask)
    result_image = compose_highlights(highlight_segment, inpainted_img)
    return matched

# ////////////


def setup(style_img, input_img, style_mask, input_mask, save=True):
        style_name = os.path.basename(style_img).split('.')[0]
        input_name = os.path.basename(input_img).split('.')[0]

        style_img = np.float32(cv2.imread(style_img))
        input_img = np.float32(cv2.imread(input_img))

        # style_img = rgb2gray(style_img)
        # input_img = rgb2gray(input_img)

        # style_img = style_img[:, :, np.newaxis]
        # input_img = input_img[:, :, np.newaxis]
        
        
        
        style_mask = np.float32(cv2.imread(style_mask))
        input_mask = np.float32(cv2.imread(input_mask))
        

        # style_mask = rgb2gray(style_mask)
        # input_mask = rgb2gray(input_mask)
        # style_mask = style_mask[:, :, np.newaxis]
        # input_mask = input_mask[:, :, np.newaxis]

        # Fetch Facial Landmarks
        if os.path.exists('input/%s_%s_lm.pkl' % (style_name, input_name)):
            with open('input/%s_%s_lm.pkl' % (style_name, input_name), 'rb') as f:
                pkl = pickle.load(f)
                style_lm = pkl['style']
                input_lm = pkl['input']
        else:
            fa = FaceAlignment(LandmarksType._2D, device='cpu', flip_input=False)
            style_lm = fa.get_landmarks(style_img)[0]
            input_lm = fa.get_landmarks(input_img)[0]
            with open('input/%s_%s_lm.pkl' % (style_name, input_name),
                      'wb') as f:
                pickle.dump({
                    'style': style_lm,
                    'input': input_lm
                }, f, protocol=2)

        output_filename = '_'.join({input_name, style_name})
        save = save

        warped, vx, vy = dense_matching(style_img, input_img, style_lm, input_lm)
        
        matched = local_matching(style_img, input_img, style_mask, input_mask, vx, vy)
        matched = replace_bkg(matched, style_img, input_img, style_mask, input_mask, vx, vy)
        # matched = eye_highlight(matched, style_img, input_img, style_mask, input_mask, vx, vy)
        extra_bg = []
        
        for i in range(input_img.shape[0]):
            temp = []
            for j in range(input_img.shape[1]):
                temp2 = []
                for k in range(input_img.shape[2]):
                   if (input_mask[i][j][k] != 255):
                        temp2.append(input_img[i][j][k])
                   else:
                        temp2.append(0)
                temp.append(temp2)
            extra_bg.append(temp)
                            
        matched += extra_bg
        cv2.imwrite('output.jpg', matched)
        

setup('../images/style1.jpg', '../images/face1.jpg', '../images/masks/mask_style1.jpg', '../images/masks/mask_custom1.jpg')