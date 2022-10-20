import numpy as np
import random
import train.data as data
import utils.visualize as visualize

# https://en.wikipedia.org/wiki/Seven-segment_display

""" generating labeled data for 7-segment digits """

def get(size, digits=range(10), noisy_image_count=0, noise_count=0):
    height_inc = 7 # thickness = 1 (three horizontal segments, four spaces)
    image       = np.zeros([size,size])
    evidence    = []
    marginals   = []
    count_visualize = {d:0 for d in digits} # for counting how many images are visualized
    for digit in digits:
        marginal = __label2distribution(digit,digits)
        for r in range(0,size-height_inc+1):
            for height in range(7,size-r+1,height_inc):
                assert height <= size and height >= 7 and height % 7 == 0
#                thickness = height // 7   # width of segment
                thickness = 1
                length    = 4 * thickness # length of segment
#                assert height == 2*(length-thickness)+thickness
                for c in range(0,size-length+1):                                                                                                        
                    __add_digit(digit,image,r,c,length,thickness)
                    #visualize.image(image,str(digit))
                    # data for perfect image
                    lambdas = __image2lambdas(image)
                    evidence.append(lambdas)
                    marginals.append(marginal)
                    # data for noisy images
                    for _ in range(noisy_image_count):
                        noisy_image = np.array(image)
                        __add_noise(noisy_image,noise_count)
                        if False: # visualize
                            if count_visualize[digit] <= 10 and random.randint(0,100) <= 1:
                                visualize.image(noisy_image,str(digit))
                                count_visualize[digit] += 1
                        lambdas = __image2lambdas(noisy_image)
                        evidence.append(lambdas)
                        marginals.append(marginal)
                    image.fill(0) # clear image
    evidence  = data.evd_row2col(evidence)
    marginals = np.array(marginals) 
    return (evidence, marginals)   
        
def __add_noise(noisy_image,noise_count):
    size  = noisy_image.shape[0]
    added = 0
    while added < noise_count:
        i = random.randint(0,size-1)
        j = random.randint(0,size-1)
        if noisy_image[i,j] != 1:
            noisy_image[i,j] = 1
            added += 1
    
def __add_digit(digit,image,rs,cs,length,thickness):
    size   = image.shape[0]
    height = 2*(length-thickness)+thickness
    
    assert rs >= 0 and rs+height <= size
    assert cs >= 0 and cs+length <= size
    
    if   digit == 0: segments = ('A','B','C','D','E','F')
    elif digit == 1: segments = ('B','C')
    elif digit == 2: segments = ('A','B','D','E','G')
    elif digit == 3: segments = ('A','B','C','D','G')
    elif digit == 4: segments = ('B','C','F','G') 
    elif digit == 5: segments = ('A','C','D','F','G')   
    elif digit == 6: segments = ('A','C','D','E','F','G')  
    elif digit == 7: segments = ('A','B','C')
    elif digit == 8: segments = ('A','B','C','D','E','F','G')    
    else:            segments = ('A','B','C','D','F','G')
    
    __add_segments(segments,image,rs,cs,length,thickness)
    
    
def __add_segments(segments,*args):
    if 'A' in segments: __add_A(*args)
    if 'B' in segments: __add_B(*args)
    if 'C' in segments: __add_C(*args)
    if 'D' in segments: __add_D(*args)
    if 'E' in segments: __add_E(*args)
    if 'F' in segments: __add_F(*args)
    if 'G' in segments: __add_G(*args)
  

# horizontals
def __add_A(image,rs,cs,length,thickness):
    __add_horizontal(image,rs,cs,length,thickness)
    
def __add_D(image,rs,cs,length,thickness):
    rs = rs + 2*length- 2*thickness # rs + height - thickness
    __add_horizontal(image,rs,cs,length,thickness)

def __add_G(image,rs,cs,length,thickness):
    rs = rs + length - thickness
    __add_horizontal(image,rs,cs,length,thickness)
    
# verticals
def __add_B(image,rs,cs,length,thickness):
    cs = cs + length - thickness
    __add_vertical(image,rs,cs,length,thickness)
    
def __add_C(image,rs,cs,length,thickness):
    rs = rs + length - thickness
    cs = cs + length - thickness
    __add_vertical(image,rs,cs,length,thickness)
    
def __add_E(image,rs,cs,length,thickness):
    rs = rs + length - thickness
    __add_vertical(image,rs,cs,length,thickness)
    
def __add_F(image,rs,cs,length,thickness):
    __add_vertical(image,rs,cs,length,thickness)


# segments A, D, G        
def __add_horizontal(image,rs,cs,length,height):
    size = image.shape[0]
    assert rs+height <= size
    assert cs+length <= size
    for r in range(rs,rs+height):
        for c in range(cs,cs+length):
            image[r,c] = 1
            
# segments B, C, E, F   
def __add_vertical(image,rs,cs,length,width):
    size = image.shape[0]
    assert rs+length <= size
    assert cs+width  <= size
    for r in range(rs,rs+length):
        for c in range(cs,cs+width):
            image[r,c] = 1
    
    
# constants    
__distribution = [
    np.array([1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]),
    np.array([0.,1.,0.,0.,0.,0.,0.,0.,0.,0.]),
    np.array([0.,0.,1.,0.,0.,0.,0.,0.,0.,0.]),
    np.array([0.,0.,0.,1.,0.,0.,0.,0.,0.,0.]),
    np.array([0.,0.,0.,0.,1.,0.,0.,0.,0.,0.]),
    np.array([0.,0.,0.,0.,0.,1.,0.,0.,0.,0.]),
    np.array([0.,0.,0.,0.,0.,0.,1.,0.,0.,0.]),
    np.array([0.,0.,0.,0.,0.,0.,0.,1.,0.,0.]),
    np.array([0.,0.,0.,0.,0.,0.,0.,0.,1.,0.]),
    np.array([0.,0.,0.,0.,0.,0.,0.,0.,0.,1.])]

__on_pixel  = np.array([1.,0.]) # lambda for on-pixel
__off_pixel = np.array([0.,1.]) # lambda for off-pixel

def __label2distribution(digit,digits):
    return np.take(__distribution[digit],digits)

# must generate lambdas r, then c to match order of evidence nodes
def __image2lambdas(image):
    size    = image.shape[0]
    lambdas = []
    for r in range(size):
        for c in range(size):
            if image[r,c] == 1:
                pixel = __on_pixel
            elif image[r,c] == 0:
                pixel = __off_pixel
            else: # grayscale
                pixel = np.array([image[r,c],1-image[r,c]])
            lambdas.append(pixel)
    return lambdas