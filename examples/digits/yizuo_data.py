import numpy as np
import random

from sklearn.model_selection import train_test_split
from pprint import pprint 
# https://en.wikipedia.org/wiki/Seven-segment_display

"generating labeled data for 7-segment digits"

def get(size, num_clean_images=-1, num_noisy_images=-1, digits=range(10), output_type="BN", noise_type="None",  noisy_image_count=0, noise_count=0):
    height_inc = 7 # thickness = 1 (three horizontal segments, four spaces)
    image       = np.zeros([size,size])
    clean_evidence    = []
    clean_marginals   = []
    noisy_evidence = []
    noisy_marginals = []
    for digit in digits:
        marginal = __label2distribution(digit,digits)
        for r in range(0,size-height_inc+1):
            for height in range(7,size-r+1,height_inc):
                assert height <= size and height >= 7 and height % 7 == 0
                thickness = height // 7   # width of segment
                length    = 4 * thickness # length of segment
                assert height == 2*(length-thickness)+thickness
                for c in range(0,size-length+1):                                                                                                        
                    __add_digit(digit,image,r,c,length,thickness)
                    # data for perfect image
                    lambdas = __image2lambdas(image)
                    clean_evidence.append(lambdas)
                    clean_marginals.append(marginal)
                    # data for noisy images
                    for _ in range(noisy_image_count):
                        noisy_image = np.array(image)
                        __add_noise(noisy_image,noise_type, noise_count)
                        lambdas = __image2lambdas(noisy_image)
                        noisy_evidence.append(lambdas)
                        noisy_marginals.append(marginal)
                    image.fill(0) # clear image
    clean_evidence  = np.array(col_evd(clean_evidence))
    clean_marginals = np.array(clean_marginals).astype(float)
    noisy_evidence  = np.array(col_evd(noisy_evidence))
    noisy_marginals = np.array(noisy_marginals).astype(float)

    clean_evidence = np.swapaxes(clean_evidence, 0, 1)
    noisy_evidence = np.swapaxes(noisy_evidence, 0, 1)

    if num_clean_images != -1 and num_clean_images != 0:
        clean_evidence, _, clean_marginals, __ = train_test_split(clean_evidence, clean_marginals, train_size=num_clean_images)
    if num_noisy_images != -1 and num_noisy_images != 0:
        noisy_evidence, _, noisy_marginals, __ = train_test_split(noisy_evidence, noisy_marginals, train_size = num_noisy_images)
    """
    print(np.array(clean_evidence).shape)
    print(np.array(clean_marginals).shape)
    print(np.array(noisy_evidence).shape)
    print(np.array(noisy_marginals).shape)
    """
    if num_clean_images == 0:
        evidence = noisy_evidence
        marginals = noisy_marginals
    elif num_noisy_images == 0:
        evidence = clean_evidence
        marginals = clean_marginals
    else:
        evidence = np.concatenate((clean_evidence, noisy_evidence), axis=0)
        marginals = np.concatenate((clean_marginals, noisy_marginals), axis=0)

    evidence = np.swapaxes(evidence, 0, 1)

    """
    print(evidence.shape)
    print(marginals.shape)
    """

    if output_type == "BN":
        return (list(evidence), marginals)
    elif output_type == "CNN":
        X = Reformat_For_NN(evidence, size)
        return (X, marginals)   
        
def __add_noise(noisy_image, noise_type, noise_count):
    size  = noisy_image.shape[0]
    added = 0

    max_height = size - 1;
    max_width = size - 1;
    
    if noise_type == "2-1":
        max_height -= 1
    elif noise_type == "1-2":
        max_width -= 1
    
    while added < noise_count:
        i = random.randint(0,max_height)
        j = random.randint(0,max_width)
        if noise_type == "2-1" and noisy_image[i,j] != 1 and noisy_image[i+1,j] != 1:
            noisy_image[i, j] = 1
            noisy_image[i+1, j] = 1
            added += 1
        elif noise_type == "1-2" and noisy_image[i,j] != 1 and noisy_image[i,j+1] != 1:
            noisy_image[i, j] = 1
            noisy_image[i, j + 1] = 1
            added += 1
        elif noise_type == "None" and noisy_image[i,j] != 1:
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
            pixel = __on_pixel if image[r,c] == 1 else __off_pixel
            lambdas.append(pixel)
    return lambdas 

# converts row-based evidence to col-based evidence
def col_evd(rows):
    batch_size = len(rows)
    assert batch_size > 0   # at least one batch
    var_count  = len(rows[0])
    assert var_count > 0   # at least one evidence variables
    cols = [[] for _ in range(var_count)]
    for row in rows:
        for i,lambda_ in enumerate(row):
            cols[i].append(lambda_)
    return [np.array(col) for col in cols]

def Reformat_For_NN(X, size):
    img = []
    for i in X:
        l = []
        for j in i:
            t = np.argmax(j)
            l.append(t)
        img.append(l)
    img = np.swapaxes(np.array(img).astype(float), 0, 1)
    img = np.reshape(img, [len(img), size, size, 1])
    return img

"""
X, Y = get(size = 10, num_clean_images=10, num_noisy_images=10, noise_type="None",  output_type = "CNN", noisy_image_count=10, noise_count=3)
X=np.reshape(X, [-1, 10, 10])
for img in X:
    pprint(img)
print(np.array(Y))
"""
