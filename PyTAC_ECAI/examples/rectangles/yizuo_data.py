import numpy as np
from random import sample
from pprint import pprint
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import copy
import train.data as data
from sklearn.model_selection import train_test_split
'''
 Call Generate(tp, size, tot_clean, num_noisy, num_rec,
              num_noise, area_ratio, num_remove, is_Visualize) 
        to generate images and labels

 Input:
      tp: type of label - "label", "width", "height", "row", "col"
      size: size of the Square map. Ex: 10 - 10x10
      tot_clean: Number of Clean images
      num_noisy: number of noisy images associated with each clean image
      num_rec: maximal number of noisy rectangles on each noisy image
      num_noise: maximal number of noises(1pixel) on each noisy image
      area_ratio: area of largest noisy rectangle / area of the main rectangle 
      num_remove: maximal number of pixels removed from the main rectangle
      is_Visualize: whether to print out all the images

Output:
      images: list of (size,size) images
      labels: list of labels, index match with images 
'''

class Data_Gen:

    def __init__(self):
        self.Clear();
    
    def Clear(self):
        self.m_clean_data = []
        self.m_clean_label = []
        self.m_noisy_data = []
        self.m_noisy_label = []

    def Query_Label(self, lx, ly, width, height):
        if self.m_type == 'label':
            self.flat_size = 2
            return float((width > height))
        if self.m_type == 'height':
            self.flat_size = self.m_size
            return float(height)
        if self.m_type == 'width':
            self.flat_size = self.m_size
            return float(width)
        if self.m_type == 'row':
            self.flat_size = self.m_size
            return float(lx)
        if self.m_type == 'col':
            self.flat_size = self.m_size
            return float(ly)
        
    def Generate_Clean(self, lx, ly, width, height):

        img = np.array([[0.] * self.m_size] * self.m_size)
        self.area = width * height

        # Make Sure the rectangles are valid

        #print("%d, %d, %d, %d" %(lx, ly, width, height))
        #mark on pixels
        for i in range(lx, lx + height):
            for j in range(ly, ly + width):
                img[i][j] = 1.
                
        lab = self.Query_Label(lx, ly, width, height)
        self.m_clean_data.append(img)
        self.m_clean_label.append(lab)
        
        img = copy.deepcopy(img)
        #mark grey area
        for i in range(max(0, lx - 1), min(lx + height + 1, self.m_size)):
            for j in range(max(0, ly - 1), min(ly + width + 1, self.m_size)):
                if img[i][j] != 1.:
                    img[i][j] = 0.5

        return img, lab

    def Search_Right(self, img, x, y, dx):
        for pos in range(y, self.m_size):
            for j in range(x, dx):
                if img[j][pos] != 0.00:
                    return (pos - y)
        return (self.m_size - y)
                

    def Search_Down(self, img, x, y):
        pos = 0
        while (x + pos < self.m_size and img[x + pos][y] == 0.00):
            pos += 1
        return pos
            
    def Get_White_Pixels(self, img):
        white_pixels = []
        for i in range(self.m_size):
            for j in range(self.m_size):
                if img[i][j] == 0.00:
                    white_pixels.append(i * self.m_size + j)
        return white_pixels

    def Add_Noise_Rectangles(self, num_rec, white_pixels, img, allowed_area, noise_type, rec_shape):

        for id in range(num_rec):
            if (len(white_pixels) == 0):
                break
            # Randomnize a white pixel to be the upper-left point of rectangle
            pt = white_pixels[np.random.randint(0, len(white_pixels))]
            x = int(pt / self.m_size)
            y = int(pt % self.m_size)
            # determine a random width and height of the rectangle
            # if fixed, only need to check whether it's valid
            if noise_type == "None":
                down_most = self.Search_Down(img, x, y)
                lenx = np.random.randint(1, min(down_most, allowed_area) + 1)
                right_most = self.Search_Right(img, x, y, x + lenx)
                leny = np.random.randint(1, min(right_most, int(allowed_area / lenx)) + 1)
                # mark the on pixels
                for i in range(x, x + lenx):
                    for j in range(y, y + leny):
                        img[i][j] = 1.00

            elif noise_type == "Rectangle":
                flag = True
                for i in range(x, x + rec_shape[0]):
                    for j in range(y, y + rec_shape[1]):
                        if i >= self.m_size or j >= self.m_size or img[i][j] != 0.00:
                            flag = False

                if rec_shape[0] * rec_shape[1] <= allowed_area and flag == True:
                    lenx = rec_shape[0]
                    leny = rec_shape[1]
                    # mark the on pixels
                    for i in range(x, x + lenx):
                        for j in range(y, y + leny):
                            img[i][j] = 1.00
                else:
                    continue

            elif noise_type == "Triangle":
                flag = True
                for i in range(x, x + rec_shape[0]):
                    if i >= self.m_size or img[i][y] != 0.00:
                        flag = False
                if flag == False:
                    continue
                loc_x = np.random.randint(0, 2)
                loc_y = int(np.sign(np.random.randint(0, 2) - 1e-3))
                if y + loc_y < 0 or y + loc_y >= self.m_size or img[x + loc_x][y + loc_y] != 0.00:
                    flag = False
                if 3 <= allowed_area and flag == True:
                    lenx = 2
                    leny = 2
                    img[x + loc_x][y + loc_y] = 1.00
                    img[x][y] = 1.00
                    img[x + 1][y] = 1.00
                    if loc_y == -1:
                        y += loc_y
                else:
                    continue

            #mark grey area
            for i in range(max(0, x - 1), min(x + lenx + 1, self.m_size)):
                for j in range(max(0,y - 1), min(y + leny + 1, self.m_size)):
                    if img[i][j] != 1.:
                        img[i][j] = 0.5
                    index = np.argwhere(white_pixels == i * self.m_size + j)
                    white_pixels = np.delete(white_pixels, index)
            
        return white_pixels, img
                        
                        
    def Add_Noise(self, num_noise, white_pixels, img):
        if (num_noise != 0):
            mark_pixels = np.random.choice(white_pixels, size = (int(num_noise)))
            for pix in mark_pixels:
                img[int(pix / self.m_size)][int(pix % self.m_size)] = 1.00
        return img
    
    def Generate_Noisy_Rec_and_Noise(self, img, label, num_rec, num_noise, main_area, allowed_area, noise_type, rec_shape):
        # Find all the white pixels
        white_pixels = np.array(self.Get_White_Pixels(img))
        
        if allowed_area > 0:
            white_pixels, img = self.Add_Noise_Rectangles(num_rec, white_pixels, img, allowed_area, noise_type, rec_shape)
                            
        # Add more noises
        num_noise = min(num_noise, len(white_pixels) / 2)
        num_noise = min(num_noise, main_area - 1)
        img = self.Add_Noise(num_noise, white_pixels, img)
        
        # Turn all the grey pixels to white
        for i in range(self.m_size):
            for j in range(self.m_size):
                if img[i][j] == 0.5:
                    img[i][j] = 0.00

        return img, label
            
    def Remove_Pixels_From_Main_Rec(self, img, num_removal):
        list_black = []
        for i in range(len(img)):
            for j in range(len(img[0])):
                if img[i][j] == 1:
                    list_black.append(i * self.m_size + j)

        if num_removal == 0:
            return img, num_removal
        
        num_removal = np.random.randint(0, num_removal + 1)
        mark_pixels = np.random.choice(list_black, size = num_removal)
        # Mark the removed pixels as grey. It will turn into white during the adding noise procedure
        for pix in mark_pixels:
            img[int(pix / self.m_size)][int(pix % self.m_size)] = 0.5
            
        return img, num_removal
        
    def Generate(self, tp, size, num_noisy_per_clean, num_rec, 
                 num_noise, area_ratio, num_remove, num_clean_img, 
                 num_noisy_img, noise_type="None", rec_shape = (0,0), is_Visualize=False, out_use = "CNN"):
        self.Clear();
        self.m_type = tp
        self.m_size = size

        id = 0
        # Generate clean image
        for lx in range(0, self.m_size):
            for ly in range(0, self.m_size):
                for height in range(1, self.m_size - lx + 1):
                    for width in range(1, self.m_size - ly + 1):
                        if width != height:
                            id += 1
                            img, label = self.Generate_Clean(lx, ly, width, height)
                            if is_Visualize == True:
                                print("======== Clean Data #%d =======" %(id))
                                self.Visualize(self.m_clean_data[-1], self.m_clean_label[-1])
                                # Generate noisy images with rectangles
                            for j in range(num_noisy_per_clean):
                                new_img = copy.deepcopy(img)
                                num_removal = min([num_remove, width - 1, height - 1]) # num_removal < width, height - 1
                                new_img, removed_pixel = self.Remove_Pixels_From_Main_Rec(new_img, num_removal)
                                main_area = self.area - removed_pixel

                                if noise_type == "None":
                                    new_img, label = self.Generate_Noisy_Rec_and_Noise(new_img, label, num_rec, num_noise, main_area,
                                                                                   int(main_area * area_ratio), noise_type, (-1, -1))
                                elif noise_type == "Rectangle":
                                    new_img, label = self.Generate_Noisy_Rec_and_Noise(new_img, label, num_rec, num_noise, main_area,
                                                                                       int(main_area * area_ratio), noise_type, rec_shape)
                                elif noise_type == "Triangle":
                                    new_img, label = self.Generate_Noisy_Rec_and_Noise(new_img, label, num_rec, num_noise, main_area,
                                                                                   int(main_area * area_ratio), noise_type, (2, 1))

                                self.m_noisy_data.append(new_img)
                                self.m_noisy_label.append(label)
                                    
                                if is_Visualize == True:
                                    print("====== Noisy Data %d for Clean Data #%d ====" %(j + 1, id))
                                    self.Visualize(self.m_noisy_data[-1], self.m_noisy_label[-1])
        
        self.m_clean_data = np.array(self.m_clean_data)
        self.m_noisy_data = np.array(self.m_noisy_data)

        if out_use == "CNN":
            self.Reformat_CNN()
        elif out_use == "BN":
            self.Reformat_BN()
            
        self.Convert_One_Hot()

        # sample the clean data and noisy data
        if num_clean_img != -1:
            self.m_clean_data, _, self.m_clean_label, __ = train_test_split(self.m_clean_data, self.m_clean_label, train_size = num_clean_img)
        if num_noisy_img != -1:
            self.m_noisy_data, _, self.m_noisy_label, __ = train_test_split(self.m_noisy_data, self.m_noisy_label, train_size = num_noisy_img)
        
        return data.evd_row2col(np.concatenate((self.m_clean_data, self.m_noisy_data))), np.concatenate((self.m_clean_label, self.m_noisy_label))

    def Reformat_CNN(self):
        self.m_clean_data = np.reshape(self.m_clean_data, [len(self.m_clean_data), self.m_size, self.m_size, 1])
        
        self.m_noisy_data = np.reshape(self.m_noisy_data, [len(self.m_noisy_data), self.m_size, self.m_size, 1])

    def Reformat_BN(self):
        self.m_clean_data = np.reshape(self.m_clean_data, [-1, self.m_size ** 2]).astype(int)
        self.m_clean_data = np.eye(2)[self.m_clean_data].astype(float)
                
        self.m_noisy_data = np.reshape(self.m_noisy_data, [-1, self.m_size ** 2]).astype(int)
        self.m_noisy_data = np.eye(2)[self.m_noisy_data].astype(float)
        
        
    def Convert_One_Hot(self):
        if self.m_type == 'label' or self.m_type == 'row' or self.m_type == 'col':
            self.m_clean_label = [int(x) for x in self.m_clean_label]
            self.m_noisy_label = [int(x) for x in self.m_noisy_label]
        else:
            self.m_clean_label = [int(x - 1) for x in self.m_clean_label]
            self.m_noisy_label = [int(x - 1) for x in self.m_noisy_label]
        self.m_clean_label = np.eye(self.flat_size)[self.m_clean_label]
        self.m_noisy_label = np.eye(self.flat_size)[self.m_noisy_label]
        
    def Visualize(self, data, label):
        print("Label %lf\n" %label)
        pprint(data)


"""
data, label = Data_Gen().Generate(tp = 'label', size = 10, num_noisy_per_clean = 9, num_rec = 10, num_noise = 0, area_ratio = .8,
                                  num_remove = 0, num_clean_img = 10, num_noisy_img = 20, noise_type="Rectangle", rec_shape = (1, 2), is_Visualize = True) 
#print(data)
"""
