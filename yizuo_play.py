import examples.digits.play as dplay
import examples.networks.play as nplay
import examples.rectangles.play as rplay

import examples.digits.model as dmodel
import examples.rectangles.model as rmodel

from examples.rectangles.yizuo_data import *
import examples.digits.yizuo_data as ddata

import tac
import utils.utils as u

import statistics as s
from sklearn.model_selection import train_test_split
import os


def play_Recs_Sizes(use_bk,tie_parameters):

    fd1 = '' if tie_parameters else '_ntie'
    fd2 = '' if use_bk else '_nbk'
    folder = f"./yizuo_results/rec_size{fd1}{fd2}/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    ########################### Parameters ################################
    train_sizes = [25, 50, 100, 250, 500, 1000, 1500, 2000, 4000, 8000]
    #out_tp = ["label", "row", "col", "width", "height"]
    out_tp = ["label", "col", "height"]
    
    clean_train_ratio = 0.2
    clean_test_ratio = 0.2

    num_noisy_per_clean = 10
    num_rec = 5
    num_noise = 5
    area_ratio = .5
    num_remove = 1

    num_test = 10000
    size = 10
    #######################################################################

    for output_type in out_tp:
        model = rBN(size,output_type,use_bk,tie_parameters)
        for num_train in train_sizes:
            filename = "tr_" + output_type + "_" + str(num_train) + ".txt"
            file = open(folder + filename, 'a+')
            results = []
            for ntime in range(5):
            
                ################# Generating Images ###########################
                num_clean_train_img = int(num_train * clean_train_ratio)
                num_noisy_train_img = int(num_train - num_clean_train_img)
                num_clean_test_img = int(num_test * clean_test_ratio)
                num_noisy_test_img = int(num_test - num_clean_test_img)
                trdata, trlabel = Data_Gen().Generate(output_type, 10, num_noisy_per_clean, num_rec, num_noise, area_ratio, num_remove,
                                                    num_clean_train_img, num_noisy_train_img, is_Visualize = False, out_use = 'BN')
                tsdata, tslabel = Data_Gen().Generate(output_type, 10, num_noisy_per_clean, num_rec, num_noise, area_ratio, num_remove, num_clean_test_img, num_noisy_test_img, is_Visualize = False, out_use = 'BN')
                ###############################################################

                # Training the model
                model.Train(t_evidence=trdata, t_labels=trlabel)

                # Testing the model
                acc, mod = model.Test(v_evidence=tsdata, v_labels=tslabel)

    ######################## Record the final accuracy ########
                file.write("Train Size: %d\n" %num_train)
                file.write("Final Accuracy: %lf\n" %acc)
                results.append(acc)
            max_, mean, stdev = max(results), s.mean(results), s.stdev(results)
            file.write(f'\nmax    {max_:.2f}')
            file.write(f'\nmean   {mean:.2f}')
            file.write(f'\nstdev  {stdev:.2f}')
            file.close()

def play_Recs_Sizes(use_bk,tie_parameters):

    fd1 = '' if tie_parameters else '_ntie'
    fd2 = '' if use_bk else '_nbk'
    folder = f"./yizuo_results/rec_size{fd1}{fd2}/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    ########################### Parameters ################################
    train_sizes = [25, 50, 100, 250, 500, 1000, 1500, 2000, 4000, 8000]
    #out_tp = ["label", "row", "col", "width", "height"]
    out_tp = ["label", "col", "height"]
    
    clean_train_ratio = 0.2
    clean_test_ratio = 0.2

    num_noisy_per_clean = 10
    num_rec = 5
    num_noise = 5
    area_ratio = .5
    num_remove = 1

    num_test = 10000
    size = 10
    #######################################################################

    for output_type in out_tp:
        model = rBN(size,output_type,use_bk,tie_parameters)
        for num_train in train_sizes:
            filename = "tr_" + output_type + "_" + str(num_train) + ".txt"
            file = open(folder + filename, 'a+')
            results = []
            for ntime in range(5):
            
                ################# Generating Images ###########################
                num_clean_train_img = int(num_train * clean_train_ratio)
                num_noisy_train_img = int(num_train - num_clean_train_img)
                num_clean_test_img = int(num_test * clean_test_ratio)
                num_noisy_test_img = int(num_test - num_clean_test_img)
                trdata, trlabel = Data_Gen().Generate(output_type, 10, num_noisy_per_clean, num_rec, num_noise, area_ratio, num_remove,
                                                    num_clean_train_img, num_noisy_train_img, is_Visualize = False, out_use = 'BN')
                tsdata, tslabel = Data_Gen().Generate(output_type, 10, num_noisy_per_clean, num_rec, num_noise, area_ratio, num_remove, num_clean_test_img, num_noisy_test_img, is_Visualize = False, out_use = 'BN')
                ###############################################################

                # Training the model
                model.Train(t_evidence=trdata, t_labels=trlabel)

                # Testing the model
                acc, mod = model.Test(v_evidence=tsdata, v_labels=tslabel)

    ######################## Record the final accuracy ########
                file.write("Train Size: %d\n" %num_train)
                file.write("Final Accuracy: %lf\n" %acc)
                results.append(acc)
            max_, mean, stdev = max(results), s.mean(results), s.stdev(results)
            file.write(f'\nmax    {max_:.2f}')
            file.write(f'\nmean   {mean:.2f}')
            file.write(f'\nstdev  {stdev:.2f}')
            file.close()

def play_Recs_Sizes(use_bk,tie_parameters):

    fd1 = '' if tie_parameters else '_ntie'
    fd2 = '' if use_bk else '_nbk'
    folder = f"./yizuo_results/rec_size{fd1}{fd2}/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    ########################### Parameters ################################
    train_sizes = [25, 50, 100, 250, 500, 1000, 1500, 2000, 4000, 8000]
    #out_tp = ["label", "row", "col", "width", "height"]
    out_tp = ["label", "col", "height"]
    
    clean_train_ratio = 0.2
    clean_test_ratio = 0.2

    num_noisy_per_clean = 10
    num_rec = 5
    num_noise = 5
    area_ratio = .5
    num_remove = 1

    num_test = 10000
    size = 10
    #######################################################################

    for output_type in out_tp:
        model = rBN(size,output_type,use_bk,tie_parameters)
        for num_train in train_sizes:
            filename = "tr_" + output_type + "_" + str(num_train) + ".txt"
            file = open(folder + filename, 'a+')
            results = []
            for ntime in range(5):
            
                ################# Generating Images ###########################
                num_clean_train_img = int(num_train * clean_train_ratio)
                num_noisy_train_img = int(num_train - num_clean_train_img)
                num_clean_test_img = int(num_test * clean_test_ratio)
                num_noisy_test_img = int(num_test - num_clean_test_img)
                trdata, trlabel = Data_Gen().Generate(output_type, 10, num_noisy_per_clean, num_rec, num_noise, area_ratio, num_remove,
                                                    num_clean_train_img, num_noisy_train_img, is_Visualize = False, out_use = 'BN')
                tsdata, tslabel = Data_Gen().Generate(output_type, 10, num_noisy_per_clean, num_rec, num_noise, area_ratio, num_remove, num_clean_test_img, num_noisy_test_img, is_Visualize = False, out_use = 'BN')
                ###############################################################

                # Training the model
                model.Train(t_evidence=trdata, t_labels=trlabel)

                # Testing the model
                acc, mod = model.Test(v_evidence=tsdata, v_labels=tslabel)

    ######################## Record the final accuracy ########
                file.write("Train Size: %d\n" %num_train)
                file.write("Final Accuracy: %lf\n" %acc)
                results.append(acc)
            max_, mean, stdev = max(results), s.mean(results), s.stdev(results)
            file.write(f'\nmax    {max_:.2f}')
            file.write(f'\nmean   {mean:.2f}')
            file.write(f'\nstdev  {stdev:.2f}')
            file.close()

def play_Recs_250(use_bk,tie_parameters):

    fd1 = '' if tie_parameters else '_ntie'
    fd2 = '' if use_bk else '_nbk'
    folder = f"./yizuo_results/rec_size{fd1}{fd2}/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    ########################### Parameters ################################
    #train_sizes = [25, 50, 100, 250, 500, 1000, 1500, 2000, 4000, 8000]
    train_sizes = 250
    #out_tp = ["label", "row", "col", "width", "height"]
    out_tp = ["label", "col", "height"]
    
    clean_train_ratio = 0.2
    clean_test_ratio = 0.2

    num_noisy_per_clean = 10
    num_rec = 5
    num_noise = 5
    area_ratio = .5
    num_remove = 1

    num_test = 10000
    size = 10
    #######################################################################

    for output_type in out_tp:
        model = rBN(size,output_type,use_bk,tie_parameters)
        for num_train in train_sizes:
            filename = "tr_" + output_type + "_" + str(num_train) + ".txt"
            file = open(folder + filename, 'a+')
            results = []
            for ntime in range(5):
            
                ################# Generating Images ###########################
                num_clean_train_img = int(num_train * clean_train_ratio)
                num_noisy_train_img = int(num_train - num_clean_train_img)
                num_clean_test_img = int(num_test * clean_test_ratio)
                num_noisy_test_img = int(num_test - num_clean_test_img)
                trdata, trlabel = Data_Gen().Generate(output_type, 10, num_noisy_per_clean, num_rec, num_noise, area_ratio, num_remove,
                                                    num_clean_train_img, num_noisy_train_img, is_Visualize = False, out_use = 'BN')
                tsdata, tslabel = Data_Gen().Generate(output_type, 10, num_noisy_per_clean, num_rec, num_noise, area_ratio, num_remove, num_clean_test_img, num_noisy_test_img, is_Visualize = False, out_use = 'BN')
                ###############################################################

                # Training the model
                model.Train(t_evidence=trdata, t_labels=trlabel)

                # Testing the model
                acc, mod = model.Test(v_evidence=tsdata, v_labels=tslabel)

    ######################## Record the final accuracy ########
                file.write("Train Size: %d\n" %num_train)
                file.write("Final Accuracy: %lf\n" %acc)
                results.append(acc)
            max_, mean, stdev = max(results), s.mean(results), s.stdev(results)
            file.write(f'\nmax    {max_:.2f}')
            file.write(f'\nmean   {mean:.2f}')
            file.write(f'\nstdev  {stdev:.2f}')
            file.close()


def play_Recs_Per(use_bk,tie_parameters):

    fd1 = '' if tie_parameters else '_ntie'
    fd2 = '' if use_bk else '_nbk'
    folder = f"./yizuo_results/rec_per{fd1}{fd2}/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    ########################### Parameters ################################
    #out_tp = ["label", "row", "col", "width", "height"]
    out_tp = ["label", "col", "height"]
    
    clean_test_ratio = 0.2

    num_noisy_per_clean = 10
    num_rec = 5
    num_noise = 5
    area_ratio = .5
    num_remove = 1

    num_train = 2000
    num_test = 10000
    size = 10
    #######################################################################

    for output_type in out_tp:
        model = rBN(size,output_type,use_bk,tie_parameters)
        for per in range(1, 10):
            clean_train_ratio = per / 10.0 + 1e-9
            filename = "tr_" + output_type + "_" + str(per) + ".txt"
            file = open(folder + filename, 'a+')
            results = []
            for ntime in range(5):
            
                ################# Generating Images ###########################
                num_clean_train_img = int(num_train * clean_train_ratio)
                num_noisy_train_img = int(num_train - num_clean_train_img)
                num_clean_test_img = int(num_test * clean_test_ratio)
                num_noisy_test_img = int(num_test - num_clean_test_img)
                trdata, trlabel = Data_Gen().Generate(output_type, 10, num_noisy_per_clean, num_rec, num_noise, area_ratio, num_remove,
                                                    num_clean_train_img, num_noisy_train_img, is_Visualize = False, out_use = 'BN')
                tsdata, tslabel = Data_Gen().Generate(output_type, 10, num_noisy_per_clean, num_rec, num_noise, area_ratio, num_remove, num_clean_test_img, num_noisy_test_img, is_Visualize = False, out_use = 'BN')
                ###############################################################

                # Training the model
                model.Train(t_evidence=trdata, t_labels=trlabel)

                # Testing the model
                acc, mod = model.Test(v_evidence=tsdata, v_labels=tslabel)
                
                    ######################## Record the final accuracy ########
                file.write("Train Size: %d\n" %num_train)
                file.write("Final Accuracy: %lf\n" %acc)
                results.append(acc)
            max_, mean, stdev = max(results), s.mean(results), s.stdev(results)
            file.write(f'\nmax    {max_:.2f}')
            file.write(f'\nmean   {mean:.2f}')
            file.write(f'\nstdev  {stdev:.2f}')
            file.close()

def play_Recs_Diff(use_bk,tie_parameters,train_shape=(2,1),test_shape=(1,2)):

    fd1 = '' if tie_parameters else '_ntie'
    fd2 = '' if use_bk else '_nbk'
    folder = f"./yizuo_results/rec_diff_noise2_{train_shape}_{test_shape}{fd1}{fd2}/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    ########################### Parameters ################################
    #out_tp = ["label", "row", "col", "width", "height"]
    out_tp = ["label", "col", "height"]
    
    clean_train_ratio = 0.2
    clean_test_ratio = 0.2

    num_train = 2000
    num_test = 10000
    size = 10
    #######################################################################

    for output_type in out_tp:
        model = rBN(size,output_type,use_bk,tie_parameters)
        filename = "tr_" + output_type + ".txt"
        file = open(folder + filename, 'a+')
        results = []
        for ntime in range(5):
            
                ################# Generating Images ###########################
            num_clean_train_img = int(num_train * clean_train_ratio)
            num_noisy_train_img = int(num_train - num_clean_train_img)
            num_clean_test_img = int(num_test * clean_test_ratio)
            num_noisy_test_img = int(num_test - num_clean_test_img)
            trdata, trlabel = Data_Gen().Generate(tp=output_type, size=10, num_noisy_per_clean=10, num_rec=6,
                                                  num_noise=0, area_ratio=.8, num_remove=0,
                                                  num_clean_img=num_clean_train_img, num_noisy_img=num_noisy_train_img,
                                                  noise_type="Rectangle", rec_shape = train_shape, is_Visualize = False, out_use = 'BN')
            tsdata, tslabel = Data_Gen().Generate(tp=output_type, size=10, num_noisy_per_clean=4, num_rec=6,
                                                  num_noise=0, area_ratio=.8, num_remove=0,
                                                  num_clean_img=num_clean_test_img, num_noisy_img=num_noisy_test_img,
                                                  noise_type="Rectangle", rec_shape = test_shape, is_Visualize = False, out_use = 'BN')
                ###############################################################

            # Training the model
            model.Train(t_evidence=trdata, t_labels=trlabel)

            # Testing the model
            acc, mod = model.Test(v_evidence=tsdata, v_labels=tslabel)
                
            ######################## Record the final accuracy ########
            file.write("Train Size: %d\n" %num_train)
            file.write("Final Accuracy: %lf\n" %acc)
            results.append(acc)
        max_, mean, stdev = max(results), s.mean(results), s.stdev(results)
        file.write(f'\nmax    {max_:.2f}')
        file.write(f'\nmean   {mean:.2f}')
        file.write(f'\nstdev  {stdev:.2f}')
        file.close()


def Get_Noise(difficulty_level):
    if difficulty_level == "Null":    
        test_num_rec = 0
        test_num_noise = 0
        test_area_ratio = 0
        test_num_remove = 0

    if difficulty_level == "Ignorable":
        test_num_rec = 2
        test_num_noise = 2
        test_area_ratio = 0.5
        test_num_remove = 0
        
    if difficulty_level == "Easy":
        test_num_rec = 2
        test_num_noise = 5
        test_area_ratio = .5
        test_num_remove = 1

    if difficulty_level == "Medium":
        test_num_rec = 5
        test_num_noise = 5
        test_area_ratio = .5
        test_num_remove = 1
    
    if difficulty_level == "Moderate":
        test_num_rec = 7
        test_num_noise = 5
        test_area_ratio = .5
        test_num_remove = 1

    if difficulty_level == "Hard":
        test_num_rec = 7
        test_num_noise = 7
        test_area_ratio = 0.5
        test_num_remove = 2

    if difficulty_level == "SuperHard":
        test_num_rec = 10
        test_num_noise = 10
        test_area_ratio = 0.5
        test_num_remove = 2

    return [test_num_rec, test_num_noise, test_area_ratio, test_num_remove]


def play_Recs_Noise(use_bk,tie_parameters):

    fd1 = '' if tie_parameters else '_ntie'
    fd2 = '' if use_bk else '_nbk'
    folder = f"./yizuo_results/rec_noise{fd1}{fd2}/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    ########################### Parameters ################################
    #out_tp = ["label", "row", "col", "width", "height"]
    out_tp = ["label", "col", "height"]
    
    clean_train_ratio = 0.2
    clean_test_ratio = 0.2

    num_noisy_per_clean = 10

    train_num_rec, train_num_noise, train_area_ratio, train_num_remove = Get_Noise("Medium")
    num_train = 2000
    num_test = 10000

    num_clean_train_img = int(num_train * clean_train_ratio)
    num_noisy_train_img = int(num_train - num_clean_train_img)
    num_clean_test_img = int(num_test * clean_test_ratio)
    num_noisy_test_img = int(num_test - num_clean_test_img)

    size = 10
    #######################################################################

    noise_levels = ["Null", "Ignorable", "Easy", "Medium", "Moderate", "Hard", "SuperHard"]

    for output_type in out_tp:
        model = rBN(size,output_type,use_bk,tie_parameters)
        results = {}
        for lv in noise_levels:
            results[lv] = []
        for ntimes in range(5):
            trdata, trlabel = Data_Gen().Generate(output_type, 10, num_noisy_per_clean, train_num_rec, train_num_noise, train_area_ratio, train_num_remove,
                                                    num_clean_train_img, num_noisy_train_img, is_Visualize = False, out_use = 'BN')
            # Training the model
            model.Train(t_evidence=trdata, t_labels=trlabel)

            for lv in noise_levels:

                # Generate Testing Noises
                test_num_rec, test_num_noise, test_area_ratio, test_num_remove = Get_Noise(lv)
                tsdata, tslabel = Data_Gen().Generate(output_type, 10, num_noisy_per_clean, test_num_rec, test_num_noise, test_area_ratio, test_num_remove, num_clean_test_img, num_noisy_test_img, is_Visualize = False, out_use = 'BN')
        
                # Testing the model
                acc, mod = model.Test(v_evidence=tsdata, v_labels=tslabel)
                
                ######################## Record the final accuracy ########
                results[lv].append(acc)

        for lv in noise_levels:
            filename = "tr_" + output_type + "_" + lv + ".txt"
            file = open(folder + filename, 'a+')
            for acc in results[lv]:
                file.write("Train Size: %d\n" %num_train)
                file.write("Final Accuracy: %lf\n" %acc)
            max_, mean, stdev = max(results[lv]), s.mean(results[lv]), s.stdev(results[lv])
            file.write(f'\nmax    {max_:.2f}')
            file.write(f'\nmean   {mean:.2f}')
            file.write(f'\nstdev  {stdev:.2f}')
            file.close()

def play_Digits_Noise(use_bk,tie_parameters,group=None):
    assert group is None or group in ('a','b','c','d')

    fd1 = '' if tie_parameters else '_ntie'
    fd2 = '' if use_bk else '_nbk'
    fd3 = f'_{group}' if group else ''
    folder = f"./yizuo_results/digits_noise{fd1}{fd2}{fd3}/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    ########################### Parameters ################################
    out_tp = [x for x in range(0, 10)]

    train_size = 2000
    size = 10
            
                ################# Generating Images ###########################
    per = 1 / 100
    num_clean_images = int(train_size * per)
    num_noisy_images = int(train_size - num_clean_images)

    model = dBN(size,use_bk,tie_parameters)        
    
    if group:
            group_range = range(0,3) if group=='a' else range(3,6) if group=='b' \
                          else range(6,9) if group=='c' else range(9,11) 
    else:
        group_range = range(0,11)
        
    results = {}
    for num_noise in group_range:
        results[num_noise * 2] = []

    for ntimes in range(5):
        # data
        trdata, trlabel = ddata.get(size, num_clean_images=num_clean_images, num_noisy_images=num_noisy_images, noisy_image_count=100, noise_count=8)
        trdata = np.array(trdata).transpose(1, 0, 2)
        #trdata, _, trlabel, __ = train_test_split(trdata, trlabel, train_size = num_train)
        #print(trdata.shape)
        trdata = trdata.transpose(1, 0, 2)
        trdata = [x for x in trdata]
    
        # Training the model
        model.Train(t_evidence=trdata, t_labels=trlabel)

        for num_noise in group_range:
            # Generate Testing Noises
            n_noise = num_noise * 2
            tsdata, tslabel = ddata.get(size, noisy_image_count=99, noise_count=n_noise)

            # Testing the model
            acc, mod = model.Test(v_evidence=tsdata, v_labels=tslabel)

            results[n_noise].append(acc)

    for num_noise in group_range:
        n_noise = num_noise * 2
        filename = "tr_" + str(n_noise) + ".txt"
        file = open(folder + filename, 'a+')
        for acc in results[n_noise]:
            ######################## Record the final accuracy ########
            file.write("Train Size: %d\n" %train_size)
            file.write("Final Accuracy: %lf\n" %acc)
        max_, mean, stdev = max(results[n_noise]), s.mean(results[n_noise]), s.stdev(results[n_noise])
        file.write(f'\nmax    {max_:.2f}')
        file.write(f'\nmean   {mean:.2f}')
        file.write(f'\nstdev  {stdev:.2f}')
        file.close()


def play_Digits_Sizes(use_bk,tie_parameters,group=None):
    assert group is None or group in ('a','b','c','d')

    fd1 = '' if tie_parameters else '_ntie'
    fd2 = '' if use_bk else '_nbk'
    fd3 = f'_{group}' if group else ''
    folder = f"./yizuo_results/digits_size{fd1}{fd2}{fd3}/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    ########################### Parameters ################################
    train_sizes = [25, 50, 100, 250, 500, 1000, 1500, 2000, 4000, 8000]
    #train_sizes = [8000]
    out_tp = [x for x in range(0, 10)]
    num_test = 2000
    size = 10
    #######################################################################

    model = dBN(size,use_bk,tie_parameters)
    
    if group:
        group_train_sizes = train_sizes[0:3] if group=='a' else train_sizes[3:6] if \
                group=='b' else train_sizes[6:8] if group=='c' else train_sizes[8:10]
    else:
        group_train_sizes = train_sizes
    
    for num_train in group_train_sizes:
        percent = 1/100
        num_clean_images = int(num_train * percent)
        num_noisy_images = num_train - num_clean_images
        best_acc = 0.00
        
        filename = "tr_" + str(num_train) + ".txt"
        file = open(folder + filename, 'a+')
        results = []
        for ntime in range(5):
            
                ################# Generating Images ###########################
            trdata, trlabel = ddata.get(size, num_clean_images=num_clean_images, num_noisy_images=num_noisy_images, noisy_image_count=100, noise_count=size)
            trdata = np.array(trdata).transpose(1, 0, 2)
            #trdata, _, trlabel, __ = train_test_split(trdata, trlabel, train_size = num_train)
            #print(trdata.shape)
            trdata = trdata.transpose(1, 0, 2)
            trdata = [x for x in trdata]
            tsdata, tslabel = ddata.get(size, noisy_image_count=99, noise_count=size)

            # Training the model
            model.Train(t_evidence=trdata, t_labels=trlabel)

            # Testing the model
            acc, mod = model.Test(v_evidence=tsdata, v_labels=tslabel)

            file.write("Train Size: %d\n" %num_train)
            file.write("Final Accuracy: %lf\n" %acc)
            results.append(acc)
            
        max_, mean, stdev = max(results), s.mean(results), s.stdev(results)
        file.write(f'\nmax    {max_:.2f}')
        file.write(f'\nmean   {mean:.2f}')
        file.write(f'\nstdev  {stdev:.2f}')
        file.close()


def play_Digits_Noise_20(use_bk,tie_parameters):

    fd1 = '' if tie_parameters else '_ntie'
    fd2 = '' if use_bk else '_nbk'
    folder = f"./yizuo_results/digits_noise_20{fd1}{fd2}/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    ########################### Parameters ################################
    out_tp = [x for x in range(0, 10)]

    num_train = 2000
    num_test = 2000
    size = 10
                ################# Generating Images ###########################

    trdata, trlabel = ddata.get(size, noisy_image_count=100, noise_count=8)
    trdata = np.array(trdata).transpose(1, 0, 2)
    #trdata, _, trlabel, __ = train_test_split(trdata, trlabel, train_size = num_train)
    #trdata.shape)
    trdata = trdata.transpose(1, 0, 2)
    trdata = [x for x in trdata]
            
    model = dBN(size,use_bk,tie_parameters)

    # Training the model
    model.Train(t_evidence=trdata, t_labels=trlabel)

    for num_noise in range(10, 11):
        # Generate Testing Noises
        n_noise = num_noise * 2
        tsdata, tslabel = ddata.get(size, noisy_image_count=100, noise_count=n_noise)

        # Testing the model
        acc, mod = model.Test(v_evidence=tsdata, v_labels=tslabel)

        ######################## Record the final accuracy ########
        filename = "tr_" + str(n_noise) + ".txt"
        file = open(folder + filename, 'a+')
        file.write("Train Size: %d\n" %num_train)
        file.write("Final Accuracy: %lf\n" %acc)
        file.close()


def play_Digits_Large_Size(use_bk,tie_parameters):

    fd1 = '' if tie_parameters else '_ntie'
    fd2 = '' if use_bk else '_nbk'
    folder = f"./yizuo_results/digits_large_size{fd1}{fd2}/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    "train"
    
    ########################### Parameters ################################
    train_sizes = [15000, 20000, 40000, 80000, 150000]
    out_tp = [x for x in range(0, 10)]
    num_test = 2000
    size = 10
    #######################################################################

    model = dBN(size,use_bk,tie_parameters)
    
    for num_train in train_sizes:
        best_acc = 0.00
        for ntime in range(5):
            
                ################# Generating Images ###########################
            trdata, trlabel = ddata.get(size, noisy_image_count=2000, noise_count=size)
            trdata = np.array(trdata).transpose(1, 0, 2)
            trdata, _, trlabel, __ = train_test_split(trdata, trlabel, train_size = num_train)
            #trdata.shape)
            trdata = trdata.transpose(1, 0, 2)
            trdata = [x for x in trdata]
            
            tsdata, tslabel = ddata.get(size, noisy_image_count=100, noise_count=size)
            
            # Training the model
            model.Train(t_evidence=trdata, t_labels=trlabel)

            # Testing the model
            acc, mod = model.Test(v_evidence=tsdata, v_labels=tslabel)

            filename = "tr_" + str(num_train) + ".txt"
            file = open(folder + filename, 'a+')
            file.write("Train Size: %d\n" %num_train)
            file.write("Final Accuracy: %lf\n" %acc)
            file.close()

def play_Digits_12_21(use_bk,tie_parameters):

    fd1 = '' if tie_parameters else '_ntie'
    fd2 = '' if use_bk else '_nbk'
    folder = f"./yizuo_results/digits_12_21{fd1}{fd2}/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    ########################### Parameters ################################
    train_sizes = [2000]
    out_tp = [x for x in range(0, 10)]
    num_test = 2000
    size = 10
    #######################################################################

    model = dBN(size,use_bk,tie_parameters)
    
    for num_train in train_sizes:
        per = 1 / 100
        num_clean_images = int(num_train * per)
        num_noisy_images = num_train - num_clean_images
        best_acc = 0.00
        
        filename = "tr_" + str(num_train) + ".txt"
        file = open(folder + filename, 'a+')
        results = []
        for ntime in range(5):
            
                ################# Generating Images ###########################
            trdata, trlabel = ddata.get(size, num_clean_images=num_clean_images, num_noisy_images=num_noisy_images, noise_type = "1-2", noisy_image_count=100, noise_count=3)
            trdata = np.array(trdata).transpose(1, 0, 2)
            #trdata, _, trlabel, __ = train_test_split(trdata, trlabel, train_size = num_train)
            #print(trdata.shape)
            trdata = trdata.transpose(1, 0, 2)
            trdata = [x for x in trdata]
            
            tsdata, tslabel = ddata.get(size, noise_type = "2-1", noisy_image_count=99, noise_count=3)

            # Training the model
            model.Train(t_evidence=trdata, t_labels=trlabel)

            # Testing the model
            acc, mod = model.Test(v_evidence=tsdata, v_labels=tslabel)

            file.write("Train Size: %d\n" %num_train)
            file.write("Final Accuracy: %lf\n" %acc)
            results.append(acc)
        max_, mean, stdev = max(results), s.mean(results), s.stdev(results)
        file.write(f'\nmax    {max_:.2f}')
        file.write(f'\nmean   {mean:.2f}')
        file.write(f'\nstdev  {stdev:.2f}')
        file.close()

def play_Digits_21_12(use_bk,tie_parameters):

    fd1 = '' if tie_parameters else '_ntie'
    fd2 = '' if use_bk else '_nbk'
    folder = f"./yizuo_results/digits_21_12{fd1}{fd2}/"
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    ########################### Parameters ################################
    train_sizes = [2000]
    out_tp = [x for x in range(0, 10)]
    num_test = 2000
    size = 10
    #######################################################################

    model = dBN(size,use_bk,tie_parameters)
                
    for num_train in train_sizes:
        best_acc = 0.00
        per = 1 / 100
        num_clean_images = int(num_train * per)
        num_noisy_images = num_train - num_clean_images
        
        filename = "tr_" + str(num_train) + ".txt"
        file = open(folder + filename, 'a+')
        results = []
        for ntime in range(5):
                ################# Generating Images ###########################
            trdata, trlabel = ddata.get(size, num_clean_images=num_clean_images, num_noisy_images=num_noisy_images, noise_type = "2-1", noisy_image_count=100, noise_count=3)
            trdata = np.array(trdata).transpose(1, 0, 2)
            #trdata, _, trlabel, __ = train_test_split(trdata, trlabel, train_size = num_train)
            #print(trdata.shape)
            trdata = trdata.transpose(1, 0, 2)
            trdata = [x for x in trdata]
            
            tsdata, tslabel = ddata.get(size, noise_type = "1-2", noisy_image_count=99, noise_count=3)

            # Training the model
            model.Train(t_evidence=trdata, t_labels=trlabel)

            # Testing the model
            acc, mod = model.Test(v_evidence=tsdata, v_labels=tslabel)

            file.write("Train Size: %d\n" %num_train)
            file.write("Final Accuracy: %lf\n" %acc)
            results.append(acc)
        max_, mean, stdev = max(results), s.mean(results), s.stdev(results)
        file.write(f'\nmax    {max_:.2f}')
        file.write(f'\nmean   {mean:.2f}')
        file.write(f'\nstdev  {stdev:.2f}')
        file.close()


class dBN:
    # compile digits model into circuit
    def __init__(self,size,use_bk,tie_parameters):
        u.show(f'\n===Compiling AC for Digit in {size}x{size} images, use_bk={use_bk}, tie_parameters={tie_parameters}')
        bn, inputs, output = dmodel.get(size=size,digits=range(10),testing=False,
                                use_bk=use_bk,tie_parameters=tie_parameters)
        self.circuit = tac.TAC(bn,inputs,output,trainable=True,profile=False)
        
    # trains circuit
    def Train(self,t_evidence,t_labels):
        u.show(f'\nTraining data {len(t_labels)}')
        # fixing batch size to 32
        self.circuit.fit(t_evidence,t_labels,loss_type='CE',metric_type='CA',batch_size=32)

    # compute accuracy
    def Test(self,v_evidence,v_labels):
        accuracy = 100*self.circuit.metric(v_evidence,v_labels,metric_type='CA',batch_size=128)
        u.show(f'Testing data  {len(v_labels)}')
        u.show(f'\nAC accuracy {accuracy:.2f}')
        return (accuracy, self.circuit)
        
class rBN:
    # compile rectangle model into circuit
    def __init__(self,size,output,use_bk,tie_parameters):
        u.show(f'\n===Compiling AC for rectangle {output} in {size}x{size} images, use_bk={use_bk}, tie_parameters={tie_parameters}')
        bn, inputs = rmodel.get(size,output,testing=False,
                        use_bk=use_bk,tie_parameters=tie_parameters)
        self.circuit = tac.TAC(bn,inputs,output,trainable=True,profile=False)
     
    # trains circuit
    def Train(self,t_evidence,t_labels):    
        u.show(f'\nTraining data {len(t_labels)}')
        # fixing batch size to 32
        self.circuit.fit(t_evidence,t_labels,loss_type='CE',metric_type='CA',batch_size=32)

    # compute accuracy
    def Test(self,v_evidence,v_labels):
        accuracy = 100*self.circuit.metric(v_evidence,v_labels,metric_type='CA',batch_size=128)
        u.show(f'Testing data  {len(v_labels)}')
        u.show(f'\nAC accuracy {accuracy:.2f}')
        return (accuracy, self.circuit)        
