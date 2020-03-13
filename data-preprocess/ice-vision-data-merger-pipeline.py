#!/usr/bin/env python
# coding: utf-8

# In[93]:


import os
from shutil import copyfile
import json

print("cwd = ", os.getcwd())

current_folder = os.getcwd()
#extracted_train_data = os.path.join(current_folder, "extracted_train_data")
extracted_train_data = "/dataset/training/"
#annotations_dir = '/data/annotations'
copied_train_data = "/data/dataset/training/"


# In[100]:


data_location = "/dataset/training/"

files_list = []
neural_net_list = []
linear_mappings = []

for subdir, dirs, files in os.walk(data_location):
    for file in set(files):
        if file.endswith('.pnm'):
            current_file = os.path.join(subdir, file)
            files_list.append(current_file)
print(len(files_list))

prev_file_number = 0
prev_file_dir_name = ""
prev_neural_net = ""
counter = 0
linear_list = []

########################################## EDIT #####################################################################
for file in sorted(files_list):
    file_name_split = file.split('/')

    file_number = int(file_name_split[-1].split(".pnm")[0])
    dir_name = file_name_split[-3] + file_name_split[-2]

    counter += file_number - prev_file_number

    if(prev_file_dir_name != dir_name):
        counter = 0
        neural_net_list.append(file)
        prev_neural_net = file
        linear_list = []
    else:   

        if(counter >= 5):
            neural_net_list.append(file)
            linear_mappings.append({ "linear_list": linear_list, "predecessor": prev_neural_net, "successor": file })
            counter = 0
            prev_neural_net = file
            linear_list = []
        else:
            #linear_mappings[file] = "linear"
            linear_list.append(file)
         #   print("making linear", file)

    prev_file_number = file_number 
    prev_file_dir_name = dir_name

 
    
    
with open('linear_mappings.json', 'w') as outfile:
    json.dump(linear_mappings, outfile)


    

# for file in file_body:
#     if (file_body[file] == "neuralnet"):
#         print(file)
        

        
# for file in file_body:
#     if (file_body[file] == "linear"):
#         print(file)       


# In[97]:


#neural_net_list[] - list of images to be sent to neural network

import os
import glob
from mmdet.apis import init_detector, inference_detector, show_result, write_result
import time
import datetime

config_file = '/root/ws/mmdetection-icevision/configs/dcn/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_all_classes.py'
#model = init_detector(config_file, checkpoint_file, device='cuda:0')
#epch_count = 1
#for epochs in glob.glob(os.path.join('/data_tmp/icevisionmodels/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_all_classes/', '*.pth')): 
checkpoint_file = '/data/trained_models/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_135_classes/epoch_15.pth'
        #checkpoint_file = epochs

        # build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

TEST_RESULT_PATH = "/data/test_results/"
img_count = 0
#print(img_count)
FINAL_ONLINE_TEST_PATH = "/data/train_subset/"
#FINAL_ONLINE_TEST_PATH = '/data/test_results/2018-02-13_1418/left/'
#for TEST_SET_PATH in (FINAL_ONLINE_TEST_PATH + "2018-02-16_1515_left/", FINAL_ONLINE_TEST_PATH + "2018-03-16_1424_left/", FINAL_ONLINE_TEST_PATH + "2018-03-23_1352_right/"):
#print(TEST_SET_PATH)
#imgs = glob.glob('/dataset/training/**/*.pnm', recursive=True)
for img in neural_net_list:
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        print ("time =", st)
        #imgs = ['test.jpg', '000000.jpg']
        #print(img) # /dataset/training/2018-02-13_1418/left/020963.pnm    --> required format 2018-02-13_1418_left/000033
        name = img.split("/") # ['', 'home', 'luminosity', 'ws', 'icevision', 'data', 'final', '2018-02-16_1515_left', '001887.jpg'] 
        #print(name) 
        base = name[-1].split(".")[0] # ['001887', 'jpg'] 
        #print(base)
        name = name[-3] + "_" + name[-2]
        tmp = name
        name = name + "/" + base
        #print(name)
        ######## Remove
        #name_tmp = base.split("_") 
        #name = name_tmp[0] + "_" + name_tmp[1] + "_" + name_tmp[2] + "/" + name_tmp[-1]
        #name = "annotation_train_subset/" + base
        #base_list = base.split("_")
        #name = base_list[0] + "_" + base_list[1] + "_" + base_list[2] + "/" + base_list[3]
        ##########Remove
        result = inference_detector(model, img)
        #write_result(name, result, model.CLASSES, out_file=os.path.join(TEST_RESULT_PATH, 'my_test_multi_scale_epch_{}.tsv'.format(epch_count))) # use name instead name1 for hackthon submission
        #show_result(img, result, model.CLASSES, out_file= TEST_RESULT_PATH + 'bboxs/' + tmp + ".pnm")
        write_result(name, result, model.CLASSES, out_file=os.path.join(TEST_RESULT_PATH, 'my_test_epch_15_interpolation.tsv')) # use name instead name1 for hackthon submission
        img_count+=1
        #print(img_count)
        print("num = %d name = %s" %(img_count,name))
        


# In[103]:


import os
import glob
import csv
from shutil import copyfile


def linear_interpolation(pred, succ, lin_images, input_tsv, step, out_tsv):
    lin_images.sort()
    succ_base_name = os.path.basename(succ).split(".")[0]
    pred_base_name = os.path.basename(pred).split(".")[0]
    #copyfile(input_tsv, out_tsv)
    tsv_file = csv.reader(open(input_tsv, "r"), delimiter="\t")
    prd_classes = []
    suc_classes = []
    prd_keys = set()
    suc_keys = set()
    
    for row in tsv_file:
        
#         print("row = ", row)
#         print('ped_keys = ', prd_keys)
#         print('suc_keys = ', suc_keys)
        # frame	xtl	ytl	xbr	ybr	class	temporary	data
        # 2018-02-13_1418_left/020963	679	866	754	941	3.27
        prd_record = {} #defaultdict(list)
        suc_record = {} #defaultdict(list)
        #print("row[0] = ", row[0])
        x = os.path.join(os.path.basename(os.path.dirname(pred)),os.path.basename(pred))
        y = os.path.basename(os.path.dirname(os.path.dirname(pred)))
        dict_key = y + "_" + x
        
        x2 = os.path.join(os.path.basename(os.path.dirname(succ)),os.path.basename(succ))
        y2 = os.path.basename(os.path.dirname(os.path.dirname(succ)))
        dict_key2 = y2 + "_" + x2
#         print('y = ', y)
#         print("x = ", x)
#         print("dict_key = ", dict_key.split('.')[0])
        if row[0] ==  dict_key.split('.')[0]:
            if row[5] not in prd_keys:
                print("pred check cleared")
                prd_record["class"] = row[5]
                prd_record["xtl"] = row[1]
                prd_record["ytl"] = row[2]
                prd_record["xbr"] = row[3]
                prd_record["ybr"] = row[4]
                print("prd_record['ybr'] = ", prd_record["ybr"])
                prd_keys.add(row[5])
                # #prd_record[row[5]].append(row[1]) #xtl
                # prd_record[row[5]].append(row[2]) #ytl
                # prd_record[row[5]].append(row[3]) #xbr
                # prd_record[row[5]].append(row[4]) #ybr
                prd_classes.append(prd_record)
            else:
                for prd_class in prd_classes:
                    if prd_class["class"] == row[5]:
                        del prd_class
                        print("del prd_class")


        elif row[0] == dict_key2.split('.')[0]:
            print("Succ check cleared")
            if row[5] not in suc_keys:
                suc_record["class"] = row[5]
                suc_record["xtl"] = row[1]
                suc_record["ytl"] = row[2]
                suc_record["xbr"] = row[3]
                suc_record["ybr"] = row[4]
                suc_keys.add(row[5])
                # suc_record[row[5]].append(row[1])
                # suc_record[row[5]].append(row[2])
                # suc_record[row[5]].append(row[3])
                # suc_record[row[5]].append(row[4])
                suc_classes.append(suc_record)
            else:
                for suc_class in suc_classes:
                    if suc_class["class"] == row[5]:
                        del suc_class
                        print("del prd_class")
                        
    #print("prd_keys = ", prd_keys)
    
    common_classes = prd_keys.intersection(suc_keys)
    print(common_classes)
    for common_class in common_classes:
        for prd_class in prd_classes:
            if prd_class["class"]  == common_class:
                for suc_class in suc_classes:
                    if suc_class["class"]  == common_class:
                        xtl_gr = (int(prd_class["xtl"]) - int(suc_class["xtl"])) / step
                        ytl_gr = (int(prd_class["ytl"]) - int(suc_class["ytl"])) / step
                        xbr_gr = (int(prd_class["xbr"]) - int(suc_class["xbr"])) / step
                        ybr_gr = (int(prd_class["ybr"]) - int(suc_class["ybr"])) / step
                        print(xtl_gr, ytl_gr, xbr_gr, ybr_gr)
                        
                        for f in lin_images:
                            curr_base = os.path.basename(f).split(".")[0]
#                             print("curr_base = ", curr_base)
#                             print("pred_base_name = ", pred_base_name)
#                             print("f = ", f)
                            
                            factor = int(curr_base) - int(pred_base_name) 
                            curr_xtl = int(prd_class["xtl"]) + (factor * xtl_gr) 
                            curr_ytl = int(prd_class["ytl"]) + (factor * ytl_gr) 
                            curr_xbr = int(prd_class["xbr"]) + (factor * xbr_gr) 
                            curr_ybr = int(prd_class["ybr"]) + (factor * ybr_gr)
                            temp = ''
                            with open(out_tsv, mode = 'a') as result_file:
                                result_file_writer = csv.writer(result_file, delimiter = '\t')
                                result_file_writer.writerow([f, str(curr_xtl), str(curr_ytl), str(curr_xbr), str(curr_ybr), prd_class["class"], temp, temp])


# In[105]:



#load the linear mappings.json
import csv

linear_mappings = "/root/ws/mmdetection-icevision/data-preprocess/linear_mappings.json"
input_tsv = os.path.join(TEST_RESULT_PATH, 'my_test_epch_15_interpolation_copy.tsv')
out_tsv = os.path.join(TEST_RESULT_PATH, 'my_test_epch_15_interpolation_copy.tsv')


interpolation_mappings = []
with open(linear_mappings, 'r') as f:
    interpolation_mappings = json.load(f)

for i in interpolation_mappings:
    pred = i["predecessor"] 
    succ = i['successor']
    interpol_list = i['linear_list']
    step = 5
    linear_interpolation(pred, succ, interpol_list, input_tsv, step, out_tsv)
    
#     if i["predecessor"] == neural_net_list[100]:
#         break
    
    
    


# In[70]:


# trial code 
# extracted_train_data = "/home/sgj/temp/test_data/2018-03-16_1324"
# for subdir, dirs, files in os.walk(extracted_train_data):
#     print("subdir = ", subdir)
#     for file in files:
#         if file.endswith('.jpg'):
#             current_file = os.path.join(subdir, file)
#             #folder_name = os.path.basename(os.path.dirname(current_file))
#             #expected_name = folder_name + '_' + os.path.basename(current_file)
#             y = file.split("_")
#             expected_name = y[0] + "_" + y[1] + "_left_jpgs_" + y[2]
#             absolute_expected_name = os.path.join(os.path.dirname(current_file),expected_name)
#             os.rename(current_file, absolute_expected_name)
            



# In[37]:


extracted_train_data = "/home/sgj/temp/train_data/2018-02-13_1418_left_jpgs"

for subdir, dirs, files in os.walk(extracted_train_data):
    print("subdir = ", subdir)
    for file in files:
        if file.endswith('.jpg'):
            current_file = os.path.join(subdir, file)
            folder_name = os.path.basename(os.path.dirname(current_file))
            expected_name = folder_name + '_' + os.path.basename(current_file)
            absolute_expected_name = os.path.join(os.path.dirname(current_file),expected_name)
            os.rename(current_file, absolute_expected_name)
            


# In[25]:


# move out un-annotated images - 
# ARGS - 
    # Annotations data tsv
    # Extracted images folder
    # Destination folder for annotated_data
    
import os

annotation_data_tsv_folder = "/home/sgj/nvme/ice-vision/annotations/test/all_validation_annotations"
extracted_images_folder = "/home/sgj/temp/test_data/all_validation_images"
#dest_annotated_imgs = "/home/sgj/nvme/ice-vision/annotated_data/val"
dest_annotated_imgs = "/home/sgj/temp/ice-vision/annotated_data/val"

os.makedirs(dest_annotated_imgs)

img_count = 0
for root, dirs, files in os.walk(annotation_data_tsv_folder):
    for name in files:
        if name.endswith('.tsv'):
            prefix = name.split(".")[0]
            image_name = prefix + ".jpg"
            expected_img_path = os.path.join(extracted_images_folder, image_name)
            new_image_path = os.path.join(dest_annotated_imgs, image_name)
            if os.path.exists(expected_img_path):
                img_count = img_count + 1
                os.rename(expected_img_path, new_image_path)
            else:
                print("image missing-----------------------")
            

print("total images  = ", img_count)


# In[18]:


temp = "2018-02-13_1418_left_jpgs_014810.tsv"
temp.split(".")[0]


# In[3]:



for subdir, dirs, files in os.walk(copied_train_data):
    print("subdir = ", subdir)
    for file in files:
        if file.endswith('.pnm'):
            current_file = os.path.join(subdir, file)
            print('current file = ', current_file)
            cam_dir = current_file.split('/')[-2]
            #print("cam dir = ", cam_dir)
            date_dir = current_file.split('/')[-3]
            #print("date_dir = ", date_dir)
            expected_folder = '/data/train_subset/'
            expected_file_name = date_dir + "_" + cam_dir + "_" + os.path.basename(current_file)
            expected_file_path = os.path.join(expected_folder, expected_file_name)
            #copyfile(current_file, dst_file_path)
            os.rename(current_file, expected_file_path)
            print("expected_file_path = ", expected_file_path)


# In[4]:





# In[ ]:




