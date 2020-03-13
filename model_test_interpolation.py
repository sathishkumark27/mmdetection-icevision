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
for img in glob.glob(os.path.join(FINAL_ONLINE_TEST_PATH, '*.pnm')):
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
        base_list = base.split("_")
        name = base_list[0] + "_" + base_list[1] + "_" + base_list[2] + "/" + base_list[3]
        ##########Remove
        result = inference_detector(model, img)
        #write_result(name, result, model.CLASSES, out_file=os.path.join(TEST_RESULT_PATH, 'my_test_multi_scale_epch_{}.tsv'.format(epch_count))) # use name instead name1 for hackthon submission
        #show_result(img, result, model.CLASSES, out_file= TEST_RESULT_PATH + 'bboxs/' + tmp + ".pnm")
        write_result(name, result, model.CLASSES, out_file=os.path.join(TEST_RESULT_PATH, 'my_test_epch_15_stress_train_all.tsv')) # use name instead name1 for hackthon submission
        img_count+=1
        #print(img_count)
        print("num = %d name = %s" %(img_count,name))
        

