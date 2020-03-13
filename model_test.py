import os
import glob
from mmdet.apis import init_detector, inference_detector, show_result, write_result

# config_file = '/home/sathish/ws/mmdetection/configs/cascade_rcnn_r50_fpn_1x.py'
# checkpoint_file = '/home/sathish/ws/mmdetection/train_model/latest.pth'
config_file = '/home/luminosity/ws/icevision/mmdetection/configs/dcn/cascade_rcnn_dconv_c3-c5_r50_fpn_1x.py'
checkpoint_file = '/home/luminosity/ws/icevision/mmdetection/work_dirs/cascade_rcnn_dconv_c3-c5_r50_fpn_1x/epoch_5.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
#print(model)

# test a single image and show the results
#img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
#result = inference_detector(model, img)
#print((result[0]))
#print(model)
#print(model.CLASSES)
#show_result(img, result, model.CLASSES, out_file="image_1.jpg")
#write_result(img, result, model.CLASSES, out_file="result.tsv")

# test a list of images and write the results to image files
TEST_SET_PATH = "/home/luminosity/ws/icevision/data/annotated_data/val/"
TEST_RESULT_PATH = "/home/luminosity/ws/icevision/val_results/dcn_10e/"
img_count = 0
print(img_count)
for img in glob.glob(os.path.join(TEST_SET_PATH, '*.jpg')):
    #imgs = ['test.jpg', '000000.jpg']
    #print(img) # /home/luminosity/ws/icevision/data/annotated_data/val/2018-03-16_1324_left_jpgs_035026.jpg    --> required format 2018-02-13_1418_left/000033
    base = os.path.basename(img) # 2018-03-16_1324_left_jpgs_035026.jpg 
    #print(base)
    name1 = base.split(".") # ['2018-03-16_1324_left_jpgs_035026', 'jpg']
    name2 = name1[0].split("_") # ['2018-03-16', '1324', 'left', 'jpgs', '035026']  
    #name = str(name2[0]) + "_" + str(name2[1]) + "_" + str(name2[2]) + "/" + str(name2[4])  # use name for hackthon submission
    score_check_name = str("all_validation_annotations/") + name1[0] # only for our model score generation, USE NAME FOR write_image FOR HACKTHON SUBMISSION
    print(score_check_name) 
    result = inference_detector(model, img)
    #for i, result in enumerate(inference_detector(model, imgs)):    
    #show_result(img, result, model.CLASSES, out_file=os.path.join(TEST_RESULT_PATH, 'bboxs/', 'result_{}.jpg'.format(img_count)))
    #show_result(img, result, model.CLASSES, out_file= TEST_RESULT_PATH + 'bboxs/' + name1[0] + "_" + name2[4] + ".jpg")
    write_result(score_check_name, result, model.CLASSES, out_file=os.path.join(TEST_RESULT_PATH, "my_test_epch_5.tsv")) # use name instead name1 for hackthon submission
    img_count+=1 