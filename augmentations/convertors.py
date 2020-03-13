
from utils import *

###################### HLS #############################

def hls(image,src='RGB'):
    verify_image(image)
    if(is_list(image)):
        image_HLS=[]
        image_list=image
        for img in image_list:
            eval('image_HLS.append(cv2.cvtColor(img,cv2.COLOR_'+src.upper()+'2HLS))')
    else:
        image_HLS = eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2HLS)')
    
    return image_HLS

def hue(image,src='RGB'):
    verify_image(image)
    if(is_list(image)):
        image_Hue=[]
        image_list=image
        for img in image_list:
            image_Hue.append(hls(img,src)[:,:,0])
    else:
        image_Hue= hls(image,src)[:,:,0]
    return image_Hue

def lightness(image,src='RGB'):
    verify_image(image)
    if(is_list(image)):
        image_lightness=[]
        image_list=image
        for img in image_list:
            image_lightness.append(hls(img,src)[:,:,1])
    else:
        image_lightness= hls(image,src)[:,:,1]
    return image_lightness

def saturation(image,src='RGB'):
    verify_image(image)
    if(is_list(image)):
        image_saturation=[]
        image_list=image
        for img in image_list:
            image_saturation.append(hls(img,src)[:,:,2])
    else:
        image_saturation= hls(image,src)[:,:,2]
    return image_saturation

###################### HSV #############################

def hsv(image,src='RGB'):
    verify_image(image)
    if(is_list(image)):
        image_HSV=[]
        image_list=image
        for img in image_list:
            eval('image_HSV.append(cv2.cvtColor(img,cv2.COLOR_'+src.upper()+'2HSV))')
    else:
        image_HSV = eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2HSV)')
    return image_HSV

def value(image,src='RGB'):
    verify_image(image)
    if(is_list(image)):
        image_value=[]
        image_list=image
        for img in image_list:
            image_value.append(hsv(img,src)[:,:,2])
    else:
        image_value= hsv(image,src)[:,:,2]
    return image_value

###################### BGR #############################

def bgr(image, src='RGB'):
    verify_image(image)
    if(is_list(image)):
        image_BGR=[]
        image_list=image
        for img in image_list:
            eval('image_BGR.append(cv2.cvtColor(img,cv2.COLOR_'+src.upper()+'2BGR))')
    else:
        image_BGR= eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2BGR)')
    return image_BGR

###################### RGB #############################
def rgb(image, src='BGR'):
    verify_image(image)
    if(is_list(image)):
        image_RGB=[]
        image_list=image
        for img in image_list:
            eval('image_RGB.append(cv2.cvtColor(img,cv2.COLOR_'+src.upper()+'2RGB))')
    else:
        image_RGB= eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2RGB)')
    return image_RGB

def red(image,src='BGR'):
    verify_image(image)
    if(is_list(image)):
        image_red=[]
        image_list=image
        for img in image_list:
            i= eval('cv2.cvtColor(img,cv2.COLOR_'+src.upper()+'2RGB)')
            image_red.append(i[:,:,0])
    else:
        image_red= eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2RGB)[:,:,0]')
    return image_red

def green(image,src='BGR'):
    verify_image(image)
    if(is_list(image)):
        image_green=[]
        image_list=image
        for img in image_list:
            i= eval('cv2.cvtColor(img,cv2.COLOR_'+src.upper()+'2RGB)')
            image_green.append(i[:,:,1])
    else:
        image_green= eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2RGB)[:,:,1]')
    return image_green

def blue(image,src='BGR'):
    verify_image(image)
    if(is_list(image)):
        image_blue=[]
        image_list=image
        for img in image_list:
            i=eval('cv2.cvtColor(img,cv2.COLOR_'+src.upper()+'2RGB)')
            image_blue.append(i[:,:,2])
    else:
        image_blue= eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2RGB)[:,:,2]')
    return image_blue