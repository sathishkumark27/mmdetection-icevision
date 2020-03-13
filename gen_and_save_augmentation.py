import argparse
from albu import orig2aug

parser = argparse.ArgumentParser(description='convert tsv to coco dataset')

parser.add_argument('-c', '--categories', help='path to json categories', type=str, default=None)
parser.add_argument('-r', '--root', help='root folder', type=str, required=True)
parser.add_argument('-e', '--extension', help='image extension', type=str, default='.pnm')
parser.add_argument('-imdir', '--image_dir', help='image directory', type=str, required=True)
parser.add_argument('-anndir', '--annotation_dir', help='annotation directory', type=str, required=True)
parser.add_argument('-d', '--data', help='train or validation set', required=True, choices=['train', 'val'])
parser.add_argument('-imsavedir', '--image_save_dir', help='image storage directory', type=str, required=True)
parser.add_argument('-annsavedir', '--annotation_save_dir', help='annotation storage directory', type=str, required=True)

args = vars(parser.parse_args())

orig2aug(**args)


