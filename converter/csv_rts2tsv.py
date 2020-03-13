import pandas as pd
import argparse
import os

classes = ['2.1', '2.4', '3.1', '3.24', '3.27', '4.1.1', '4.1.2', '4.1.3',
           '4.1.4', '4.1.5', '4.1.6', '4.2.1', '4.2.2', '4.2.3', '5.19.1',
           '5.19.2', '5.20', '8.22.1', '8.22.2', '8.22.3']

def csv_rtv2tsv(csv_file, annotation_dir):
    df = pd.read_csv(csv_file)
    cur_name = ''

    for index, row in df.iterrows():
        if row['filename'] != cur_name:
            if cur_name != '':
                if cur_image['class']:
                    new_df = pd.DataFrame.from_dict(cur_image)                

                    path = os.path.join(annotation_dir, cur_name.split('.')[0] + '.tsv')
                    new_df.to_csv(path, sep='\t', index=False)

            cur_name = row['filename']
            cur_image = {'class': [], 'xtl' : [], 'ytl' : [], 'xbr' : [], 'ybr' : [], 
                         'temporary' : [], 'occluded' : [], 'data' : []}

        if row['sign_class'].startswith('3_24'):
            class_ = '3.24'
        else:            
            class_ = '.'.join(row['sign_class'].split('_'))

        if not class_ in classes: #filter
            continue

        cur_image['class'].append(class_)

        xtl = row['x_from']
        cur_image['xtl'].append(xtl)

        ytl = row['y_from']
        cur_image['ytl'].append(ytl)

        xbr = xtl + row['width']
        cur_image['xbr'].append(xbr)
        
        ybr = ytl + row['height']
        cur_image['ybr'].append(ybr)

        cur_image['temporary'].append('false')
        cur_image['occluded'].append('false')
        cur_image['data'].append('')

    if cur_name != '':
        if cur_image['class']:
            new_df = pd.DataFrame.from_dict(cur_image)                

            path = os.path.join(annotation_dir, cur_name.split('.')[0] + '.tsv')
            new_df.to_csv(path, sep='\t', index=False)        


parser = argparse.ArgumentParser(description='Convert RTS csv to IceVision tsv')

parser.add_argument('-i', '--input', help='RTS csv file', type=str,  required=True)
parser.add_argument('-o', '--outdir', help='output folder', type=str, required=True)

args = vars(parser.parse_args())        

csv_rtv2tsv(args['input'], args['outdir'])        