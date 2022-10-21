import easyocr
import os
import torch.backends.cudnn as cudnn
import yaml
# from train import train
from trainer.utils import AttrDict
import pandas as pd

def get_config(file_path):
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == 'None':
        characters = ''
        for data in opt['select_data'].split('-'):
            csv_path = os.path.join(opt['train_data'], data, 'labels.csv')
            df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
            all_char = ''.join(df['words'])
            characters += ''.join(set(all_char))
        characters = sorted(set(characters))
        opt.character= ''.join(characters)
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt

opt = get_config("./trainer/config_files/en_filtered_config.yaml")


impath = "./test_img1.png"
reader = easyocr.Reader(lang_list=['en'], user_network_directory='./user_network', model_storage_directory="./model", recog_network="custom_model", opt=opt)
result = reader.readtext(impath, detail=0)
print(result)
