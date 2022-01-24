import alphabets

proj_name = 'crnn_mobilenetV3'
hidden_unit = 128
raw_folder = ''
train_data = r'/data/zjj/dataset/train'
test_data = r'/data/zjj/dataset/test'
pretrain_model = r''
output_dir = './crnnout'
random_sample = True
random_seed = 3484
using_cuda = True
keep_ratio = False
gpu_id = '3'
data_worker = 6
batch_size = 512
img_height = 32
img_width = 160
alphabet = alphabets.alphabet
epoch = 200 
display_interval = 10
save_interval = 100
test_interval = 100
test_disp = 100
test_batch_num = 32
lr = 0.0008
beta1 = 0.5
infer_img_w = 160
