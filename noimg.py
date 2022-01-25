# 计时
import os
import time
import numpy as np
import Config
import Net.net_new as Net
import alphabets
import cv2
import lib.convert
import lib.dataset
import torch
from models import build_model
from post_processing import get_post_processing
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont



###################
# DBNet 模型处理部分
# -*- coding: utf-8 -*-
# @Time    : 2019/8/24 12:06
# @Author  : zhoujun


def get_transforms(transforms_config):
    tr_list = []
    for item in transforms_config:
        if 'args' not in item:
            args = {}
        else:
            args = item['args']
        cls = getattr(transforms, item['type'])(**args)
        tr_list.append(cls)
    tr_list = transforms.Compose(tr_list)
    return tr_list


def resize_image(img, short_size):
    height, width, _ = img.shape
    if height < width:
        new_height = short_size
        new_width = new_height / height * width
    else:
        new_width = short_size
        new_height = new_width / width * height
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img


class Pytorch_model:
    def __init__(self, model_path, post_p_thre=0.7):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        '''
        checkpoint = torch.load(model_path, map_location="cpu")
        config = checkpoint['config']
        config['arch']['backbone']['pretrained'] = False
        self.model = build_model(config['arch'])
        self.post_process = get_post_processing(config['post_processing'])

        self.post_process.box_thresh = post_p_thre
        self.img_mode = config['dataset']['train']['dataset']['args']['img_mode']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        self.transform = []
        for t in config['dataset']['train']['dataset']['args']['transforms']:
            if t['type'] in ['ToTensor', 'Normalize']:
                self.transform.append(t)
        self.transform = get_transforms(self.transform)

    def predict(self, img_path: str, is_output_polygon=False, short_size: int = 1024):
        '''
        对传入的图像进行预测，支持图像地址
        :param img_path: 图像地址
        :return:
        '''
        assert os.path.exists(img_path), 'file is not exists'
        img = cv2.imread(img_path, 1 if self.img_mode != 'GRAY' else 0)
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_r = img.copy()
        h, w = img.shape[:2]
        img = resize_image(img, short_size)
        tensor = self.transform(img)
        tensor = tensor.unsqueeze_(0)

        batch = {'shape': [(h, w)]}
        with torch.no_grad():
            preds = self.model(tensor)
            box_list, score_list = self.post_process(batch, preds, is_output_polygon=is_output_polygon)
            box_list, score_list = box_list[0], score_list[0]
            if len(box_list) > 0:
                idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
                box_list, score_list = box_list[idx], score_list[idx]
            else:
                box_list, score_list = [], []
        return box_list, img_r


def crnn_recognition(cropped_image, model):
    alphabet = alphabets.alphabet

    # 图像预处理
    image = cropped_image.convert('L')

    w = int(image.size[0] / (280 * 1.0 / Config.infer_img_w))

    transformer = lib.dataset.resizeNormalize((w, Config.img_height))
    image = transformer(image)
    image = image.view(1, *image.size())
    image = Variable(image)
    # 推理
    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    converter = lib.convert.strLabelConverter(alphabet)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    return sim_pred


def get_retc(box_list):
    points = []
    for i, box in enumerate(box_list):
        point = [np.min(box, axis=0).tolist(), np.max(box, axis=0).tolist()]
        points.append(point)

    return points


"""
裁剪图像，之后将送入CRNN
"""


def crop_lines(points, img,IMGPATH):
    """
    @return     返回data列表，列表由字典组成，每个字典包含坐标、图像内容两个key
    """

    data = []
    for point in points:
        datum = {}
        box = (point[0][0], point[0][1], point[1][0], point[1][1])
        datum["img"] = img.crop(box)
        datum["point"] = point
        data.append(datum)
        ##
        # 画框、 保存
        # draw = ImageDraw.Draw(img)
        # draw.rectangle([point[0][0], point[0][1], point[1][0], point[1][1]], outline='red', width=2)
        # img.save(f"out_img/crop_result_{IMGPATH.split('.')[0].split('/')[-1]}.jpg")
        ##
    return data


"""
数据送入CRNN，进行处理
"""


###########


# after  crnn
# input: 
# result -> list
# result的格式为一个列表，列表中的元素由字典组成，每个字典由识别出的中文以及该图像坐标两个数据组成
# return:
# I -> dict
# 项目名和该项目的坐标
# 后处理中”代码“和”号码“需要转换成”发票代码“和”发票号码“


def contain(word: str, key: str):
    c = 0
    for i in key:
        if i in word:
            c += 1
    return c == len(key)


def fitName(results: list):
    L = ["代码", "号码", "车号", "证号", "日期", "上车", "下车", "单价", "里程", "等候", "金额", "卡号", "原额", "余额"]
    I = {}

    for l in L:
        for result in results:
            if contain(result["label"], l):
                if l in I.keys():  # 如果检测到了相同的关键词（很有可能是大框包小框），那就面积框大的那个候选框
                    x0, y0 = I[l][0]
                    x1, y1 = I[l][1]
                    a0, b0 = result["point"][0]
                    a1, b1 = result["point"][1]
                    # 稍微判断一下是否是重复检测的情况（两框距离不要太远）
                    if abs(x0 - a0) < (x1 - x0):
                        area_ori = (x1 - x0) * (y1 - y0)
                        area_cur = (a1 - a0) * (b1 - b0)
                        if area_cur > area_ori:
                            I[l] = result["point"]
                else:
                    I[l] = result["point"]
    return I


def fitlabel(data, I: dict):
    O = {}
    for k, v in I.items():
        mlength = 1e10
        # 坐标的底线需要与所有候选框的底线最接近   
        # 此时data 里面已经有的内容：point、label、img
        y = v[1][1]
        for datum in data:
            length = abs(datum["point"][1][1] - y)
            if length < mlength and length != 0 and datum["label"]!="":
                O[k] = datum["label"]
                mlength = length
    return O


# ALL

def run(MODEL_DB_PATH, MODEL_CRNN_PATH, IMGPATH):
    a = time.time()
    print(IMGPATH)
    OUT.write(IMGPATH)
    model = Pytorch_model(MODEL_DB_PATH, post_p_thre=0.5)
    boxes_list, img = model.predict(IMGPATH)
    print(f"DBNet所使用的时间：{time.time() - a}s")
    OUT.write(f"\nDBNet所使用的时间：{time.time() - a}s\n")
    a = time.time()

    # DBNet 模型处理部分 结束 
    # 输出 boxes_list, img
    ###########
    """
    DB网络检测，传出矩形框预测坐标信息，下面对传出的信息进行处理
    input:
        img
        box_list
    """
    ##
    # DBNet

    points = get_retc(boxes_list)
    # 输入的是 cv 格式的img，需要转到PIL
    img = Image.fromarray(img)
    data = crop_lines(points, img, IMGPATH)
    # CRNN
    # 模型加载，最好常驻内存
    alphabet = alphabets.alphabet
    nclass = len(alphabet) + 1
    model = Net.CRNN(nclass,hidden_unit=Config.hidden_unit)
    model.load_state_dict(torch.load(MODEL_CRNN_PATH, map_location='cpu')["state_dict"])

    # 传递参数：data
    # 列表由字典组成，每个字典包含坐标、图像内容两个key
    # 处理之后增加文字label的key
    all = 0
    for datum in data:
        if datum["point"][1][1]-datum["point"][0][1]>datum["point"][1][0]-datum["point"][0][0]:
            datum["label"] = ""
            continue
        datum["label"] = crnn_recognition(datum["img"], model)
        # ttf = ImageFont.truetype("simsun.ttc", size=int((datum["point"][1][1]-datum["point"][0][1])/3))
        # draw.text((datum["point"][0][0], datum["point"][0][1]), datum["label"], font=ttf, fill=(255, 0, 0))
    # img.save(f"out_img/result_withText_{IMGPATH.split('.')[0].split('/')[-1]}.jpg")
    ##
    # 添加标签的文字
    ##
    I = fitName(data)
    O = fitlabel(data, I)
    print(O)
    print(f"CRNN所使用的时间：{time.time() - a}s")
    import json  
    OUT.write(f"\nCRNN所使用的时间：{time.time() - a}s\n")
    OUT.write("识别结果（json格式）：")
    OUT.write(json.dumps(O,ensure_ascii=False))
    
if __name__ == "__main__":
    OUT = open("result.log","w")
    MODEL_DB_PATH = "saved_models/DBNet/loc.pth"
    MODEL_CRNN_PATH = "saved_models/CRNN/model_best.pth"
    DIR = "test_imgs"
    for root, dirs, files in os.walk(DIR):
        for file in files:
            print(1)
            if file.split(".")[-1] == "jpg":
                path = os.path.join(root, file)
                run(MODEL_DB_PATH, MODEL_CRNN_PATH, path.replace('\\','/'))
                print("***********")
                OUT.write("---------------\n")
    OUT.close()