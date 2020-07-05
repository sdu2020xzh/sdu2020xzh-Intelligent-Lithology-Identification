import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchcv.transform import resize
from torchcv.models import DSOD, SSDBoxCoder
from torchcv.utils.config import opt
from PIL import Image
import numpy as np

class Detect():
    def __init__(self, load_path):
        self.img = None

        print('Loading model..')
        self.net = DSOD(num_classes=10)
        self.net.cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.net.load_state_dict(torch.load(load_path)['net'], False)
        self.net.eval()

        self.box_coder = SSDBoxCoder(self.net.module)

    def caffe_normalize(self, x):

        return transforms.Compose([
            transforms.Lambda(lambda x: 255 * x[[2, 1, 0]]),
            transforms.Normalize([104, 117, 123], (1, 1, 1)),  # make it the same as caffe
            # bgr and 0-255
        ])(x)


    def transform(self, img, boxes):
        img, boxes = resize(img, boxes, size=(opt.img_size, opt.img_size))
        img = transforms.Compose([
            transforms.ToTensor(),
            self.caffe_normalize
        ])(img)
        return img, boxes

    def py_cpu_nms(self, dets, score, thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]


        areas = (y2 - y1 + 1) * (x2 - x1 + 1)

        scores = score
        keep = []


        index = scores.argsort()[::-1]

        while index.size > 0:
            i = index[0]  # every time the first is the biggst, and add it directly
            keep.append(i)

            # calculate the points of overlap
            x11 = np.maximum(x1[i], x1[index[1:]])
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])

            w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
            h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

            overlaps = w * h

            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

            idx = np.where(ious <= thresh)[0]
            # print(idx)
            index = index[idx + 1]  # because index start from 1

        return keep

    def detect(self,img):
        print('Processing img..')
        self.img = Image.open(img)
        if self.img.mode != 'RGB':
            self.img = self.img.convert('RGB')

        boxes = None
        inputs, boxes = self.transform(self.img, boxes)
        inputs = inputs.unsqueeze(0)
        with torch.no_grad():
            loc_preds, cls_preds = self.net(Variable(inputs.cuda()))
        box_preds, label_preds, score_preds = self.box_coder.decode(
            loc_preds.cpu().data.squeeze(),
            F.softmax(cls_preds.squeeze(), dim=1).cpu().data,
            score_thresh=0.1)

        score = score_preds.numpy()
        keep = self.py_cpu_nms(box_preds.numpy(), score, thresh=0.3)
        box_preds = box_preds.numpy()[keep]
        score_preds = score_preds.numpy()[keep]
        label_preds = label_preds.numpy()[keep]

        sw = float(self.img.size[0]) / float(opt.img_size)
        sh = float(self.img.size[1]) / float(opt.img_size)
        boxes = box_preds * np.array([sw, sh, sw, sh])
        index = np.argmax(score_preds)

        x1 = int(boxes[index][0])
        x2 = int(boxes[index][2])
        y1 = int(boxes[index][1])
        y2 = int(boxes[index][3])
        if x1<0: x1=0
        if y1 < 0: y1 = 0
        if x2 < 0: x2 = 0
        if y2 < 0: y2 = 0

        if x1>=x2 or y1>=y2:
            print('no detection')
        else:
            return x1,y1,x2,y2

load_path = 'checkpoint/45-loss=0.6232420088642318.pth'
img = '/home/yxq/rock/123.JPG'
detectmodel = Detect(load_path)
x1,y1,x2,y2 = detectmodel.detect(img)
print('coordinate：',[x1,y1,x2,y2])


