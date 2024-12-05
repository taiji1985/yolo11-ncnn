# 
#
#   Give me a star please : 
#   https://github.com/taiji1985/yolo11-ncnn
#
#   @author taiji1985
#
#

import ncnn as pyncnn
import yaml
import os
import sys
import numpy as np
import torch.nn as nn
import torch
import cv2

class NCNNRunner:
    def __init__(self, model_path,use_gpu=False):
        self.model_path = model_path
        self.net = pyncnn.Net()
        self.net.opt.use_vulkan_compute = use_gpu
        param_path = os.path.join(model_path,"model.ncnn.param")
        # check if param file exists
        if not os.path.exists(param_path):
            print("param file not found")
            sys.exit(1)
        self.net.load_param(param_path)
        model_bin = os.path.join(model_path,"model.ncnn.bin")
        # check if bin file exists
        if not os.path.exists(model_bin):
            print("bin file not found")
            sys.exit(1)
        self.net.load_model(model_bin)
        self.metadata =   os.path.join(model_path,"metadata.yaml")
        # check if metadata file exists
        if not os.path.exists(self.metadata):
            print("metadata file not found")
            sys.exit(1)
        with open(self.metadata, 'r') as f:
            data = yaml.safe_load(f)
        self.model_input_shape = data["imgsz"]
        self.class_names = data["names"]
    
    #   pre_transform
    def pre_transform(self,img):
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = self.model_input_shape  # set this to the size you want
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        # if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
        #     r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding


        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)   #    new_unpad=(312,640)
        top, bottom = int(round(dh - 0.1)) , int(round(dh + 0.1)) # 0,0
        left, right = int(round(dw - 0.1)) , int(round(dw + 0.1)) # 164,164
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        ) 
        self.dw = dw
        self.dh = dh
        self.ratio = ratio

        return img  #640 640 3 
    def preprocess(self,im):
        im2 = np.stack([self.pre_transform(im)])  # (h, w, c) to (1, h, w, c)，1 ，640 ，640 ，3 

        im2 = im2[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)   # 1,3,640,640
        im2 = np.ascontiguousarray(im2)  # contiguous
        im2 = torch.from_numpy(im2)
        im2 = im2.half() if False else im2.float()
        if True:
            im2 /= 255  # 0 - 255 to 0.0 - 1.0
        return im2

    def predict(self,img2):
        b, ch, h, w = img2.shape  # batch, channel, height, width   1 3 640 640
        mat_in = pyncnn.Mat(img2[0].cpu().numpy())  # im[0].shape  3,640,640
        with self.net.create_extractor() as ex:
            ex.input(self.net.input_names()[0], mat_in) # self.net.input_names()[0] == in0
            # WARNING: 'output_names' sorted as a temporary fix for https://github.com/pnnx/pnnx/issues/130
            y = [np.array(ex.extract(x)[1])[None] for x in sorted(self.net.output_names())]  # out0

        
        return y

    def postprocess(self,input_image, output,confidence_thres,iou_thres):
        print(f"Postprocessing shape: {len(output)} {output[0].shape}")

        # 转置并压缩输出，以匹配预期形状
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []

        input_image_height, input_image_width = input_image.shape[:2]
        model_input_height, model_input_width = self.model_input_shape
        # 计算缩放比例和填充
        ratio = (input_image_width / model_input_width, input_image_height / model_input_height)
        np.set_printoptions(suppress=True)
        for i in range(rows):
            #print(outputs[i])
            #print("{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(outputs[i]))
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                
                left = (x - w / 2)
                top = (y - h / 2)
                width = (w)
                height = (h)

                boxes.append([left, top, width, height])
                scores.append(max_score)
                class_ids.append(class_id)
        
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)
        #print(indices,type(indices))
        boxes = np.array(boxes)
        scores = np.array(scores)
        class_ids = np.array(class_ids)

        for b in boxes:
            # 移除填充
            b[0] -= self.dw
            b[1] -= self.dh        
        
        box_got = boxes[indices]*np.max(ratio)
        score_got = scores[indices]
        class_id_got = class_ids[indices]
        return box_got, score_got, class_id_got
    def show(self, input_image, r):
        # 绘图
        dimg = input_image.copy()
        box_got, score_got, class_id_got = r
        for i in range(len(box_got)):
            box = box_got[i]
            score = score_got[i]
            class_id = class_id_got[i]
            cv2.rectangle(dimg, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), (0, 255, 0), 2)
            cv2.putText(dimg, f"{self.class_names[class_id]} {score:.2f}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        cv2.imwrite("result.png", dimg)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,10))
        dimgv = cv2.cvtColor(dimg, cv2.COLOR_BGR2RGB)
        plt.imshow(dimgv)
        plt.show()
        pass
    def run(self, input_image,show=True,confidence_thres=0.5,iou_thres=0.45):
        im2 =self.preprocess(input_image)
        y=self.predict(im2)
        r = self.postprocess(input_image,y, confidence_thres, iou_thres)
        # 打印r
        box_got, score_got, class_id_got = r
        print(f"box_got: {box_got}")
        print(f"score_got: {score_got}")
        print(f"class_id_got: {class_id_got}")
        if show:
            self.show(input_image,r)
        return r

if __name__ == '__main__':
    fileName = "bus.jpg"
    im = cv2.imread(fileName)
    if im is None:
        print("cv2.imread failed")
        exit(1)
    
    # if args has model path ,use it 
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print("use model path:", model_path)
    else:
        model_path = "yolo11n_ncnn_model"
    if len(sys.argv) > 2:
        fileName  = sys.argv[2]


    m = NCNNRunner(model_path)
    m.run(im)
    print("finish")


