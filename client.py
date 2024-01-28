#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys,os
import numpy as np
import cv2

###################################
#沿用newXrayDetector接口不变，其内容不修改
from newXrayDetector.ttypes import imgWithId
from newXrayDetector import Xray_Detector #引入客户端类
###################################

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol,TCompactProtocol

#transport = TSocket.TSocket('10.15.90.61', 6668)
transport = TSocket.TSocket('127.0.0.1', 6668)
#选择传输层，这块要和服务端的设置一致
transport = TTransport.TBufferedTransport(transport)
#选择传输协议，这个也要和服务端保持一致，否则无法通信
protocol = TBinaryProtocol.TBinaryProtocol(transport)

#创建客户端
client = Xray_Detector.Client(protocol)
transport.open()

#连接可以重复使用，在实际应用时需要考虑发生异常重新建立连接

# img_folder = '/data2/chen/uad-tire/3-常用规格整理/12R225-18PR-AZ189-金冠无内#1614/defect'
# files = os.listdir(img_folder)
# for file in files:
#     path = os.path.join(img_folder, file)
#     if cv2.imread(path).shape[1] <= 2000:
#         specid = '12R225-18PR-AZ189-1614'
#     else:
#         specid = '12R225-18PR-AZ189-3456'
#     id = f'{os.path.splitext(file)[0]}.{specid}'  ###  文件名.规格名  ！！！
#     bytes=[]
#     with open(path,'rb') as fp:
#         bytes = fp.read()
#         img_data = imgWithId()
#         img_data.img = bytes
#         img_data.id = id
#         retjson = client.service_detector(img_data)
#         print(retjson)

import json

path = '/data2/chen/uad-tire/3-常用规格整理/650R16-12PR-CR926-朝阳#1614/defect/E3L2B20282.jpg'
id = f'C3D1D21173.650R16-12PR-CR926-1614'  ###  文件名.规格名  ！！！
bytes=[]
with open(path,'rb') as fp:
    bytes = fp.read()
    img_data = imgWithId()
    img_data.img = bytes
    img_data.id = id
    retjson = client.service_detector(img_data)
    retjson = json.loads(retjson)
    print(retjson)
    img = cv2.imread(path)
    for result in retjson["Result"]:
        PointS = result['PointS']
        PointE = result['PointE']
        cv2.rectangle(img, (int(PointS[0]), int(PointS[1])), (int(PointE[0]), int(PointE[1])), (0, 255, 0), 1)  # 这里的(0, 255, 0)表示矩形框的颜色，参数2表示线条宽度
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f'{result["RateAdd"]}, {result["Rate"]}', (int(PointS[0]), int(PointS[1])), font, 0.9, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite('result.png', img)
transport.close()
