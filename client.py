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

img_folder = '/data2/chen/uad-tire/2-常用规格/6.50R16-12PR[CR926]朝阳/合格'
files = os.listdir(img_folder)
for file in files:
    path = os.path.join(img_folder, file)
    if cv2.imread(path).shape[1] <= 2000:
        specid = 'xxxxxxx-yyyyyy-2000'
    else:
        specid = 'xxxxxxx-yyyyyy-4000'
    id = f'{os.path.splitext(file)[0]}.{specid}'  ###  文件名.规格名  ！！！
    bytes=[]
    with open(path,'rb') as fp:
        bytes = fp.read()
        img_data = imgWithId()
        img_data.img = bytes
        img_data.id = id
        retjson = client.service_detector(img_data)
        print(retjson)

transport.close()
