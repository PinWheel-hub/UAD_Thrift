#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys,os
import numpy as np

###################################
#沿用newXrayDetector接口不变，其内容不修改
from newXrayDetector.ttypes import imgWithId
from newXrayDetector import Xray_Detector #引入客户端类
###################################

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol,TCompactProtocol

#transport = TSocket.TSocket('10.15.90.61', 6668)
transport = TSocket.TSocket('10.15.90.61', 3022)
#选择传输层，这块要和服务端的设置一致
transport = TTransport.TBufferedTransport(transport)
#选择传输协议，这个也要和服务端保持一致，否则无法通信
protocol = TBinaryProtocol.TBinaryProtocol(transport)

#创建客户端
client = Xray_Detector.Client(protocol)
transport.open()

#连接可以重复使用，在实际应用时需要考虑发生异常重新建立连接
while True:
    path = 'D3L2C22112.png'
    specid='D3L2C22112.xxxxxxx-yyyyyy-zzzz'  ###  文件名.规格名  ！！！
    bytes=[]
    with open(path,'rb') as fp:
        bytes = fp.read()
        img_data = imgWithId()
        img_data.img = bytes
        img_data.id = specid
        retjson = client.service_detector(img_data)
        print(retjson)

transport.close()
