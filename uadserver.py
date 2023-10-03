#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, signal, math
import socket, hashlib, uuid  # fcntl,struct
import numpy as np
import json, glob
import subprocess
import time, cv2
import threading, multiprocessing
import logging
import traceback

###################################
# 沿用newXrayDetector接口不变，其内容不修改
from newXrayDetector import Xray_Detector
from newXrayDetector.ttypes import imgWithId
###################################

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from io import BytesIO

from uadetect import init_uadetect_func, uadetect_func, init_uadetectSide_func, uadetectSide_func
from uatraining import TrainingProc

#规范使用logging utility
class GlobLogger:
    def __init__(self):
        self._logN = logging.getLogger("normal_log")
        self._logF = logging.getLogger("defect_log")
        self._logE = logging.getLogger("error_log")
        self._date = ''

    def chkdate(self):
        today = time.strftime("%Y-%m-%d", time.localtime(time.time()))
        if self._date == today:
            return

        self._date = today
        path = 'logs/' + today + '/'
        if not os.path.exists(path):
            os.makedirs(path)

        formatter = logging.Formatter('%(asctime)s : %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        if self._logN.handlers:
            self._logN.removeHandler(self._logN.handlers[0])
        handler = logging.FileHandler(path + "normal.log")
        handler.setFormatter(formatter)
        self._logN.addHandler(handler)
        self._logN.setLevel(logging.INFO)

        if self._logF.handlers:
            self._logF.removeHandler(self._logF.handlers[0])
        handler = logging.FileHandler(path + "defect.log")
        handler.setFormatter(formatter)
        self._logF.addHandler(handler)
        self._logF.setLevel(logging.INFO)

        if self._logE.handlers:
            self._logE.removeHandler(self._logE.handlers[0])
        handler = logging.FileHandler(path + "error.log")
        handler.setFormatter(formatter)
        self._logE.addHandler(handler)
        self._logE.setLevel(logging.INFO)

    def logN(self, msg):
        self.chkdate()
        if self._logN is not None:
            self._logN.info(msg)

    def logF(self, msg):
        self.chkdate()
        if self._logF is not None:
            self._logF.info(msg)

    def logE(self, msg):
        self.chkdate()
        if self._logE is not None:
            self._logE.info(msg)

###服务回调总入口
class XrayDetectorHandler:
    #暂时dummy，不用
    def get_modelver(self):
        global modelver
        return modelver

    #暂时dummy，不用
    def get_modellist(self):
        return ''

    #暂时dummy，不用
    def change_model(self, modelnew):
        """
        Parameters:
         - modelver
        """
        return ''

    #暂时dummy，不用
    def upload_model(self, data):
        """
        Parameters:
         - data
        """
        return ''

    ###########################################
    #目前唯一的服务入口
    ###########################################
    def service_detector(self, img_data):
        global procFlag, pids
        statuscode = 500
        try:
            result = []
            t0 = time.time()
            img_bytes = img_data.img
            img_name = img_data.id
            ##解析文件名与规格名，文件名需要用于文件保存
            tmp = img_name.split('.', 1)
            img_name = tmp[0]
            spec_name = tmp[1] if len(tmp) > 1 else ""

            ###图片解析代码，沿用即可
            # imgBig = np.asarray(bytearray((BytesIO(img_bytes).read())), dtype=np.uint8)
            imgBig = np.frombuffer(img_bytes, np.uint8)
            imgBig = cv2.imdecode(imgBig, cv2.IMREAD_COLOR)
            if len(imgBig.shape) != 3:
                raise Exception("Invalid image shape: %s %d" % (img_name, len(imgBig.shape)))
            r_h, r_w, _ = imgBig.shape

            ###本服务会发生多线程重入，如果线程之间需要排队处理，请加锁
            ###也就是说会有可能两张不同规格图片同时运行
            with lockMain:
                procFlag = len(pids) > 0
                ###各总检查，模型是否存在
                if os.path.exists(f'xuadetect/loadlist/{spec_name}'):
                    statuscode = 250
                    raise Exception("Models not loaded.")
                if not os.path.exists(f'xuadetect/models/{spec_name}.pth'):
                    if not os.path.exists(f'xuadetect/img_raw/{spec_name}'):
                        os.makedirs(f'xuadetect/img_raw/{spec_name}')
                    if os.path.exists(f'xuadetect/trainlist/{spec_name}'):
                        if not procFlag:
                            procFlag = True
                            spawnBackGroundWorker(spec_name)  ##此处一定需要线程锁！！
                        statuscode = 200
                        raise Exception("Model training.")
                    cv2.imwrite(f'xuadetect/img_raw/{spec_name}/{img_name}.jpg', imgBig)
                    img_num = len(os.listdir(f'xuadetect/img_raw/{spec_name}'))
                    if img_num >= cfg['img_raw_num']:
                        with open(f'xuadetect/trainlist/{spec_name}', "w") as f:
                            pass
                        if not procFlag:
                            procFlag = True
                            spawnBackGroundWorker(spec_name)  ##此处一定需要线程锁！！
                        statuscode = 200
                        raise Exception("Model training.")
                    else:
                        statuscode = img_num
                        raise Exception("No models.")

                ##缩小与裁切处理
                ratioxy = 2
                img = cv2.resize(imgBig, (int(r_w // ratioxy), int(r_h // ratioxy)), interpolation=cv2.INTER_CUBIC)
                image_h, image_w, _ = img.shape
                # 。。。。。。。
                col_num = 3
                ww = image_w // col_num
                hh = ww
                dh = int(0 * image_h)
                part_images = []
                part_images_side = []
                part_locations = []
                num = int(math.ceil(1.0 * (image_h - dh) / (hh - dh)))
                for l in range(num):
                    for c in range(col_num):
                        x1 = ww * c
                        y1 = min(l * (hh - dh), image_h - hh)
                        part_locations.append((x1, y1))
                        if c == 0:
                            part_images_side.append(img[y1:y1 + hh, x1:x1 + ww, :])
                        elif c == col_num - 1:
                            part_images_side.append(cv2.flip(img[y1:y1 + hh, x1:x1 + ww, :], 1))
                        else:
                            part_images.append(img[y1:y1 + hh, x1:x1 + ww, :])

            ###本服务会发生多线程重入，如果线程之间需要排队处理，请加锁
            ###也就是说会有可能两张不同规格图片同时运行
            with lockMain:
                ####对裁切小片识别处理
                retlist=[]
                retlistSide=[]
                for part in part_images:
                    retlist.append(pool.map_async(uadetect_func(part)))
                for part in part_images_side:
                    retlistSide.append(pool.map_async(uadetectSide_func(part)))

            ##等待识别阶段，务必先解锁
            time.sleep(1)

            for ritem in retlist:
                waitresult = ritem.get(timeout=3)

                ###返回结果处理，仅供参考
                ###返回需要指示位置坐标
                for fitem in waitresult:
                    (img_name2, part_ret, _) = fitem
                    partjs = json.loads(part_ret)
                    part_result = partjs['Result']
                    ###################################
                    #每个flaw记录，需要FaultID，Rate概率，左上坐标(x,y),右下坐标(x,y)
                    #flaw['FaultID'] = 'XX'
                    #flaw['Rate'] = '60'
                    #flaw['PointS'] = [str(200), str(200)]
                    #flaw['PointE'] = [str(r_w - 200), str(600)]
                    for flaw in part_result:
                        result.append(flaw)

            ####返回json格式示例#####
            img_data = {}
            img_data['ID'] = img_name
            img_data['StatusCode'] = 0

            img_data['Result'] = result
            img_data['Note'] = ''
            img_data_json = json.dumps(img_data)

            ###logging示例###
            if not result:
                gLogger.logN(img_data)
            else:
                gLogger.logF(img_data)

        except Exception as e:
            # print 'error at time: ', time.ctime()
            # print("service_detector error in processing: %s Err: %s" % (img_name, str(e)))
            traceback.print_exc()
            # time.sleep(3)
            img_data = {}
            img_data["ID"] = img_name
            img_data['StatusCode'] = statuscode
            img_data["Result"] = result
            img_data["Note"] = 'Error: ' + str(e)

            img_data_json = json.dumps(img_data)
            gLogger.logE(img_data)

        finally:
            if debugMode: print('%s Total time for : %.2f' % (img_name, time.time() - t0))
            for p in pids:
                if not p.is_alive():
                    p.join(1)
                    pids.remove(p)

            return img_data_json


####异步进程，训练模型
def spawnBackGroundWorker(spec_name):
    global pids
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(target=TrainingProc, args=(spec_name,))
    p.start()
    pids.append(p)
    return pids

###主程序退出时，需要把附带的进程一同关闭！！！
def sigterm(num1, num2):
    '''
    global procFlag, termFlag
    if (not termFlag) and procFlag:
        termFlag = True
        return
    '''
    global pids     ###这里记录异步启动的训练进程####
    print("To kill all Workers and Terminate...")
    pool.terminate()
    poolSide.terminate()

    ###让训练继续直至完成？？
    # for p in pids:
    #     if p.is_alive():
    #         os.kill(p.pid, signal.SIGHUP)
    # for p in pids:
    #     p.join(2)
    #
    # for p in pids:
    #     if p.is_alive():
    #         p.terminate()
    os._exit(0)


#配置文件config.json解析
def ParseCfg():
    global cfgfile
    strall = ''
    if os.path.isfile(cfgfile) and os.path.getsize(cfgfile) > 0:
        with  open(cfgfile, 'r') as fr:
            strall = fr.read()
            try:
                jsconfig = json.loads(strall)
                #.......
                #.......
                #.......
                return jsconfig
            except:
                return None
    else:
        return None

#####初始化入口#######
if __name__ == '__main__':
    procFlag = False
    cfgfile = 'config.json'
    if len(sys.argv) > 1:
        cfgfile = sys.argv[1]
    cfg = ParseCfg()
    print(cfg)

    gLogger = GlobLogger()
    debugMode = False

    ##多线程锁
    lockMain = threading.Lock()
    pids = []

    ####spawn task !!!###
    ctx = multiprocessing.get_context('spawn')
    pool = ctx.Pool(1, init_uadetect_func)
    poolSide = ctx.Pool(1, init_uadetectSide_func)

    # handler signal SIGTERM  需要关掉其它相关进程
    signal.signal(signal.SIGTERM, sigterm)

    # 创建服务端
    handler = XrayDetectorHandler()
    processor = Xray_Detector.Processor(handler)
    # 监听端口
    transport = TSocket.TServerSocket('0.0.0.0', port=6668)
    # 选择传输层
    tfactory = TTransport.TBufferedTransportFactory()
    # 选择传输协议
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()
    # 创建服务端
    server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)
    # server.setNumThreads(4)
    print("Starting main thrift server for UADetect...")
    server.serve()
    print("UADetect thrift server ends.")
