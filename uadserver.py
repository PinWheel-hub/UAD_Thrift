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

from uadetect import init_uadetect_func, uadetect_func, load_func, del_func
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
        global procFlag, pids, model_num
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
                    if model_num >= cfg["model_memory_num"]:
                        delete = pool.apply_async(del_func)
                        deleteSide = poolSide.apply_async(del_func)
                        del_spec = delete.get(timeout=2)
                        deleteSide.wait(timeout=2)
                        model_num -= 1
                        print(del_spec)
                        with open(f'xuadetect/loadlist/{del_spec}', "w") as f:
                            pass
                    load = pool.apply_async(load_func, (spec_name, False))
                    loadSide = poolSide.apply_async(load_func, (spec_name, True))
                    load.wait(timeout=2)
                    loadSide.wait(timeout=2)
                    model_num += 1
                    
                    if os.path.exists(f'xuadetect/loadlist/{spec_name}'):
                        os.remove(f'xuadetect/loadlist/{spec_name}')
                elif not os.path.exists(f'xuadetect/models/{spec_name}.pth'):
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
                part_locations_side = []
                num = int(math.ceil(1.0 * (image_h - dh) / (hh - dh)))
                for l in range(num):
                    for c in range(col_num):
                        x1 = ww * c
                        y1 = min(l * (hh - dh), image_h - hh)
                        if c == 0:
                            part_images_side.append(img[y1:y1 + hh, x1:x1 + ww, :])
                            part_locations_side.append((y1 * ratioxy, x1 * ratioxy))
                        elif c == col_num - 1:
                            part_images_side.append(cv2.flip(img[y1:y1 + hh, x1:x1 + ww, :], 1))
                            part_locations_side.append((y1 * ratioxy, x1 * ratioxy))
                        else:
                            part_images.append(img[y1:y1 + hh, x1:x1 + ww, :])
                            part_locations.append((y1 * ratioxy, x1 * ratioxy))
                        

            ###本服务会发生多线程重入，如果线程之间需要排队处理，请加锁
            ###也就是说会有可能两张不同规格图片同时运行
            with lockMain:
                ####对裁切小片识别处理
                retlist=[]
                retlistSide=[]
                for i, part in enumerate(part_images):
                    retlist.append(pool.apply_async(uadetect_func, (part, spec_name, part_locations[i])))
                for i, part in enumerate(part_images_side):
                    retlistSide.append(poolSide.apply_async(uadetect_func, (part, spec_name, part_locations_side[i])))

            ##等待识别阶段，务必先解锁
            # time.sleep(1)

            for ritem in retlist:
                waitresult = ritem.get(timeout=3)
                ###返回结果处理，仅供参考
                ###返回需要指示位置坐标
                (score, location, fault_flag) = waitresult
                ###################################
                # 每个flaw记录，需要FaultID，Rate概率，左上坐标(x,y),右下坐标(x,y)
                if fault_flag:
                    flaw = {}
                    flaw['FaultID'] = '85'
                    flaw['Rate'] = score
                    flaw['PointS'] = location
                    flaw['PointE'] = (location[0] + ww, location[1] + ww)
                    result.append(flaw)

            for ritem in retlistSide:
                waitresult = ritem.get(timeout=3)
                ###返回结果处理，仅供参考
                ###返回需要指示位置坐标
                (score, location, fault_flag) = waitresult
                ###################################
                # 每个flaw记录，需要FaultID，Rate概率，左上坐标(x,y),右下坐标(x,y)
                if fault_flag:
                    flaw = {}
                    flaw['FaultID'] = '86'
                    flaw['Rate'] = score
                    flaw['PointS'] = location
                    flaw['PointE'] = (location[0] + ww, location[1] + ww)
                    result.append(flaw)

            ####返回json格式示例#####
            img_data = {}
            img_data['ID'] = img_name
            img_data['StatusCode'] = 0

            img_data['Result'] = result
            img_data['Note'] = f'Image size: {r_h} x {r_w}'
            img_data['DetectTime'] = f'{time.time() - t0:.2f}'

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
                    if model_num >= cfg["model_memory_num"]:
                        delete = pool.apply_async(del_func)
                        deleteSide = poolSide.apply_async(del_func)
                        del_spec = delete.get(timeout=2)
                        deleteSide.wait(timeout=2)
                        model_num -= 1
                        print(del_spec)
                        with open(f'xuadetect/loadlist/{del_spec}', "w") as f:
                            pass
                    load = pool.apply_async(load_func, (spec_name, False))
                    loadSide = poolSide.apply_async(load_func, (spec_name, True))
                    load.wait(timeout=2)
                    loadSide.wait(timeout=2)
                    model_num += 1
                    if os.path.exists(f'xuadetect/loadlist/{spec_name}'):
                        os.remove(f'xuadetect/loadlist/{spec_name}')


            return img_data_json


####异步进程，训练模型
def spawnBackGroundWorker(spec_name):
    global pids, cfg
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(target=TrainingProc, args=(spec_name, cfg))
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
    debugMode = True
    model_num = 0

    ##多线程锁
    lockMain = threading.Lock()
    pids = []

    ####spawn task !!!###
    ctx = multiprocessing.get_context('spawn')
    pool = ctx.Pool(1, init_uadetect_func, (False,))
    poolSide = ctx.Pool(1, init_uadetect_func, (True,))

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
    with open(f'xuadetect/loadlist/12R225-18PR-AZ189-1614', "w") as f1, open(f'xuadetect/loadlist/700R16-8PR-EZ525-1614', "w") as f2, open(f'xuadetect/loadlist/xxxxxxx-yyyyyy-2000', "w") as f3:
        pass
    print("Starting main thrift server for UADetect...")
    server.serve()
    print("UADetect thrift server ends.")
