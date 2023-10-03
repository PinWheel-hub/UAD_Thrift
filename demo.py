import multiprocessing
from uatraining import TrainingProc
import time

# 定义一个函数，将在进程中运行
def my_function():
    while True:
        print("进程正在运行")
        time.sleep(2)

if __name__ == "__main__":
    ctx = multiprocessing.get_context('spawn')
    p = ctx.Process(target=TrainingProc, args=('1'))
    # 启动进程
    p.start()

    while True:
        print(p.is_alive())

    # 等待进程完成（可选）
    p.join()