import cv2
import numpy as np
def remove_white_cols(image_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 获取图像列的平均亮度
    col_means = np.mean(gray, axis=0)
    threshold = np.mean(col_means)
    mask = col_means < threshold
    # 找到第一个和最后一个非白色列的索引
    first_col = np.argmax(mask)

    # 找到最后一个 True 的位置
    last_col = len(mask) - 1 - np.argmax(np.flip(mask))

    # 从原始图像中裁剪除去白色条纹的部分
    result = image[:, first_col: last_col]

    # 保存输出图像
    cv2.imwrite(output_path, result)

# 使用示例
input = '/data2/chen/uad-tire/3-常用规格整理/650R16-12PR-CR926-朝阳#1614/defect/A3K3C21259_1.jpg'
remove_white_cols(input, 'output_image_cv2.jpg')