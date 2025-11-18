import cv2
import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang='en')

# 查找绿色液位区域
def find_green_level(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 找绿色像素的最上面位置
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) == 0:
        return None
    top = coords[:, 0].min()  # y坐标最小的点 = 顶部
    return top


# OCR 识别刻度数字
def detect_ticks(img):
    result = ocr.ocr(img)
    ticks = []
    for line in result:
        text = line[1][0]
        if text.replace('.', '').isdigit():  # 保留数字
            x1, y1 = line[0][0]
            x2, y2 = line[0][2]
            y = int((y1 + y2) / 2)
            ticks.append((y, float(text)))
    ticks.sort()
    return ticks


# 根据绿色高度计算实际水位
def compute_level(green_y, ticks):
    ys = [t[0] for t in ticks]
    vs = [t[1] for t in ticks]

    # 如果落在范围外
    if green_y <= ys[0]:
        return vs[0]
    if green_y >= ys[-1]:
        return vs[-1]

    # 区间查找
    for i in range(len(ticks)-1):
        y1, v1 = ticks[i]
        y2, v2 = ticks[i+1]
        if y1 <= green_y <= y2:
            # 线性插值
            ratio = (green_y - y1) / (y2 - y1)
            return v1 + ratio * (v2 - v1)

    return None


# 主流程
def read_level_meter(img_path):
    img = cv2.imread(img_path)

    # 1. 寻找绿色液位位置
    green_top = find_green_level(img)
    if green_top is None:
        return {"error": "no green level detected"}

    # 2. OCR 识别刻度
    ticks = detect_ticks(img)
    if len(ticks) < 2:
        return {"error": "no scale detected"}

    # 3. 计算真实水位
    level = compute_level(green_top, ticks)

    return {
        "green_top_y": int(green_top),
        "ticks": ticks,
        "water_level": level
    }


# 示例调用
result = read_level_meter("level_meter.jpg")
print(result)
