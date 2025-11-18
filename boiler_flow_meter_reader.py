import cv2
import numpy as np
from paddleocr import PaddleOCR
import json

# 初始化 OCR（英文模式）
ocr = PaddleOCR(lang="en", use_angle_cls=True)


# ---------------------------------------
# 功能1：图像预处理
# ---------------------------------------
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur


# ---------------------------------------
# 功能2：自动检测圆形仪表区域（用霍夫圆）
# ---------------------------------------
def detect_circle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=200,
        param1=70,
        param2=40,
        minRadius=100,
        maxRadius=300,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        return x, y, r
    return None


# ---------------------------------------
# 功能3：识别数字、文字（OCR）
# ---------------------------------------
def ocr_text(img):
    result = ocr.ocr(img)
    texts = [line[1][0] for line in result]
    return texts


# ---------------------------------------
# 功能4：识别 ON / OFF 开关
# ---------------------------------------
def detect_switch(img):
    # 假设开关区域是黑底白条
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    # 左右亮度比较
    h, w = th.shape
    left = np.mean(th[:, : w // 2])
    right = np.mean(th[:, w // 2 :])

    if right > left:
        return "ON"
    else:
        return "OFF"


# ---------------------------------------
# 功能5：检测指示灯亮灭
# ---------------------------------------
def detect_led(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])
    return "ON" if brightness > 150 else "OFF"


# ---------------------------------------
# 功能6：主流程：读取摄像头，自动识别
# ---------------------------------------
def read_meter():

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("摄像头无法打开")
        return

    print("开始读取摄像头... 按 Ctrl+C 结束")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        img = preprocess(frame)

        # 自动检测圆形仪表
        circle = detect_circle(frame)
        if circle is None:
            cv2.imshow("meter", frame)
            cv2.waitKey(1)
            continue

        x, y, r = circle
        meter_roi = frame[y - r : y + r, x - r : x + r]

        # 定义 ROI（根据仪表结构自定义）
        # ------------------------------
        h, w, _ = meter_roi.shape

        # 数字显示区域
        number_roi = meter_roi[int(h * 0.30) : int(h * 0.55), int(w * 0.15) : int(w * 0.85)]

        # 开关区域
        switch_roi = meter_roi[int(h * 0.60) : int(h * 0.80), int(w * 0.65) : int(w * 0.90)]

        # 指示灯区域
        led_roi = meter_roi[int(h * 0.20) : int(h * 0.35), int(w * 0.60) : int(w * 0.80)]

        # 端子文字区域
        terminal_roi = meter_roi[int(h * 0.55) : int(h * 0.75), int(w * 0.10) : int(w * 0.50)]

        # OCR识别
        numbers = ocr_text(number_roi)
        terminals = ocr_text(terminal_roi)

        # 状态识别
        switch_state = detect_switch(switch_roi)
        led_state = detect_led(led_roi)

        # 打包 JSON 输出
        output = {
            "flow_value": numbers,
            "switch": switch_state,
            "led": led_state,
            "terminals": terminals,
        }

        print(json.dumps(output, indent=4, ensure_ascii=False))

        # 展示界面（调试用）
        cv2.imshow("meter_roi", meter_roi)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# 启动系统
read_meter()
