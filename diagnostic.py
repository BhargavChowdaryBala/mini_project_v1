import sys
print(f"Python: {sys.version}")

try:
    print("Importing ultralytics...")
    from ultralytics import YOLO
    print("Success.")
except Exception as e:
    print(f"FAILED: {e}")

try:
    print("Importing cv2...")
    import cv2
    print("Success.")
except Exception as e:
    print(f"FAILED: {e}")

try:
    print("Importing paddleocr...")
    from paddleocr import PaddleOCR
    print("Success.")
    print("Initializing PaddleOCR...")
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    print("PaddleOCR Initialized.")
except Exception as e:
    print(f"FAILED: {e}")

try:
    print("Importing streamlit...")
    import streamlit
    print("Success.")
except Exception as e:
    print(f"FAILED: {e}")
