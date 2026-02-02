try:
    from paddleocr import PaddleOCR
    print("Attempting to initialize PaddleOCR with enable_mkldnn=False...")
    # Try initializing with the argument
    ocr = PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False)
    print("Success: PaddleOCR initialized.")
except TypeError as e:
    print(f"TypeError: {e}")
except Exception as e:
    print(f"Error: {e}")
