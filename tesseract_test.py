try:
    from PIL import Image
    import pytesseract

    # Specify Tesseract executable path
    pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

    # OCR on an image
    text = pytesseract.image_to_string(
        Image.open("C:/Users/LEGION/Pictures/Screenshots/Screenshot 2025-01-21 111240.png"),
        lang="mal",
    )
    print(text)
except Exception as e:
    print(f"An error occurred: {e}")
