import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.title("🎨 Image to Sketch & Art Filter App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Original Image", use_column_width=True)

    option = st.selectbox(
        "Choose a filter",
        ("Pencil Sketch", "Sepia", "Cartoon", "Pastel")
    )

    def pencil_sketch(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256.0)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    def sepia(img):
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        return cv2.transform(img, kernel)

    def cartoon(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img, 9, 300, 300)
        return cv2.bitwise_and(color, color, mask=edges)

    def pastel(img):
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        return cv2.convertScaleAbs(blur, alpha=1.2, beta=20)

    filters = {
        "Pencil Sketch": pencil_sketch,
        "Sepia": sepia,
        "Cartoon": cartoon,
        "Pastel": pastel
    }

    result = filters[option](cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    st.image(result_rgb, caption=f"{option} Effect", use_column_width=True)

    # Convert to bytes for download
    buf = io.BytesIO()
    result_pil = Image.fromarray(result_rgb)
    result_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Result",
        data=byte_im,
        file_name=f"{option.lower().replace(' ', '_')}.png",
        mime="image/png"
    )
