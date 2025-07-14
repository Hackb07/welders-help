import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("welding.pt")

st.set_page_config(page_title="Welding Detection App", layout="centered")
st.title("🖼️ Welding Detection App with YOLOv8")
st.markdown("Upload an image or use live webcam feed for detection.")

# -------- Image Upload Detection --------
with st.expander("📸 Upload Image for Detection"):
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Run Detection"):
            with st.spinner("Detecting..."):
                results = model(image_cv, verbose=False)
                detected = results[0].plot()
                detected_rgb = cv2.cvtColor(detected, cv2.COLOR_BGR2RGB)
                st.image(detected_rgb, caption="Detected Image", use_column_width=True)
                st.success("Detection Complete ✅")

# -------- Live Camera Detection --------
st.markdown("---")
st.subheader("🎥 Live Webcam Detection")

class YOLOVideoTransformer(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        image = frame.to_ndarray(format="bgr24")
        results = model(image, verbose=False)
        annotated_frame = results[0].plot()
        return annotated_frame

webrtc_streamer(
    key="live",
    video_processor_factory=YOLOVideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
