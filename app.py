"""
Streamlit front-end for the Computer-Aided Diagnosis Assistant.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from openai import OpenAI

from device import DEVICE
from heatmap import generate_heatmap_overlay
from metadata import extract_mask_metadata
from model import build_unet, predict

IMG_SIZE = 256
HOLDOUT_DIR = Path("data/holdout")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

@st.cache_resource
def load_model(weights_path: str | None = None) -> torch.nn.Module:
    """Build the U-Net and optionally load trained weights."""
    model = build_unet()
    if weights_path:
        state = torch.load(weights_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(path: str | Path) -> torch.Tensor:
    """Load an image from disk and return a ``(3, 256, 256)`` tensor in [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def build_llm_prompt(meta: dict) -> str:
    """Format extracted metadata into a structured prompt for the LLM."""
    lines = [
        "You are a board-certified radiologist reviewing an ultrasound image.",
        "Based ONLY on the following quantitative segmentation metadata, "
        "write exactly 3 professional sentences summarizing the findings.\n",
        f"- Lesion present: {meta['lesion_present']}",
        f"- Area ratio (lesion / total): {meta['area_ratio']:.4%}",
        f"- Centroid (x, y): {meta['centroid']}",
        f"- Bounding box: {meta['bounding_box']}",
    ]
    return "\n".join(lines)


def generate_report(meta: dict, api_key: str) -> str:
    """Call the OpenAI API to produce a short radiology summary."""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a concise radiology report generator. "
                    "Only use the metadata provided. Do not invent findings."
                ),
            },
            {"role": "user", "content": build_llm_prompt(meta)},
        ],
        temperature=0.3,
        max_tokens=256,
    )
    return response.choices[0].message.content


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="CAD Ultrasound Assistant",
        page_icon="🩺",
        layout="wide",
    )

    st.title("Computer-Aided Diagnosis Assistant")
    st.caption("Select a holdout ultrasound image to generate a segmentation heatmap and AI-drafted radiology report.")

    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("OpenAI API Key", type="password")
        weights_path = st.text_input(
            "Model weights path (optional)",
            placeholder="e.g. checkpoints/best.pth",
        )
        threshold = st.slider("Mask threshold", 0.0, 1.0, 0.5, 0.05)

    holdout_images = sorted(HOLDOUT_DIR.glob("*.png"))
    if not holdout_images:
        st.error(f"No images found in `{HOLDOUT_DIR}`.")
        return

    display_names = [p.name for p in holdout_images]
    selected_name = st.selectbox("Select a holdout image", display_names)
    selected_path = HOLDOUT_DIR / selected_name

    image_tensor = preprocess_image(selected_path)

    model = load_model(weights_path if weights_path else None)
    prob_map, binary_mask = predict(model, image_tensor, threshold=threshold)

    overlay = generate_heatmap_overlay(prob_map, image_tensor, alpha=0.4)
    meta = extract_mask_metadata(binary_mask)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        original_np = (
            image_tensor.permute(1, 2, 0).numpy() * 255
        ).astype(np.uint8)
        st.image(original_np, use_container_width=True)

    with col2:
        st.subheader("Heatmap Overlay")
        st.image(overlay, use_container_width=True)

    st.divider()

    st.subheader("Extracted Metadata")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Area Ratio", f"{meta['area_ratio']:.4%}")
    mc2.metric("Centroid", str(meta["centroid"]) if meta["centroid"] else "N/A")
    mc3.metric(
        "Bounding Box",
        f"{meta['bounding_box']['width']}×{meta['bounding_box']['height']}"
        if meta["bounding_box"]
        else "N/A",
    )

    st.divider()

    st.subheader("AI Radiology Report")
    if not api_key:
        st.warning("Enter your OpenAI API key in the sidebar to generate a report.")
    elif st.button("Generate Report"):
        with st.spinner("Generating clinical summary…"):
            report = generate_report(meta, api_key)
        st.success(report)


main()
