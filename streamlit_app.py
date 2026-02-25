import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_PATH = "tumor_resnet50.pth"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary"
}

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor MRI Classifier — Clinical View",
    layout="centered"
)

# -------------------------------------------------
# CLINICAL STYLING
# -------------------------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #eef3ff, #f9fafb);
}
.big-title {
    font-size: 34px;
    font-weight: 800;
    color: #2563eb;   /* Clinical blue */
}
.subtitle {
    font-size: 14px;
    color: #475569;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
    border: 1px solid #e2e8f0;
}
.small-note {
    font-size: 12px;
    color: #64748b;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# MODEL LOADING
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)

    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, 4)
    )

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# -------------------------------------------------
# TRANSFORMS
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------
# GRAD-CAM
# -------------------------------------------------
def generate_gradcam(model, input_tensor, target_class, orig_img):

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_fwd = model.layer4.register_forward_hook(forward_hook)
    handle_bwd = model.layer4.register_full_backward_hook(backward_hook)

    output = model(input_tensor)
    score = output[0, target_class]

    model.zero_grad()
    score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    act = activations[0].detach()[0]
    grad = gradients[0].detach()[0]

    weights = grad.mean(dim=(1, 2))
    cam = torch.zeros_like(act[0])

    for c, w in enumerate(weights):
        cam += w * act[c]

    cam = torch.relu(cam)
    cam = cam.cpu().numpy()

    cam -= cam.min()
    cam /= (cam.max() + 1e-6)

    cam_img = Image.fromarray(np.uint8(cam * 255))
    cam_img = cam_img.resize(orig_img.size, Image.BILINEAR)
    cam = np.array(cam_img) / 255.0

    heatmap = np.zeros((cam.shape[0], cam.shape[1], 3))
    heatmap[..., 0] = cam
    heatmap[..., 1] = 1 - cam
    heatmap[..., 2] = 0.3

    orig = np.array(orig_img.convert("RGB")) / 255.0
    overlay = (0.5 * heatmap + 0.5 * orig)

    overlay = np.uint8(overlay * 255)
    return Image.fromarray(overlay)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown('<div class="big-title">Brain Tumor MRI Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ResNet-50 · 4 output categories · Deep Learning Decision Support</div>', unsafe_allow_html=True)
st.markdown("---")

# Model status
status_col1, status_col2 = st.columns([1,3])
with status_col1:
    st.success("Model Ready")
with status_col2:
    st.write(f"Running on: {DEVICE}")

st.markdown("---")

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Brain MRI Scan (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.image(image, caption="Uploaded MRI Slice", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad_(True)

    with st.spinner("Running model inference..."):

        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred = torch.max(probs, dim=0)

        prediction = CLASSES[int(pred.item())]
        confidence = float(conf.item() * 100)

        heatmap = generate_gradcam(model, input_tensor, int(pred.item()), image)

    st.markdown("## Prediction Summary")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown(f"**Predicted Category:** {prediction}")
    st.markdown(f"**Model Confidence:** {confidence:.2f}%")

    st.progress(confidence / 100)

    st.markdown("""
    <div class="small-note">
    Confidence values reflect softmax probabilities and do not directly correspond
    to clinical risk. Interpret in conjunction with complete imaging and clinical context.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("## Class Probabilities")

    for i in range(len(CLASSES)):
        st.write(f"{CLASSES[i]} — {probs[i].item()*100:.2f}%")
        st.progress(float(probs[i].item()))

    st.markdown("## Model Attention (Grad-CAM)")

    show_heatmap = st.toggle("Show attention heatmap", value=True)
    opacity = st.slider("Heatmap Opacity", 0.0, 1.0, 0.5)

    if show_heatmap:
        overlay = np.array(heatmap)
        base = np.array(image.resize(heatmap.size))

        blended = (opacity * overlay + (1 - opacity) * base).astype(np.uint8)
        st.image(blended, use_container_width=True)

    st.markdown("## Interpretation Notes")
    st.info("""
    This system is a deep learning research prototype intended to support
    educational and exploratory analysis. It is not a diagnostic device
    and must not replace expert radiologic interpretation.
    """)