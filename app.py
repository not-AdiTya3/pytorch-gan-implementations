import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from models import *

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"

latent_dim = 100
img_size = 64
channels = 3
num_classes = 2

st.set_page_config(page_title="GAN-001", layout="wide")
st.title("🌻 Generative Adversarial Network ")

# CSS styling
st.markdown("""
<style>
[data-testid="stImage"] img {
    border: 3px solid #64748B;
    border-radius: 10px;
    padding: 4px;
}

/* Change button background */
div.stButton > button {
    background-color: #D04E00;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Image preprocessing for uploaded images
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class_to_idx = {"Daisy":0, "Sunflower":1}


@st.cache_resource
def load_models(model_type):

    if model_type == "Vanilla GAN":
        G, D = GAN_Generator(), GAN_Discriminator()
        G.load_state_dict(torch.load("Saved_Models/VANILLA_generator.pth", map_location=device))
        D.load_state_dict(torch.load("Saved_Models/VANILLA_discriminator.pth", map_location=device))

    elif model_type == "DCGAN":
        G, D = DCGAN_Generator(), DCGAN_Discriminator()
        G.load_state_dict(torch.load("Saved_Models/DCGAN_generator.pth", map_location=device))
        D.load_state_dict(torch.load("Saved_Models/DCGAN_discriminator.pth", map_location=device))

    else:
        G, D = CGAN_Generator(), CGAN_Discriminator()
        G.load_state_dict(torch.load("Saved_Models/CGAN_generator.pth", map_location=device))
        D.load_state_dict(torch.load("Saved_Models/CGAN_discriminator.pth", map_location=device))

    G.to(device).eval()
    D.to(device).eval()

    return G, D

tab_generate, tab_classify = st.tabs(["Generate Images", "Classify Real/Fake"])

with tab_generate:

    st.subheader("Generate Synthetic Flowers")
    col1, col2 = st.columns([1,3])
    with col1:

        model_type = st.selectbox(
            "Select GAN Type",
            ["Vanilla GAN", "DCGAN", "CGAN"],
            key="generate_model"
        )

        if model_type == "CGAN":
            flower_class = st.selectbox(
                "Flower Class",
                list(class_to_idx.keys()),
                key="flower_class"
            )
        else:
            flower_class = None

        num_images = st.slider(
            "Number of images",
            1, 8, 4,
            key="num_images"
        )

        generate = st.button("Generate Images", key="generate_btn")

    with col2:

        if generate:

            G, D = load_models(model_type)

            # Random latent vector
            z = torch.randn(num_images, latent_dim, device=device)

            if model_type == "CGAN":
                label_idx = class_to_idx[flower_class]
                labels = torch.tensor([label_idx]*num_images, device=device)

                with torch.no_grad():
                    imgs = G(z, labels)
            else:
                with torch.no_grad():
                    imgs = G(z)

            cols = st.columns(num_images)

            for i in range(num_images):

                img = imgs[i].detach().cpu()
                img = (img + 1) / 2
                img = img.permute(1,2,0).numpy()

                with cols[i]:
                    st.image(img, use_container_width=True)

                    # Evaluate using discriminator
                    if model_type == "CGAN":
                        score = D(imgs[i].unsqueeze(0),
                                  torch.tensor([label_idx], device=device))
                    else:
                        score = D(imgs[i].unsqueeze(0))

                    score = score.item()

                    if score > 0.5:
                        st.success(f"Real {score:.3f}")
                    else:
                        st.error(f"Fake {score:.3f}")

with tab_classify:

    st.subheader("Upload Image for Discriminator")
    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg","jpeg","png"],
        key="upload"
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width=300)
        tensor = transform(img).unsqueeze(0).to(device)
        model_type = st.selectbox(
            "Select GAN Type",
            ["Vanilla GAN", "DCGAN", "CGAN"],
            key="classify_model"
        )

        G, D = load_models(model_type)

        if model_type == "CGAN":
            label_name = st.selectbox(
                "Class Label",
                list(class_to_idx.keys()),
                key="class_label"
            )

            label = torch.tensor([class_to_idx[label_name]], device=device)
            score = D(tensor, label)

        else:
            score = D(tensor)

        score = score.item()
        st.write("Discriminator Score:", round(score,3))

        if score > 0.5:
            st.success("REAL IMAGE")
        else:
            st.error("FAKE IMAGE")