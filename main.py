import streamlit as st 
import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb, rgb2gray
import io
import torch.nn as nn
from skimage.metrics import structural_similarity
import pandas as pd
import os
import re
from tqdm import tqdm
import matplotlib.pyplot as plt

#############################
# Fungsi-fungsi bantu (PENTING untuk Evaluate)
#############################
def list_images_recursively(directory):
    image_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('jpg','jpeg','png')):
                image_files.append(os.path.join(root, file))
    return image_files

def compute_ssim(img1, img2):
    """
    Menghitung SSIM antara dua gambar RGB dengan skala [0,1].
    Pastikan channel_axis=2 (skimage 0.19+) atau multichannel=True (versi lama).
    """
    return structural_similarity(img1, img2, channel_axis=2, data_range=1.0)

def compute_mae(img1, img2):
    """Menghitung mean absolute error (MAE) dua gambar RGB [0,1]."""
    return np.mean(np.abs(img1 - img2))

def compute_colorfulness(image):
    """
    Menghitung Colorfulness dengan metode Hasler & Süsstrunk.
    image: [H, W, 3], range [0,1]
    """
    R = image[..., 0]
    G = image[..., 1]
    B = image[..., 2]

    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)

    std_rg, mean_rg = np.std(rg), np.mean(rg)
    std_yb, mean_yb = np.std(yb), np.mean(yb)

    std_root = np.sqrt((std_rg ** 2) + (std_yb ** 2))
    mean_root = np.sqrt((mean_rg ** 2) + (mean_yb ** 2))
    return std_root + (0.3 * mean_root)

#############################
# Model: UNet Generator
#############################
class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        self.down1 = self._downsample(1, 64, 4, apply_batchnorm=False)
        self.down2 = self._downsample(64, 128, 4)
        self.down3 = self._downsample(128, 256, 4)
        self.down4 = self._downsample(256, 512, 4)
        self.down5 = self._downsample(512, 512, 4)
        self.down6 = self._downsample(512, 512, 4)
        self.down7 = self._downsample(512, 512, 4)
        self.down8 = self._downsample(512, 512, 4)
        
        self.up1 = self._upsample(512, 512, 4, apply_dropout=True)
        self.up2 = self._upsample(1024, 512, 4, apply_dropout=True)
        self.up3 = self._upsample(1024, 512, 4, apply_dropout=True)
        self.up4 = self._upsample(1024, 512, 4)
        self.up5 = self._upsample(1024, 256, 4)
        self.up6 = self._upsample(512, 128, 4)
        self.up7 = self._upsample(256, 64, 4)
        
        self.last = nn.ConvTranspose2d(128, 2, 4, 2, 1)
        self.tanh = nn.Tanh()
    
    def _downsample(self, in_channels, out_channels, kernel_size, apply_batchnorm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, bias=False)]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def _upsample(self, in_channels, out_channels, kernel_size, apply_dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if apply_dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        u1 = self.up1(d8)
        u1 = torch.cat([u1, d7], dim=1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d6], dim=1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d5], dim=1)
        u4 = self.up4(u3)
        u4 = torch.cat([u4, d4], dim=1)
        u5 = self.up5(u4)
        u5 = torch.cat([u5, d3], dim=1)
        u6 = self.up6(u5)
        u6 = torch.cat([u6, d2], dim=1)
        u7 = self.up7(u6)
        u7 = torch.cat([u7, d1], dim=1)
        out = self.last(u7)
        return self.tanh(out)

#############################
# Controller: Image Processing
#############################
class ImageColorizationController:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
    
    def load_model(self, model_path):
        model = UNetGenerator()
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
            model.load_state_dict(checkpoint['generator_state_dict'])
            model.eval()
            return model
        else:
            return None
    
    def preprocess_image(self, image):
        image_resized = image.resize((256, 256))
        image_np = np.array(image_resized) / 255.0
        lab_image = rgb2lab(image_np).astype(np.float32)
        L = lab_image[:, :, 0:1] / 100.0
        return torch.tensor(L).permute(2, 0, 1).unsqueeze(0).float()
    
    def colorize_image(self, L_tensor):
        with torch.no_grad():
            predicted_AB = self.model(L_tensor).squeeze(0).permute(1, 2, 0).numpy() * 128.0
        return predicted_AB

#############################
# View: Streamlit UI
#############################
st.title("🖌️ Image Colorization with GAN")
controller = ImageColorizationController("best_checkpoint3.pth")
mode = st.sidebar.selectbox("Select Mode", ["Predict", "Evaluate"])

# Pastikan model tidak None agar bisa diakses
if controller.model is not None:
    if mode == "Predict":
        uploaded_file = st.file_uploader("Upload a grayscale image (JPG/PNG)", type=["jpg","jpeg","png"])
        if uploaded_file is not None:
            input_img = Image.open(uploaded_file).convert('RGB')
            st.image(input_img, caption="Uploaded Image (Original)", use_column_width=True)
            if st.button("Colorize Image"):
                L_tensor = controller.preprocess_image(input_img)
                predicted_AB = controller.colorize_image(L_tensor)
                L_channel = (L_tensor.squeeze().numpy() * 100.0)
                lab_result = np.concatenate((L_channel[..., np.newaxis], predicted_AB), axis=-1)
                rgb_result = lab2rgb(lab_result)
                output_img = Image.fromarray((rgb_result * 255).astype(np.uint8))
                st.image(output_img, caption="Colorized Image (Output)", use_column_width=True)

    elif mode == "Evaluate":
        st.header("Evaluation on Test Dataset")
        # Tentukan path dataset test sesuai kebutuhan
        TEST_DATASET_PATH = r"C:\Users\admin\Downloads\Linnaeus 5 256X256\Linnaeus 5 256X256\test"
        
        if os.path.exists(TEST_DATASET_PATH):
            if st.button("Evaluate on Test Dataset"):
                with st.spinner("Evaluating..."):
                    test_files = list_images_recursively(TEST_DATASET_PATH)
                    if not test_files:
                        st.warning("No valid image files found.")
                    else:
                        # Agar pemanggilan model tidak error:
                        model = controller.model

                        cumulative_ssim = 0.0
                        cumulative_mae = 0.0
                        cumulative_color_gt = 0.0
                        cumulative_color_pred = 0.0
                        count = 0

                        ssim_list = []
                        mae_list = []
                        color_gt_list = []
                        color_pred_list = []
                        results_placeholder = st.empty()

                        for idx, file_path in enumerate(tqdm(test_files, desc="Processing")):
                            file_name = os.path.basename(file_path)
                            image = Image.open(file_path).convert('RGB')
                            image_resized = image.resize((256, 256))
                            image_np = np.array(image_resized) / 255.0
                            gray_image = rgb2gray(image_np)
                            L = gray_image * 100.0
                            L_normalized = L / 100.0
                            L_tensor = torch.tensor(L_normalized).unsqueeze(0).unsqueeze(0).float()

                            with torch.no_grad():
                                predicted_AB = model(L_tensor).squeeze(0).permute(1, 2, 0).numpy() * 128.0

                            lab_generated = np.zeros((256, 256, 3), dtype=np.float32)
                            lab_generated[..., 0] = L
                            lab_generated[..., 1:] = predicted_AB
                            rgb_generated = lab2rgb(lab_generated)

                            try:
                                ssim_val = compute_ssim(rgb_generated, image_np)
                            except ValueError:
                                # Jika terjadi error pada SSIM, skip gambar ini
                                continue
                            mae_val = compute_mae(rgb_generated, image_np)

                            # Compute colorfulness untuk GT dan Pred
                            color_gt = compute_colorfulness(image_np)
                            color_pred = compute_colorfulness(rgb_generated)

                            cumulative_ssim += ssim_val
                            cumulative_mae += mae_val
                            cumulative_color_gt += color_gt
                            cumulative_color_pred += color_pred
                            count += 1

                            ssim_list.append(ssim_val)
                            mae_list.append(mae_val)
                            color_gt_list.append(color_gt)
                            color_pred_list.append(color_pred)

                            # Tampilkan contoh beberapa gambar
                            if idx < 3:
                                with results_placeholder.container():
                                    st.subheader(f"Sample {idx+1}: {file_name}")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.image(image_resized, caption="Original", use_column_width=True)
                                    with col2:
                                        gray_uint8 = (gray_image * 255).astype(np.uint8)
                                        st.image(Image.fromarray(gray_uint8), caption="Grayscale", use_column_width=True)
                                    with col3:
                                        colorized_uint8 = (rgb_generated * 255).astype(np.uint8)
                                        st.image(Image.fromarray(colorized_uint8), caption="Colorized", use_column_width=True)
                            elif idx == 3:
                                # Setelah 3 sample, kosongkan placeholder untuk hemat ruang
                                results_placeholder.empty()

                        if count > 0:
                            avg_ssim = cumulative_ssim / count
                            avg_mae = cumulative_mae / count
                            avg_color_gt = cumulative_color_gt / count
                            avg_color_pred = cumulative_color_pred / count

                            st.markdown("### Average Metrics")
                            df = pd.DataFrame({
                                'Metric': ['SSIM', 'MAE', 'Colorfulness GT', 'Colorfulness Pred'],
                                'Average': [
                                    f"{avg_ssim:.4f}", 
                                    f"{avg_mae:.4f}", 
                                    f"{avg_color_gt:.4f}", 
                                    f"{avg_color_pred:.4f}"
                                ]
                            })
                            st.table(df)
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download Metrics CSV",
                                data=csv,
                                file_name='evaluation_metrics.csv',
                                mime='text/csv'
                            )
                        else:
                            st.warning("No images evaluated.")
        else:
            st.warning("TEST_DATASET_PATH does not exist.")
else:
    st.warning("No model found.")
