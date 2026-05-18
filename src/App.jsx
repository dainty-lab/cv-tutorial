import { useState, useRef, useEffect } from "react";

const P = {
  bg:"#04060d", surface:"#080f1e", card:"#0d1529", card2:"#111f36",
  accent1:"#38bdf8", accent2:"#818cf8", accent3:"#34d399",
  accent4:"#f472b6", accent5:"#fb923c", accent6:"#facc15", accent7:"#a78bfa",
  accent8:"#4ade80", accent9:"#f87171",
  text:"#e2e8f0", muted:"#64748b", border:"#1e2d45", ok:"#34d399", warn:"#fb923c",
};
const CL=[P.accent1,P.accent2,P.accent3,P.accent4,P.accent5,P.accent6,P.accent7,P.accent8,P.accent9];
const C=i=>CL[i%CL.length];

/* ================================================================
   30 CV DOMAINS
================================================================ */
const DOMAINS = [
  { id:1, name:"Image Classification", color:P.accent1, tagline:"Assign a label to an entire image",
    theory:`Image classification assigns a single class label to an entire image. It is the foundation of computer vision. Early approaches used hand-crafted features (SIFT, HOG, SURF) paired with SVMs or k-NN classifiers. AlexNet (2012) demonstrated that deep CNNs trained end-to-end on GPUs dramatically outperform hand-crafted pipelines. Key concepts: convolutional layers extract local spatial features; pooling layers provide translation invariance; fully-connected layers aggregate global features; softmax converts logits to probabilities. Modern classifiers use residual connections (ResNet), dense connections (DenseNet), or Vision Transformers (ViT) that treat image patches as sequence tokens. EfficientNet optimises model size vs accuracy via compound scaling of width, depth, and resolution simultaneously.`,
    architectures:["LeNet-5","AlexNet","VGG-16/19","GoogLeNet/Inception","ResNet-50/101/152","DenseNet","EfficientNet","MobileNetV3","Vision Transformer (ViT)","ConvNeXt"],
    metrics:["Top-1 Accuracy","Top-5 Accuracy","F1-Score","Confusion Matrix","AUC-ROC"],
    datasets:["ImageNet (1.2M images, 1000 classes)","CIFAR-10/100","MNIST / FashionMNIST","Caltech-101/256","Oxford 102 Flowers","Stanford Cars","iNaturalist"],
    colab:`# Image Classification with ResNet-50 (pretrained on ImageNet)
import torch, torchvision, io, urllib.request
from torchvision import transforms, models
from PIL import Image

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()

tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
img = Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB")
x = tf(img).unsqueeze(0)

with torch.no_grad():
    logits = model(x)
    probs  = torch.softmax(logits, dim=1)
    top5   = probs.topk(5)

LABELS = urllib.request.urlopen(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
).read().decode().splitlines()

for p, idx in zip(top5.values[0], top5.indices[0]):
    print(f"{LABELS[idx]:35s} {p.item()*100:.2f}%")` },

  { id:2, name:"Object Detection", color:P.accent2, tagline:"Locate and classify multiple objects with bounding boxes",
    theory:`Object detection simultaneously answers WHERE and WHAT: it draws axis-aligned bounding boxes around objects and assigns class labels. Two paradigms dominate. Two-stage detectors (Faster R-CNN) first propose candidate regions then refine them via a second network. One-stage detectors (YOLO, SSD, RetinaNet) predict boxes and classes in a single forward pass, trading some accuracy for much higher speed. Key concepts: anchor boxes are predefined reference boxes at multiple scales and aspect ratios; IoU (Intersection-over-Union) measures box overlap; Non-Maximum Suppression (NMS) eliminates duplicate detections; Focal Loss addresses extreme class imbalance between foreground objects and background. DETR replaced anchors entirely with transformer encoder-decoder and bipartite matching loss, eliminating hand-engineered components. YOLOv8 through v10 represent the current real-time detection frontier.`,
    architectures:["Faster R-CNN","YOLO v5/v8/v9/v10","SSD","RetinaNet","DETR","RT-DETR","EfficientDet","CenterNet","CornerNet","DINO-Det"],
    metrics:["mAP@0.5","mAP@0.5:0.95","IoU","Precision","Recall","FPS"],
    datasets:["COCO (330K images, 80 classes)","Pascal VOC 2007/2012","Open Images v7","LVIS v1","Objects365","VisDrone","KITTI"],
    colab:`# Object Detection with YOLOv8 (Ultralytics)
# !pip install ultralytics
from ultralytics import YOLO
import urllib.request, io
from PIL import Image
import numpy as np, matplotlib.pyplot as plt, matplotlib.patches as patches

model = YOLO("yolov8n.pt")  # nano model, ~6MB
url   = "https://ultralytics.com/images/bus.jpg"
img   = Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB")

results = model(img)
boxes   = results[0].boxes

fig, ax = plt.subplots(1,1,figsize=(12,8))
ax.imshow(np.array(img))
for box in boxes:
    x1,y1,x2,y2 = box.xyxy[0].tolist()
    cls  = int(box.cls[0]); conf = float(box.conf[0])
    label = model.names[cls]
    rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=2,edgecolor="lime",facecolor="none")
    ax.add_patch(rect)
    ax.text(x1, y1-5, f"{label} {conf:.2f}", color="lime", fontsize=9, fontweight="bold")
ax.axis("off"); plt.tight_layout(); plt.savefig("detection.png",dpi=150); plt.show()` },

  { id:3, name:"Semantic Segmentation", color:P.accent3, tagline:"Assign a class label to every pixel",
    theory:`Semantic segmentation produces a dense per-pixel class map. Unlike detection, it has no notion of individual instances; all pixels of the same class share one label. Encoder-decoder networks are standard: the encoder progressively downsamples to extract rich features; the decoder upsamples back to original resolution using skip connections. FCN established the paradigm by replacing fully-connected layers with convolutions. DeepLab introduced dilated (atrous) convolutions to expand receptive fields without losing resolution, plus ASPP (Atrous Spatial Pyramid Pooling) for multi-scale context. SegFormer uses a hierarchical transformer encoder with a lightweight MLP decoder. Segment Anything Model (SAM) enables zero-shot segmentation of arbitrary objects given prompts (points, boxes, or text). Panoptic segmentation unifies semantic and instance segmentation into a single prediction.`,
    architectures:["FCN","U-Net","SegNet","DeepLab v3+","PSPNet","HRNet","SegFormer","Mask2Former","OneFormer","SAM / SAM2"],
    metrics:["Pixel Accuracy","Mean IoU (mIoU)","Frequency-Weighted IoU","Dice Coefficient","Boundary F1"],
    datasets:["Cityscapes (19 classes, urban)","ADE20K (150 categories)","Pascal VOC 2012","COCO-Stuff","SUN RGB-D","Mapillary Vistas"],
    colab:`# Semantic Segmentation with SegFormer (HuggingFace)
# !pip install transformers
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import urllib.request, io, torch, numpy as np, matplotlib.pyplot as plt

extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
model     = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
model.eval()

url = "https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=640"
img = Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB")

inputs = extractor(images=img, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits   # (1, 150, H/4, W/4)

seg = logits.argmax(1).squeeze().numpy()
seg_vis = np.array(Image.fromarray(seg.astype(np.uint8)).resize(img.size, Image.NEAREST))

fig, (a,b) = plt.subplots(1,2,figsize=(14,6))
a.imshow(img); a.set_title("Original"); a.axis("off")
b.imshow(seg_vis, cmap="tab20"); b.set_title("Semantic Segmentation (ADE20K)"); b.axis("off")
plt.tight_layout(); plt.savefig("segmentation.png",dpi=150); plt.show()` },

  { id:4, name:"Instance Segmentation", color:P.accent4, tagline:"Detect and mask each individual object separately",
    theory:`Instance segmentation distinguishes individual object instances: two dogs each get their own binary mask. Mask R-CNN extends Faster R-CNN with a parallel mask-prediction branch using RoIAlign, which fixes pixel misalignment issues of RoIPool. YOLACT introduced real-time instance segmentation by linearly combining a set of prototype masks with per-instance coefficients. SOLOv2 avoids bounding boxes entirely, predicting masks directly from spatial grid locations. Panoptic segmentation assigns every pixel a semantic label plus an instance ID for countable things (people, cars) while using only semantic labels for stuff (sky, road). YOLOv8-seg provides an accessible entry point combining detection and segmentation in one unified model.`,
    architectures:["Mask R-CNN","PANet","YOLACT","SOLOv2","CondInst","Mask2Former","YOLOv8-seg","QueryInst","CellViT","SAM2"],
    metrics:["Mask AP","Mask AP@0.5","AP per category","Panoptic Quality (PQ)","RQ (Recognition Quality)"],
    datasets:["COCO panoptic + instance","LVISv1","Cityscapes instance","SBD","OpenImages instance","iSAID (aerial)"],
    colab:`# Instance Segmentation with YOLOv8-seg
# !pip install ultralytics
from ultralytics import YOLO
import urllib.request, io
from PIL import Image
import numpy as np, matplotlib.pyplot as plt

model  = YOLO("yolov8n-seg.pt")
url    = "https://ultralytics.com/images/zidane.jpg"
img    = Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB")
img_np = np.array(img)

results = model(img)
r       = results[0]
overlay = img_np.copy()

if r.masks is not None:
    colors = plt.cm.tab10(np.linspace(0,1,len(r.masks.data)))
    for i, mask in enumerate(r.masks.data):
        m = mask.cpu().numpy().astype(bool)
        m_resized = np.array(Image.fromarray(m).resize((img_np.shape[1],img_np.shape[0]),Image.NEAREST))
        overlay[m_resized] = (overlay[m_resized]*0.4 + np.array(colors[i,:3])*255*0.6).astype(np.uint8)

plt.figure(figsize=(12,8)); plt.imshow(overlay); plt.axis("off")
plt.title("Instance Segmentation"); plt.savefig("instance_seg.png",dpi=150); plt.show()` },

  { id:5, name:"Image Restoration", color:P.accent5, tagline:"Recover clean images from degraded observations",
    theory:`Image restoration recovers a clean image from one corrupted by noise, blur, rain, haze, or compression artifacts. The fundamental challenge is ill-posedness: many clean images could produce the same degraded observation. Classical methods used Total Variation regularisation, BM3D for denoising, or Wiener filters for deblurring. DnCNN introduced residual learning for denoising: predict the noise residual and subtract it from the input. Restormer applied transformer blocks with transposed multi-head self-attention across channels, enabling high-resolution processing efficiently. NAFNet removed all non-linear activation functions, achieving strong performance with simpler modules. PromptIR handles multiple degradation types with a single model using degradation-type prompts. Key metrics: PSNR (higher is better, in dB) and SSIM (structural similarity, 0 to 1, higher is better).`,
    architectures:["DnCNN","FFDNet","DPED","SwinIR","Restormer","NAFNet","MPRNet","DiffIR","PromptIR","IRControlNet"],
    metrics:["PSNR (dB)","SSIM","LPIPS","FID","NIQE (no-reference)","BRISQUE"],
    datasets:["BSD500","CBSD68","Kodak24","Rain100L/H","SOTS (dehazing)","DIV2K","GoPro (deblurring)","SIDD (real noise)"],
    colab:`# Image Denoising: manual Gaussian noise addition + NLMeans denoising + PSNR/SSIM
import cv2, urllib.request, numpy as np, matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw = urllib.request.urlopen(url).read()
clean = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

# Add Gaussian noise (sigma=30)
rng   = np.random.default_rng(42)
noise = rng.normal(0, 30, clean.shape).astype(np.float32)
noisy = np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# Denoise: Non-Local Means (strong classical baseline)
denoised = cv2.fastNlMeansDenoisingColored(noisy, None, h=10, hColor=10,
                                            templateWindowSize=7, searchWindowSize=21)

p_noisy    = psnr(clean, noisy,    data_range=255)
p_denoised = psnr(clean, denoised, data_range=255)
s_denoised = ssim(clean, denoised, channel_axis=2, data_range=255)
print(f"PSNR noisy:    {p_noisy:.2f} dB")
print(f"PSNR denoised: {p_denoised:.2f} dB  |  SSIM: {s_denoised:.4f}")

fig, axes = plt.subplots(1,3,figsize=(15,5))
for ax,im,t in zip(axes,
    [clean, noisy, denoised],
    ["Clean (reference)", "Noisy sigma=30", f"NLMeans PSNR={p_denoised:.1f}dB"]):
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.set_title(t); ax.axis("off")
plt.tight_layout(); plt.savefig("denoising.png",dpi=150); plt.show()` },

  { id:6, name:"Super-Resolution", color:P.accent6, tagline:"Reconstruct high-resolution detail from low-resolution input",
    theory:`Single-image super-resolution (SISR) recovers a high-resolution image from a low-resolution observation. SRCNN was the first deep CNN approach, trained with pixel-wise MSE loss. SRGAN introduced perceptual loss (VGG feature-space distance) and adversarial training, producing sharper, more realistic textures than MSE-based methods which tend to be blurry. ESRGAN refined discriminator architecture and used Residual-in-Residual Dense Blocks (RRDB). Real-ESRGAN extended to real-world degradations using a high-order degradation pipeline that simulates realistic blur, noise, and JPEG compression chains. SwinIR and HAT use shifted window transformers for long-range dependency modeling. Scale is typically 2x, 3x, or 4x. Blind SR handles unknown and complex degradations. Diffusion-based SR (SR3, StableSR) generates photorealistic textures at the cost of increased inference time.`,
    architectures:["SRCNN","VDSR","ESPCN","SRGAN","ESRGAN","Real-ESRGAN","SwinIR","HAT","EDSR","StableSR"],
    metrics:["PSNR","SSIM","LPIPS","MOS (mean opinion score)","NIQE","PI (perceptual index)"],
    datasets:["DIV2K","Flickr2K","BSD100","Set5","Set14","Urban100","Manga109"],
    colab:`# Super-Resolution: bicubic vs Lanczos comparison + PSNR measurement
import urllib.request, io, numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
hr  = Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB")

# Downsample 4x then upsample back
lr           = hr.resize((hr.width//4, hr.height//4), Image.BICUBIC)
bicubic_up   = lr.resize(hr.size, Image.BICUBIC)
lanczos_up   = lr.resize(hr.size, Image.LANCZOS)
nearest_up   = lr.resize(hr.size, Image.NEAREST)

hr_np, bic_np, lan_np = np.array(hr), np.array(bicubic_up), np.array(lanczos_up)
print(f"Nearest  PSNR: {psnr(hr_np, np.array(nearest_up), data_range=255):.2f} dB")
print(f"Bicubic  PSNR: {psnr(hr_np, bic_np, data_range=255):.2f} dB")
print(f"Lanczos  PSNR: {psnr(hr_np, lan_np, data_range=255):.2f} dB")

fig, axes = plt.subplots(1,4,figsize=(18,5))
for ax,im,t in zip(axes,[hr,lr,bicubic_up,lanczos_up],["HR (original)","LR (4x down)","Bicubic x4","Lanczos x4"]):
    ax.imshow(im); ax.set_title(t); ax.axis("off")
plt.tight_layout(); plt.savefig("sr.png",dpi=150); plt.show()` },

  { id:7, name:"Image Dehazing", color:P.accent7, tagline:"Remove fog and haze to restore clear visibility",
    theory:`Haze follows the atmospheric scattering model: I(x) = J(x)*t(x) + A*(1-t(x)), where I is the hazy input, J the clean scene, t the transmission map (fraction of scene light reaching the camera), and A the global atmospheric light. The Dark Channel Prior (DCP) observes that haze-free outdoor images have at least one color channel with very low intensity in most local patches; this prior enables estimating t and A from the hazy image alone. Deep learning methods learn end-to-end mappings from hazy to clean: AOD-Net reformulated the model into a single learnable module; FFA-Net used feature fusion attention; DehazeFormer applies Swin Transformer blocks with revised normalization and activation suited to restoration. Evaluation uses PSNR/SSIM on the SOTS subset of RESIDE benchmark, covering both indoor and outdoor haze.`,
    architectures:["DCP (Dark Channel Prior)","DehazeNet","AOD-Net","GDN","FFA-Net","MSBDN","DehazeFormer","C2PNet","RIDCP","MB-TaylorFormer"],
    metrics:["PSNR","SSIM","CIEDE2000","FID","FADE (foggy density)"],
    datasets:["RESIDE (ITS/OTS/SOTS)","Dense-Haze","NH-HAZE","O-HAZE","NTIRE Dehazing","4KID"],
    colab:`# Dark Channel Prior Dehazing (classical algorithm from scratch)
import cv2, numpy as np, urllib.request, matplotlib.pyplot as plt

def dark_channel(img, patch=15):
    dc     = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(patch,patch))
    return cv2.erode(dc, kernel)

def atmospheric_light(img, dc, top_frac=0.001):
    n_px  = max(int(dc.size * top_frac), 1)
    idx   = np.argsort(dc.flatten())[-n_px:]
    h, w  = dc.shape
    rows, cols = np.unravel_index(idx, (h,w))
    return img[rows, cols].max(axis=0)

def dehaze(img_bgr, omega=0.95, patch=15):
    img = img_bgr.astype(np.float64) / 255.0
    dc  = dark_channel(img, patch)
    A   = atmospheric_light(img, dc)
    t   = 1.0 - omega * dark_channel(img / A, patch)
    t   = np.clip(t, 0.1, 1.0)[:,:,np.newaxis]
    J   = (img - A) / t + A
    return np.clip(J * 255, 0, 255).astype(np.uint8)

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Foggy_forest.jpg/320px-Foggy_forest.jpg"
raw = urllib.request.urlopen(url).read()
hazy = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
res  = dehaze(hazy)

fig,(a,b) = plt.subplots(1,2,figsize=(12,5))
a.imshow(cv2.cvtColor(hazy,cv2.COLOR_BGR2RGB)); a.set_title("Hazy Input"); a.axis("off")
b.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB)); b.set_title("DCP Dehazed"); b.axis("off")
plt.tight_layout(); plt.savefig("dehaze.png",dpi=150); plt.show()` },

  { id:8, name:"Image Deraining", color:P.accent8, tagline:"Remove rain streaks from images and video",
    theory:`Rain degrades images with additive streak-like artifacts whose orientation, density, and magnitude vary. The degradation model is O = B + R, where O is the rainy observation, B the background, and R the rain layer. Early methods used sparse coding or Gaussian mixture models to separate B and R. Deep learning methods either subtract the estimated rain layer or directly map rainy to clean. RESCAN used recurrent CNNs to model multi-scale rain accumulation. MPRNet introduced a multi-stage progressive pipeline with supervised attention modules. DerainFormer and Transweather apply transformer architectures that generalize across multiple weather degradations. Video deraining also leverages temporal consistency. Performance is measured on Rain100L (light) and Rain100H (heavy rain) benchmarks using PSNR and SSIM.`,
    architectures:["JORDER","RESCAN","PReNet","MSPFN","MPRNet","DerainFormer","Restormer","Transweather","DRSformer","IDT"],
    metrics:["PSNR","SSIM","VIF","FADE"],
    datasets:["Rain100L","Rain100H","Rain1200","Rain1400","SPA-Data","RainDS","AGAN-Rain","GT-Rain"],
    colab:`# Synthesise rain streaks then remove via morphological inpainting
import cv2, numpy as np, urllib.request, matplotlib.pyplot as plt

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw = urllib.request.urlopen(url).read()
clean = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_COLOR)
h, w = clean.shape[:2]

# Synthesise rain streaks
rain_layer = np.zeros((h,w), dtype=np.uint8)
rng = np.random.default_rng(0)
for _ in range(2000):
    x, y = rng.integers(0,w), rng.integers(0,h)
    L    = rng.integers(10,30); angle = rng.uniform(-10,10)
    x2   = int(x + L*np.sin(np.radians(angle)))
    y2   = int(y + L*np.cos(np.radians(angle)))
    cv2.line(rain_layer,(x,y),(x2,y2),255,1)
rainy = np.clip(clean.astype(int) + rain_layer[:,:,None]*0.9, 0, 255).astype(np.uint8)

# Detect and inpaint rain pixels (morphological top-hat detects thin bright streaks)
gray   = cv2.cvtColor(rainy, cv2.COLOR_BGR2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,7))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
mask   = (tophat > 25).astype(np.uint8) * 255
restored = cv2.inpaint(rainy, mask, 3, cv2.INPAINT_TELEA)

fig,(a,b,c) = plt.subplots(1,3,figsize=(15,5))
for ax,im,t in zip([a,b,c],[clean,rainy,restored],["Clean","Rainy","Restored"]):
    ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)); ax.set_title(t); ax.axis("off")
plt.tight_layout(); plt.savefig("derain.png",dpi=150); plt.show()` },

  { id:9, name:"Low-Light Enhancement", color:P.accent9, tagline:"Brighten and restore detail from dark images",
    theory:`Low-light images suffer from low brightness, high noise, color distortion, and detail loss. Retinex theory (Edwin Land, 1977) models image formation as I = R * L, where R is reflectance (intrinsic scene property) and L is illumination. Enhancement modifies L while preserving R. Classical methods: histogram equalization globally stretches contrast; CLAHE (Contrast Limited Adaptive HE) applies it locally to avoid over-amplification. Deep methods: RetinexNet decomposes images into reflectance and illumination components then enhances illumination; EnlightenGAN trains unpaired (no paired clean/dark data required) using adversarial loss; Zero-DCE (Zero-Reference Deep Curve Estimation) requires no paired data or dark images, learning local curve adjustment maps. SNR-Aware weights enhancement by signal-to-noise ratio per pixel. LOL v1/v2 dataset provides matched low-light and normal-light pairs.`,
    architectures:["LIME","RetinexNet","EnlightenGAN","Zero-DCE","KinD++","SNR-Aware","LLFlow","Bread","MIRNet","Retinexformer"],
    metrics:["PSNR","SSIM","NIQE","BRISQUE","LOE (lightness order error)","VIF"],
    datasets:["LOL v1/v2","MIT-Adobe FiveK","VE-LOL","LSRW","DICM","NPE","MEF","LIME dataset"],
    colab:`# Low-light: simulate dark image, then enhance with Gamma correction and CLAHE
import cv2, numpy as np, urllib.request, matplotlib.pyplot as plt

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"
raw = urllib.request.urlopen(url).read()
img = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_COLOR)
dark = np.clip(img.astype(np.float32)*0.15, 0, 255).astype(np.uint8)

# Gamma correction (gamma < 1 brightens)
gamma = 0.45
lut   = np.array([(i/255.0)**(gamma)*255 for i in range(256)]).astype(np.uint8)
gamma_img = cv2.LUT(dark, lut)

# CLAHE on L channel in LAB space
lab = cv2.cvtColor(dark, cv2.COLOR_BGR2LAB)
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
lab[:,:,0] = clahe.apply(lab[:,:,0])
clahe_img  = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

fig, axes = plt.subplots(1,4,figsize=(18,5))
for ax,im,t in zip(axes,[img,dark,gamma_img,clahe_img],
                        ["Original","Dark (sim.)","Gamma 0.45","CLAHE"]):
    ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)); ax.set_title(t); ax.axis("off")
plt.tight_layout(); plt.savefig("lowlight.png",dpi=150); plt.show()` },

  { id:10, name:"Image Inpainting", color:P.accent1, tagline:"Fill missing or masked regions with plausible content",
    theory:`Inpainting removes unwanted objects or repairs damaged regions by filling them with visually coherent content. Classical exemplar-based inpainting (Criminisi et al.) propagates texture patches from surrounding areas guided by a fill-priority that fills structure before texture. Context Encoders (Pathak et al., 2016) were the first CNN approach using an adversarial loss to hallucinate missing content. Partial convolutions masked the convolution response so network never sees invalid pixels. Gated convolutions learned soft masks via sigmoid gates. DeepFill used a two-stage coarse-to-fine approach with contextual attention. LaMa (Large Mask) uses Fast Fourier Convolutions that have global receptive field from layer one, excelling at repeating textures. Stable Diffusion inpainting with text guidance enables replacing arbitrary objects with semantically appropriate alternatives. Evaluation uses L1/L2 on masked regions, FID, LPIPS.`,
    architectures:["Exemplar-based (Criminisi)","Context Encoders","Partial Conv","Gated Conv","DeepFill v1/v2","LaMa","MAT","Stable Diffusion Inpaint","PowerPaint","BrushNet"],
    metrics:["PSNR (masked region)","SSIM","L1 Error","FID","LPIPS","U-IDS"],
    datasets:["CelebA-HQ","Places2","ImageNet (Places)","Paris StreetView","FFHQ"],
    colab:`# Image inpainting with OpenCV Telea and Navier-Stokes algorithms
import cv2, numpy as np, urllib.request, matplotlib.pyplot as plt

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
raw = urllib.request.urlopen(url).read()
img = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_COLOR)
h, w = img.shape[:2]

# Simulate damage: central rectangular hole + diagonal scratches
mask = np.zeros((h,w), dtype=np.uint8)
cv2.rectangle(mask,(w//4,h//3),(3*w//4,2*h//3),255,-1)
cv2.line(mask,(0,0),(w//3,h//2),255,4)

damaged = img.copy(); damaged[mask==255] = 0

telea = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
ns    = cv2.inpaint(img, mask, 5, cv2.INPAINT_NS)

fig, axes = plt.subplots(1,4,figsize=(18,5))
for ax,im,t in zip(axes,[img,damaged,telea,ns],
    ["Original","Damaged","Telea (fast marching)","Navier-Stokes"]):
    ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)); ax.set_title(t); ax.axis("off")
plt.tight_layout(); plt.savefig("inpaint.png",dpi=150); plt.show()` },

  { id:11, name:"Face Recognition", color:P.accent2, tagline:"Identify or verify people by facial appearance",
    theory:`Face recognition encompasses four stages: detection (find faces), alignment (normalise by eye/nose landmarks), embedding (map face to a compact vector), and matching (compare vectors). DeepFace and FaceNet established deep metric learning: train a CNN to embed faces into Euclidean space where same-identity pairs are close and different-identity pairs are far. Loss functions: contrastive loss (pairs), triplet loss (anchor/positive/negative), ArcFace/CosFace use additive angular margin in normalised softmax to maximise inter-class angular separation. FaceNet achieves 99.63% on LFW. Key challenge: pose variation, occlusion, aging, low resolution, makeup, and skin tone bias. MTCNN detects faces and facial landmarks in a cascaded manner suitable for real-time systems.`,
    architectures:["DeepFace","FaceNet","SphereFace","CosFace","ArcFace","AdaFace","ElasticFace","MagFace","PartialFC","UniPortrait"],
    metrics:["Accuracy@LFW","TAR@FAR (verification)","Rank-1 (identification)","AUC-ROC","EER"],
    datasets:["LFW (11K pairs)","MS-Celeb-1M","VGGFace2","CASIA-WebFace","MegaFace","IJB-B/C","AgeDB"],
    colab:`# Face detection with MTCNN pipeline (facenet-pytorch)
# !pip install facenet-pytorch
from facenet_pytorch import MTCNN
from PIL import Image
import urllib.request, io, numpy as np
import matplotlib.pyplot as plt, matplotlib.patches as patches

mtcnn = MTCNN(keep_all=True, device="cpu")

# Use a royalty-free photo with faces (Wikipedia politician photo)
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Bill_Nye_2017.jpg/240px-Bill_Nye_2017.jpg"
img = Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB")

boxes, probs = mtcnn.detect(img)

fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(img)
if boxes is not None:
    for box, p in zip(boxes, probs):
        rect = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],
                                   linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        ax.text(box[0], box[1]-5, f"conf={p:.2f}", color="lime", fontsize=10, fontweight="bold")
    print(f"Detected {len(boxes)} face(s)")
ax.axis("off"); plt.title("MTCNN Face Detection"); plt.savefig("face.png",dpi=150); plt.show()` },

  { id:12, name:"Pose Estimation", color:P.accent3, tagline:"Locate body joints and estimate human pose in 2D or 3D",
    theory:`Human pose estimation predicts locations of anatomical keypoints (joints). Top-down approaches detect persons first, then estimate keypoints within each crop (HRNet, ViTPose). Bottom-up approaches detect all keypoints globally then group them into instances (OpenPose, HigherHRNet). Heatmap-based methods predict a Gaussian probability heatmap per joint; the peak gives the joint location. Regression-based methods directly predict coordinates. 3D pose estimation additionally predicts depth; monocular methods lift 2D poses to 3D using temporal context or learned depth priors. Evaluation metric: OKS (Object Keypoint Similarity), which penalises deviations based on per-joint standard deviations calibrated on human annotators. PCK (Percentage of Correct Keypoints) is used for MPII. For 3D: MPJPE (mean per-joint position error in mm).`,
    architectures:["OpenPose","Stacked Hourglass","SimpleBaseline","HRNet","HigherHRNet","ViTPose","RTMPose","DWPose","SMPL-X","4D Humans"],
    metrics:["OKS / mAP Keypoints","PCK@0.5","MPJPE (3D, mm)","PA-MPJPE","PVE"],
    datasets:["COCO Keypoints","MPII Human Pose","Human3.6M (3D)","MPI-INF-3DHP","PoseTrack","OCHuman","CrowdPose"],
    colab:`# Pose Estimation with MediaPipe Pose
# !pip install mediapipe
import mediapipe as mp, cv2, urllib.request, numpy as np
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Bill_Nye_2017.jpg/240px-Bill_Nye_2017.jpg"
raw = urllib.request.urlopen(url).read()
img = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    results = pose.process(img_rgb)
    ann = img_rgb.copy()
    if results.pose_landmarks:
        mp_draw.draw_landmarks(ann, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark
        print(f"Detected {len(lm)} landmarks")
        # Print shoulder positions
        left_sh  = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        print(f"Left shoulder:  ({left_sh.x:.3f}, {left_sh.y:.3f})")
        print(f"Right shoulder: ({right_sh.x:.3f}, {right_sh.y:.3f})")

fig,(a,b) = plt.subplots(1,2,figsize=(12,6))
a.imshow(img_rgb); a.set_title("Original"); a.axis("off")
b.imshow(ann); b.set_title("Pose Landmarks"); b.axis("off")
plt.tight_layout(); plt.savefig("pose.png",dpi=150); plt.show()` },

  { id:13, name:"Optical Flow", color:P.accent4, tagline:"Estimate pixel-level motion between consecutive frames",
    theory:`Optical flow estimates the apparent motion field (u,v) at every pixel between two frames. The brightness constancy constraint: I(x,y,t) = I(x+u, y+v, t+dt). Lucas-Kanade solves a local weighted least-squares problem assuming constant flow within a patch, fast but limited to small motions. Horn-Schunck adds global smoothness regularisation. FlowNet brought end-to-end CNN learning with a differentiable correlation layer measuring feature similarity between two frames. PWC-Net introduced pyramid processing with cost volume warping at each scale. RAFT (Recurrent All-Pairs Field Transforms, 2020) builds a 4D cost volume over all pairs of feature vectors across both frames, then iteratively updates the flow estimate using a GRU, achieving state-of-the-art on Sintel and KITTI. Applications: video compression, action recognition, autonomous driving, video stabilization.`,
    architectures:["Lucas-Kanade (sparse)","Farneback (dense)","FlowNet 2.0","SPyNet","PWC-Net","RAFT","GMA","FlowFormer","SKFlow","UniMatch"],
    metrics:["EPE (End-Point Error)","Fl-all (% outliers > 3px or 5%)","Sintel clean/final EPE","KITTI Fl-all"],
    datasets:["MPI-Sintel","KITTI Flow 2012/2015","FlyingChairs","FlyingThings3D","HD1K","Spring"],
    colab:`# Optical Flow: Lucas-Kanade sparse + Farneback dense
import cv2, numpy as np, urllib.request, matplotlib.pyplot as plt

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw = urllib.request.urlopen(url).read()
frame1 = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_GRAYSCALE)

# Simulate motion: affine warp (translation + slight rotation)
M      = cv2.getRotationMatrix2D((frame1.shape[1]//2,frame1.shape[0]//2), 3, 1.0)
M[:,2] += np.array([8, 4])
frame2  = cv2.warpAffine(frame1, M[:,:2], (frame1.shape[1],frame1.shape[0]))
M2 = np.float32([[1,0,8],[0,1,4]])
frame2 = cv2.warpAffine(frame1, M2, (frame1.shape[1],frame1.shape[0]))

# Sparse LK flow on Shi-Tomasi corners
corners = cv2.goodFeaturesToTrack(frame1,200,0.01,10)
pts2, st, _ = cv2.calcOpticalFlowPyrLK(frame1,frame2,corners,None)

# Dense Farneback flow
flow = cv2.calcOpticalFlowFarneback(frame1,frame2,None,0.5,3,15,3,5,1.2,0)
mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
hsv = np.zeros((*frame1.shape,3), dtype=np.uint8)
hsv[...,1] = 255
hsv[...,0] = (ang*180/np.pi/2).astype(np.uint8)
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

fig,(a,b,c) = plt.subplots(1,3,figsize=(15,5))
vis = cv2.cvtColor(frame1,cv2.COLOR_GRAY2RGB)
if corners is not None:
    for p,p2,s in zip(corners,pts2,st):
        if s[0]:
            cv2.arrowedLine(vis,tuple(p[0].astype(int)),tuple(p2[0].astype(int)),(0,255,0),1,tipLength=0.3)
a.imshow(vis); a.set_title("Sparse LK Flow"); a.axis("off")
b.imshow(frame2,cmap="gray"); b.set_title("Frame 2"); b.axis("off")
c.imshow(flow_rgb); c.set_title("Dense Farneback (HSV)"); c.axis("off")
plt.tight_layout(); plt.savefig("flow.png",dpi=150); plt.show()` },

  { id:14, name:"Action Recognition", color:P.accent5, tagline:"Classify human activities in video clips",
    theory:`Action recognition classifies what a person is doing in a video. Early methods extracted hand-crafted spatiotemporal features (Dense Trajectories, iDT). Two-stream networks combined spatial (RGB) and temporal (optical flow) CNN streams. 3D CNNs (C3D, I3D) extend 2D convolutions along the temporal dimension, learning joint spatiotemporal features. Inflated 3D (I3D) inflated ImageNet-pretrained 2D filters into 3D by repeating along the temporal axis, enabling pretraining knowledge transfer. SlowFast uses two pathways: Slow (high spatial resolution, low frame rate) for spatial semantics and Fast (low spatial resolution, high frame rate) for fine temporal motion. Transformer-based methods (TimeSformer, Video Swin, VideoMAE) apply space-time attention. Skeleton-based action recognition uses Graph CNNs (ST-GCN, CTR-GCN) over body joint graphs.`,
    architectures:["Two-Stream Network","C3D","I3D","SlowFast","TimeSformer","Video Swin","VideoMAE","ST-GCN","CTRGCN","InternVideo2"],
    metrics:["Top-1 Accuracy","Top-5 Accuracy","mAP (spatio-temporal)","GFLOPs"],
    datasets:["Kinetics-400/600/700","UCF-101","HMDB-51","Something-Something v2","NTU RGB+D 120","AVA","FineGym"],
    colab:`# Action Recognition concept: load video frames, extract optical flow features
import cv2, numpy as np, urllib.request, matplotlib.pyplot as plt

# Download a short GIF as proxy for video frames
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw = urllib.request.urlopen(url).read()
frame_base = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_COLOR)
gray_base  = cv2.cvtColor(frame_base, cv2.COLOR_BGR2GRAY)

# Simulate 5-frame sequence with progressive motion
frames = [gray_base]
for i in range(1,5):
    M = np.float32([[1,0,i*5],[0,1,i*2]])
    frames.append(cv2.warpAffine(gray_base,M,(gray_base.shape[1],gray_base.shape[0])))

# Compute flow magnitudes between consecutive frames
flow_mags = []
for i in range(len(frames)-1):
    fl = cv2.calcOpticalFlowFarneback(frames[i],frames[i+1],None,0.5,3,15,3,5,1.2,0)
    flow_mags.append(np.sqrt(fl[...,0]**2+fl[...,1]**2).mean())

print("Mean optical flow magnitude per frame transition:")
for i, m in enumerate(flow_mags):
    print(f"  Frame {i}->{i+1}: {m:.3f} px/frame")

# Temporal profile plot
plt.figure(figsize=(8,4))
plt.plot(range(len(flow_mags)), flow_mags, 'o-', color='cyan', linewidth=2)
plt.xlabel("Frame Transition"); plt.ylabel("Mean Flow Magnitude (px)")
plt.title("Temporal Motion Profile (proxy for action intensity)")
plt.tight_layout(); plt.savefig("action.png",dpi=150); plt.show()` },

  { id:15, name:"Depth Estimation", color:P.accent6, tagline:"Predict per-pixel depth from single or stereo images",
    theory:`Depth estimation infers scene geometry per pixel. Stereo depth uses disparity between two calibrated cameras; the problem is well-posed given calibration. Monocular depth from a single image is ill-posed, relying on learned cues: texture gradient, defocus, relative size, perspective convergence, and atmospheric haze. MonoDepth introduced left-right consistency loss for self-supervised training on stereo pairs. AdaBins divides depth range into adaptive intervals and classifies each pixel. DPT uses a dense prediction transformer. MiDaS produces affine-invariant relative depth that generalises across datasets. Depth Anything v2 (2024) trained on 62M images via semi-supervised learning, achieving best generalization. Metric3D and ZoeDepth recover absolute metric scale without camera intrinsics. Evaluation: AbsRel (absolute relative error), RMSE, delta1 (fraction within 1.25x of ground truth).`,
    architectures:["SGBM (stereo)","MonoDepth v1/v2","DORN","AdaBins","DPT","BinsFormer","MiDaS v3.1","ZoeDepth","Depth Anything v2","Metric3D v2"],
    metrics:["AbsRel","RMSE","SqRel","log10","delta1 (< 1.25)","delta2","delta3"],
    datasets:["NYU Depth v2","KITTI Depth","SUN RGB-D","ETH3D","DIML/CVL","Hypersim","UnrealStereo4K"],
    colab:`# Monocular Depth Estimation with MiDaS (torch.hub)
# !pip install timm
import torch, urllib.request, io, numpy as np
from PIL import Image
import matplotlib.pyplot as plt

model_type = "MiDaS_small"
midas      = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
midas.eval()
transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
transform  = transforms.small_transform

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
img_pil = Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB")
img_np  = np.array(img_pil)

inp = transform(img_np).unsqueeze(0)
with torch.no_grad():
    pred = midas(inp)
    pred = torch.nn.functional.interpolate(pred.unsqueeze(1),
               size=img_np.shape[:2], mode="bicubic", align_corners=False).squeeze()
depth = pred.numpy()

fig,(a,b) = plt.subplots(1,2,figsize=(14,6))
a.imshow(img_pil); a.set_title("Input Image"); a.axis("off")
im = b.imshow(depth, cmap="plasma"); b.set_title("Depth Map (MiDaS small)"); b.axis("off")
plt.colorbar(im,ax=b,fraction=0.046,pad=0.04,label="Relative depth (inverted)")
plt.tight_layout(); plt.savefig("depth.png",dpi=150); plt.show()` },

  { id:16, name:"3D Reconstruction", color:P.accent7, tagline:"Recover 3D structure from 2D images",
    theory:`3D reconstruction recovers scene geometry from images. Structure-from-Motion (SfM, implemented in COLMAP) simultaneously estimates camera poses and a sparse point cloud from overlapping images using feature matching (SIFT, SuperPoint) and bundle adjustment. MVS (Multi-View Stereo) then densifies the reconstruction. NeRF (Neural Radiance Fields, Mildenhall et al. 2020) represents scenes as a continuous 5D function: given (x,y,z,theta,phi), output colour and density. Volume rendering integrates samples along rays. Instant-NGP uses multi-resolution hash encoding to achieve seconds-long training. 3D Gaussian Splatting (Kerbl et al. 2023) represents scenes as millions of differentiable 3D Gaussians; rasterization renders them in real-time (100+ FPS at 1080p). Applications: digital twins, VR/AR, robotics, cultural heritage.`,
    architectures:["COLMAP (SfM+MVS)","NeRF","Instant-NGP","Mip-NeRF 360","TensoRF","3D Gaussian Splatting","4D-GS","SplatterImage","CAT3D","GaussianEditor"],
    metrics:["PSNR / SSIM / LPIPS (novel view)","Chamfer Distance","F-score (geometry)","Training time","FPS"],
    datasets:["NeRF synthetic (Blender)","LLFF","Tanks and Temples","DTU MVS","Mip-NeRF 360","ScanNet","Replica"],
    colab:`# Stereo disparity map as entry-point to 3D reconstruction
import cv2, numpy as np, urllib.request, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw = urllib.request.urlopen(url).read()
left_c  = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_COLOR)
left    = cv2.cvtColor(left_c, cv2.COLOR_BGR2GRAY)
right   = np.roll(left, 20, axis=1)  # Synthetic: shift = 20 px disparity

stereo  = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=64, blockSize=9,
    P1=8*9**2, P2=32*9**2, disp12MaxDiff=1,
    uniquenessRatio=10, speckleWindowSize=100, speckleRange=32)
disp    = stereo.compute(left,right).astype(np.float32)/16.0
disp[disp<=0] = np.nan

# Back-project to 3D (simplified, no real calibration)
h, w = disp.shape
f    = 500.0; B = 20.0  # Fake focal length and baseline
Y, X = np.mgrid[0:h, 0:w].astype(float)
Z    = f*B / (disp + 1e-6)
Z[np.isnan(disp)] = np.nan

fig,(a,b) = plt.subplots(1,2,figsize=(14,6))
a.imshow(left,cmap="gray"); a.set_title("Left Image"); a.axis("off")
b.imshow(disp,cmap="plasma"); b.set_title("Disparity Map"); b.axis("off")
plt.tight_layout(); plt.savefig("3d_recon.png",dpi=150); plt.show()` },

  { id:17, name:"Image Generation (GANs)", color:P.accent8, tagline:"Synthesize photorealistic images from noise or conditions",
    theory:`Generative Adversarial Networks (GANs) consist of a Generator G mapping noise z to images and a Discriminator D distinguishing real from generated images. The minimax game: G minimises log(1-D(G(z))), D maximises log(D(x))+log(1-D(G(z))). DCGAN stabilised training with convolutional architectures and batch normalisation. WGAN-GP replaced JS divergence with Wasserstein distance for more stable gradients. Progressive GAN grew resolution incrementally from 4x4 up to 1024x1024. StyleGAN v2/3 controls image style via AdaIN layers in a mapping network producing a disentangled latent space W, enabling semantic manipulation. CycleGAN enables unpaired image-to-image translation using cycle consistency loss. BigGAN scaled class-conditional generation on ImageNet to 512x512. FID (Frechet Inception Distance) is the standard metric, measuring distribution distance between real and generated image features in InceptionNet space.`,
    architectures:["DCGAN","WGAN-GP","ProGAN","StyleGAN v2/3","BigGAN","CycleGAN","Pix2Pix","StarGAN v2","VQGAN","GigaGAN"],
    metrics:["FID","IS (Inception Score)","KID","Precision/Recall","PPL (path length)"],
    datasets:["CelebA-HQ","FFHQ (70K faces, 1024px)","LSUN Bedrooms","ImageNet","AFHQ","MetFaces","AnimalFace"],
    colab:`# DCGAN-style architecture demo (untrained, for architecture illustration)
import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, z=100, ch=64, out_c=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z,ch*8,4,1,0,bias=False), nn.BatchNorm2d(ch*8), nn.ReLU(True),
            nn.ConvTranspose2d(ch*8,ch*4,4,2,1,bias=False), nn.BatchNorm2d(ch*4), nn.ReLU(True),
            nn.ConvTranspose2d(ch*4,ch*2,4,2,1,bias=False), nn.BatchNorm2d(ch*2), nn.ReLU(True),
            nn.ConvTranspose2d(ch*2,ch,4,2,1,bias=False), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.ConvTranspose2d(ch,out_c,4,2,1,bias=False), nn.Tanh())
    def forward(self,z): return self.net(z)

print("Generator architecture (DCGAN-style):")
G = Generator(); print(G)
print(f"Parameters: {sum(p.numel() for p in G.parameters()):,}")

G.eval()
with torch.no_grad():
    z    = torch.randn(16,100,1,1)
    imgs = G(z).permute(0,2,3,1).numpy()
    imgs = (imgs*0.5+0.5).clip(0,1)

fig, axes = plt.subplots(4,4,figsize=(8,8))
for ax,im in zip(axes.flat,imgs): ax.imshow(im); ax.axis("off")
plt.suptitle("DCGAN Generator (random weights, not trained)"); plt.tight_layout()
plt.savefig("gan.png",dpi=150); plt.show()` },

  { id:18, name:"Diffusion Models", color:P.accent9, tagline:"Generate images by reversing a learned noise diffusion process",
    theory:`Diffusion models (DDPM, Ho et al. 2020) define a forward Markov chain that gradually adds Gaussian noise to an image over T steps until it becomes pure noise, then train a U-Net to reverse this process. At inference, start from Gaussian noise and denoise iteratively. Latent Diffusion Models (LDM, aka Stable Diffusion) apply diffusion in a compressed VQ-VAE latent space, reducing computation dramatically. The U-Net is conditioned via cross-attention on text embeddings from CLIP or T5. DDIM enables 10-50 step sampling instead of 1000 steps via deterministic trajectories. Classifier-Free Guidance (CFG) jointly trains conditioned and unconditioned models, boosting image-text alignment at inference via linear extrapolation. ControlNet adds spatial conditioning (edges, depth, pose) via zero-convolution adapters, enabling fine-grained layout control. DiT (Diffusion Transformer) replaces the U-Net with a transformer, used in Flux.1 and SD3.`,
    architectures:["DDPM","DDIM","Stable Diffusion v1.5/XL/3","DALL-E 2/3","Imagen","ControlNet","IP-Adapter","DiT","Flux.1","Lumina-T2X"],
    metrics:["FID","IS","CLIP Score","PickScore","HPSv2","ImageReward"],
    datasets:["LAION-5B","LAION-Aesthetics v2","CC12M","COYO-700M","JourneyDB","DiffusionDB"],
    colab:`# Stable Diffusion text-to-image with diffusers
# !pip install diffusers transformers accelerate
from diffusers import StableDiffusionPipeline
import torch, matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=dtype)
pipe = pipe.to(device)

prompt   = "a golden retriever wearing a lab coat, professional photo, sharp focus, 8k"
negative = "blurry, low quality, cartoon, deformed"

image = pipe(prompt, negative_prompt=negative,
             num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("sd_output.png")

plt.figure(figsize=(8,8)); plt.imshow(image); plt.axis("off")
plt.title(prompt[:80]); plt.savefig("diffusion.png",dpi=150); plt.show()` },

  { id:19, name:"Medical Image Analysis", color:P.accent1, tagline:"Detect and segment structures in clinical imaging",
    theory:`Medical image analysis covers CT, MRI, X-ray, ultrasound, histopathology, and endoscopy. Key tasks: lesion detection, organ segmentation, classification (benign vs malignant), and multi-modal registration. Unique challenges: 3D volumetric data, severe class imbalance (tiny lesions vs large background), limited labeled data due to expert annotation cost, and strict regulatory requirements. U-Net (Ronneberger et al. 2015) was purpose-designed for biomedical segmentation: encoder-decoder with skip connections that preserve high-frequency spatial detail needed for precise boundary delineation with small training sets. nnU-Net automatically configures U-Net hyperparameters. TransUNet and SwinUNETR integrate transformers. CheXNet matched radiologist-level performance on chest X-ray pneumonia detection. MedSAM adapts SAM for universal medical image segmentation with 1.5M image-mask pairs.`,
    architectures:["U-Net","V-Net (3D)","nnU-Net","TransUNet","SwinUNETR","CheXNet","MedSAM","SAM-Med2D","SegVol","BioViL-T"],
    metrics:["Dice Score","HD95 (Hausdorff Distance)","AUC-ROC","Sensitivity","Specificity","mIoU per organ"],
    datasets:["BraTS (brain tumor MRI)","LiTS (liver CT)","KiTS21 (kidney)","ChestX-ray14","CheXpert","ISIC (skin lesion)","Camelyon16 (histology)","PanNuke"],
    colab:`# Medical image segmentation: threshold + morphology on a synthetic 2D slice
import numpy as np, matplotlib.pyplot as plt
import cv2, urllib.request

# Simulate a brain MRI axial slice using a grayscale image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw = urllib.request.urlopen(url).read()
img = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(256,256))

# Simulate MRI-like contrast (invert + normalize)
mri_sim = cv2.equalizeHist(img)

# Otsu threshold to segment "tissue" from "background"
_, binary    = cv2.threshold(mri_sim, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
opened       = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
closed       = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

# Connected components (simulating lesion labeling)
n, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)
print(f"Found {n-1} connected components (excluding background)")

fig, axes = plt.subplots(1,4,figsize=(18,5))
for ax,im,t in zip(axes,[mri_sim,binary,opened,closed],
                        ["MRI sim","Otsu Binary","Morphological Open","Final Mask"]):
    ax.imshow(im,cmap="gray"); ax.set_title(t); ax.axis("off")
plt.tight_layout(); plt.savefig("medical.png",dpi=150); plt.show()` },

  { id:20, name:"Remote Sensing", color:P.accent2, tagline:"Analyse Earth observation imagery from satellites and drones",
    theory:`Remote sensing applies CV to satellite, aerial, and drone imagery. Challenges include very high resolution (sub-meter ground sampling distance), multiple spectral bands beyond visible RGB (NIR, SWIR, TIR, SAR), large spatial extents, temporal change detection, and varying atmospheric conditions. Key tasks: land cover/use classification, building and road extraction, change detection, crop monitoring, disaster assessment, wildfire mapping, and deforestation tracking. Object detection in aerial imagery requires models robust to small objects and arbitrary orientations, hence rotated bounding boxes (OBB) in datasets like DOTA. SAR (Synthetic Aperture Radar) provides all-weather, day-night imaging with different statistical properties from optical. Remote sensing foundation models (RingMo, SatMAE, SkySense) pretrain on large satellite image archives.`,
    architectures:["DeepLab (aerial)","U-Net (geospatial)","ChangeFormer","RemoteCLIP","RingMo","SpectralGPT","Scale-MAE","SatMAE","GeoSAM","SkySense"],
    metrics:["mIoU","Overall Accuracy","Kappa Coefficient","COCO AP (detection)","Change Detection F1"],
    datasets:["SpaceNet 1-8","DeepGlobe","xView","DOTA v2.0","Potsdam/Vaihingen","BigEarthNet","LoveDA","SEN12MS"],
    colab:`# Remote sensing: NDVI calculation and false-color composite
import numpy as np, matplotlib.pyplot as plt, urllib.request, cv2

# Load a colour image and simulate NIR channel (use red channel proxy)
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Collage_of_Nine_Dogs.jpg/480px-Collage_of_Nine_Dogs.jpg"
raw = urllib.request.urlopen(url).read()
img = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float)

R = rgb[:,:,0]; G = rgb[:,:,1]; B = rgb[:,:,2]
# Simulated NIR = amplified green (vegetation-rich areas appear high)
NIR  = np.clip(G * 1.5, 0, 255)
NDVI = (NIR - R) / (NIR + R + 1e-8)   # Normalised Difference Vegetation Index

# False-colour composite: NIR, R, G mapped to display RGB
false_colour = np.stack([NIR/255, R/255, G/255],axis=2).clip(0,1)

fig, axes = plt.subplots(1,3,figsize=(18,6))
axes[0].imshow(rgb.astype(np.uint8)); axes[0].set_title("True Colour RGB"); axes[0].axis("off")
im = axes[1].imshow(NDVI, cmap="RdYlGn", vmin=-0.5, vmax=0.5); axes[1].set_title("Simulated NDVI")
axes[1].axis("off"); plt.colorbar(im,ax=axes[1],fraction=0.046)
axes[2].imshow(false_colour); axes[2].set_title("False Colour (NIR,R,G)"); axes[2].axis("off")
plt.tight_layout(); plt.savefig("remote.png",dpi=150); plt.show()` },

  { id:21, name:"Scene Text Recognition (OCR)", color:P.accent3, tagline:"Detect and read text in natural scene images",
    theory:`Scene Text Recognition involves two sub-tasks: text detection (locate text regions) and recognition (read characters). Detection: CTPN uses anchors for text line proposals; EAST directly regresses quadrilateral boxes; DB (Differentiable Binarisation) applies a learnable threshold to a probability map, enabling very fast inference. Recognition: CRNN (CNN feature extraction + BiLSTM sequence modeling + CTC decoding) is the standard baseline. Attention-based methods (ASTER, ABINet) replace CTC with encoder-decoder attention, handling irregular text. PARSeq is a recent permutation-based autoregressive model achieving state-of-the-art. TrOCR finetunes vision-language transformers (ViT + decoder) on synthetic and real data. PaddleOCR and EasyOCR package full pipelines. Large multimodal models (GPT-4V, Qwen-VL) perform end-to-end OCR on documents.`,
    architectures:["CTPN","EAST","DB++","CRNN","ASTER","ABINet","PARSeq","PaddleOCR","TrOCR","GOT-OCR"],
    metrics:["Word Accuracy","Sequence Accuracy","Normalised Edit Distance","H-mean (det)","E2E accuracy"],
    datasets:["IIIT5K","SVT","ICDAR 2013/2015","Total-Text","CTW1500","TextOCR","HierText","DocVQA"],
    colab:`# OCR with EasyOCR
# !pip install easyocr
import easyocr, urllib.request, numpy as np, cv2
import matplotlib.pyplot as plt, matplotlib.patches as patches

reader = easyocr.Reader(["en"], gpu=False)

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/License_plate_2.jpg/320px-License_plate_2.jpg"
raw = urllib.request.urlopen(url).read()
img = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = reader.readtext(img_rgb)
print(f"Detected {len(results)} text region(s):")

fig, ax = plt.subplots(figsize=(10,6))
ax.imshow(img_rgb)
for (bbox,text,conf) in results:
    pts = np.array(bbox, dtype=np.int32)
    ax.add_patch(plt.Polygon(pts,fill=False,edgecolor="lime",linewidth=2))
    ax.text(pts[0,0],pts[0,1]-5,f"{text} ({conf:.2f})",
            color="yellow",fontsize=11,fontweight="bold")
    print(f"  '{text}'  confidence={conf:.2f}")
ax.axis("off"); plt.title("EasyOCR Result"); plt.savefig("ocr.png",dpi=150); plt.show()` },

  { id:22, name:"Visual Question Answering", color:P.accent4, tagline:"Answer natural language questions about images",
    theory:`VQA requires joint understanding of an image and a natural language question to produce an answer. Early approaches concatenated CNN image features with LSTM question embeddings and classified over a fixed answer vocabulary. Bottom-Up Top-Down (ButD) attention used Faster R-CNN object region features as the visual set, applying question-guided attention over detected objects. Vision-language transformers (ViLBERT, UNITER, OSCAR) jointly pretrained on image-text pairs via masked language modeling, image-text matching, and region-word alignment. BLIP, BLIP-2, and InstructBLIP unified captioning, VQA, and retrieval. LLaVA frames VQA as visual instruction tuning of an LLM. VQA v2 controls for language priors by pairing complementary image pairs. GQA and A-OKVQA require compositional reasoning. ScienceQA tests multi-modal scientific understanding.`,
    architectures:["ButD Attention","ViLBERT","UNITER","OSCAR","BLIP","BLIP-2","InstructBLIP","LLaVA-1.5","Qwen-VL","GPT-4V"],
    metrics:["VQA Accuracy (normalised softmax)","BLEU","CIDEr","SPICE","Exact Match","Acc@VQAv2"],
    datasets:["VQA v2","GQA","A-OKVQA","ScienceQA","TextVQA","DocVQA","OK-VQA","MMMU"],
    colab:`# VQA with BLIP-2 (Salesforce)
# !pip install transformers accelerate bitsandbytes
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import urllib.request, io, torch

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model     = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b", torch_dtype=dtype).to(device)

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
img = Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB")

questions = [
    "Question: What animal is in the image? Answer:",
    "Question: What color is the animal? Answer:",
    "Question: Is the animal indoors or outdoors? Answer:",
    "Question: What emotion does the animal express? Answer:",
]
for q in questions:
    inp = processor(img, q, return_tensors="pt").to(device, dtype)
    out = model.generate(**inp, max_new_tokens=30)
    ans = processor.decode(out[0], skip_special_tokens=True)
    print(f"Q: {q.replace('Question: ','').replace(' Answer:','')}")
    print(f"A: {ans}\\n")` },

  { id:23, name:"Image Captioning", color:P.accent5, tagline:"Generate natural language descriptions of images",
    theory:`Image captioning generates natural language sentences describing image content. Show and Tell (Vinyals et al. 2015) encoded images with InceptionNet and decoded captions with an LSTM. Show, Attend and Tell added visual attention, focusing on relevant image regions at each decoding step. Dense captioning generates region-specific captions for multiple objects. Transformer-based models replaced LSTMs and attend over image patches or detected object feature grids. BLIP unified captioning, VQA, and retrieval with a mixture of three training objectives. GIT (Generative Image-to-text Transformer) pretrained a transformer decoder on 800M image-text pairs, achieving strong captioning and VQA performance. Modern large multimodal models (GPT-4V, Gemini) produce detailed, accurate captions including text reading, reasoning about relationships, counting, and cultural context. Evaluation: BLEU-4, METEOR, CIDEr (consensus-based), SPICE (scene graph), CLIPScore (semantic).`,
    architectures:["Show and Tell","Show Attend Tell","Dense Captioning","OSCAR","VinVL","BLIP","GIT","OFA","SimVLM","LLaVA-Caption"],
    metrics:["BLEU-4","METEOR","ROUGE-L","CIDEr","SPICE","CLIPScore"],
    datasets:["MS-COCO Captions (5 captions/image)","Flickr30K","NoCaps","Conceptual Captions 3M/12M","TextCaps","VizWiz"],
    colab:`# Image Captioning with BLIP (base model)
# !pip install transformers
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import urllib.request, io, torch

device    = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model     = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base").to(device)

urls_and_labels = [
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg", "dog"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg",    "cyclists"),
]
for url, label in urls_and_labels:
    img    = Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB")
    inputs = processor(img, return_tensors="pt").to(device)
    out    = model.generate(**inputs, max_new_tokens=60)
    cap    = processor.decode(out[0], skip_special_tokens=True)
    print(f"[{label}] Caption: {cap}")` },

  { id:24, name:"Anomaly Detection", color:P.accent6, tagline:"Identify defects and outliers without labeled anomaly data",
    theory:`Visual anomaly detection (VAD) identifies unusual patterns deviating from a learned normal distribution, typically trained on normal samples only (one-class learning). Applications: industrial surface defect inspection, medical lesion detection, video surveillance. Feature-embedding methods (PatchCore, PaDiM) extract CNN or ViT features from normal images, build a memory bank or Gaussian model per spatial position, and flag regions where test features deviate. PatchCore subsamples a coreset from the memory bank for efficient inference. Reconstruction-based methods (autoencoders, diffusion models) reconstruct normal images accurately but fail on anomalous regions. Student-teacher distillation trains a student to match a pretrained teacher on normal data; disagreement at test time signals anomalies. WinCLIP and APRIL-GAN use CLIP for zero-shot anomaly detection without any training on the target domain. MVTec AD (15 industrial categories) is the main benchmark.`,
    architectures:["PaDiM","PatchCore","CFlow-AD","FastFlow","SimpleNet","WinCLIP","APRIL-GAN","RD4AD","DiAD","UniFormaly"],
    metrics:["AUROC (image-level)","PRO (pixel-level region overlap)","AP","F1-max"],
    datasets:["MVTec AD (15 categories)","VisA (12 categories)","BTAD","Uni-Medical AD","DAGM","STC (video anomaly)","MVTec LOCO"],
    colab:`# Anomaly detection: PatchCore-style kNN approach from scratch
import numpy as np, cv2, urllib.request, matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

def extract_patches(gray, p=8, stride=4):
    H, W = gray.shape
    patches = []
    for y in range(0, H-p, stride):
        for x in range(0, W-p, stride):
            patches.append(gray[y:y+p, x:x+p].flatten().astype(float)/255.0)
    return np.array(patches)

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw = urllib.request.urlopen(url).read()
normal = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_GRAYSCALE)
normal = cv2.resize(normal,(160,120))

# "Train" on normal image patches
train_p = extract_patches(normal)
nbrs    = NearestNeighbors(n_neighbors=5).fit(train_p)

# Create anomalous test image (add white blob)
test = normal.copy()
cv2.circle(test,(60,50),20,255,-1)

test_p         = extract_patches(test)
dists, _       = nbrs.kneighbors(test_p)
scores         = dists.mean(axis=1)

# Reconstruct anomaly score map
p=8; stride=4; H,W=120,160
smap = np.zeros((H,W),float); cnt = np.zeros((H,W),float)
k=0
for y in range(0,H-p,stride):
    for x in range(0,W-p,stride):
        smap[y:y+p,x:x+p]+=scores[k]; cnt[y:y+p,x:x+p]+=1; k+=1
smap/=(cnt+1e-8)

fig,(a,b,c)=plt.subplots(1,3,figsize=(15,5))
a.imshow(normal,cmap="gray"); a.set_title("Normal"); a.axis("off")
b.imshow(test,cmap="gray"); b.set_title("Anomalous (blob added)"); b.axis("off")
im=c.imshow(smap,cmap="hot"); c.set_title("Anomaly Score Map"); c.axis("off")
plt.colorbar(im,ax=c,fraction=0.046)
plt.tight_layout(); plt.savefig("anomaly.png",dpi=150); plt.show()` },

  { id:25, name:"Image Retrieval", color:P.accent7, tagline:"Find visually similar images from a large database",
    theory:`Content-based image retrieval (CBIR) returns images from a database that are visually similar to a query. Early methods used hand-crafted global descriptors (color histograms, GIST) or local feature bags-of-words (SIFT + BoW + TF-IDF with inverted index). Deep metric learning trains CNN embeddings via contrastive or triplet loss. GeM pooling generalises max and average pooling with a learnable exponent p. DINOv2 features achieve remarkable retrieval performance without fine-tuning. CLIP embeddings trained on 400M image-text pairs generalise to retrieval across many domains. FAISS (Facebook AI Similarity Search) efficiently indexes millions of embeddings using IVF (Inverted File Index) and HNSW (Hierarchical Navigable Small World) graphs for sub-linear search. Re-ranking with local spatial verification further improves precision. Applications: e-commerce visual search, duplicate detection, person re-identification.`,
    architectures:["VLAD","Fisher Vector","NetVLAD","GeM","DELF","DOLG","GLDv2 baseline","CLIP+FAISS","DINOv2","CSD (Composed Retrieval)"],
    metrics:["mAP@100","Recall@1/5/10","NDCG","CMC (re-id)","P@1"],
    datasets:["Oxford5k/Paris6k","ROxford/RParis","GLDv2 (Google Landmarks)","CUB-200","In-Shop Clothes","SOP (Stanford Online Products)","Market-1501 (re-id)"],
    colab:`# Image Retrieval with CLIP embeddings + cosine similarity ranking
# !pip install transformers
import torch, urllib.request, io
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np, matplotlib.pyplot as plt

model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

urls = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_europeo4.jpg/320px-Gatto_europeo4.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Collage_of_Nine_Dogs.jpg/480px-Collage_of_Nine_Dogs.jpg",
]
labels = ["Dog","Cat","Cyclists","Dogs Collage"]
images = [Image.open(io.BytesIO(urllib.request.urlopen(u).read())).convert("RGB") for u in urls]

inputs = processor(images=images, return_tensors="pt", padding=True)
with torch.no_grad():
    embeds = model.get_image_features(**inputs)
embeds = embeds / embeds.norm(dim=-1, keepdim=True)

query = 0  # Query with the dog image
sims  = (embeds[query] @ embeds.T).numpy()
order = np.argsort(sims)[::-1]

print(f"Query: '{labels[query]}'  | Ranked results:")
for rank, i in enumerate(order):
    print(f"  Rank {rank+1}: {labels[i]:20s}  sim={sims[i]:.4f}")` },

  { id:26, name:"Video Object Tracking", color:P.accent8, tagline:"Follow objects across video frames with consistent identity",
    theory:`Visual tracking locates targets across frames. Single-object tracking (SOT) follows one target given its first-frame location. Multi-object tracking (MOT) maintains consistent IDs for all objects simultaneously. Siamese trackers (SiamFC, SiamRPN) learn cross-correlation between a template patch and search region in a shared embedding space. Transformer trackers (TransT, OSTrack) use attention to relate template and search features. MOT frameworks use detection + re-identification: SORT uses Kalman filter prediction and Hungarian algorithm for IoU-based assignment. DeepSORT adds appearance embeddings. ByteTrack retains low-confidence detections (from occlusion or motion blur) using a two-stage association strategy, significantly improving long-occlusion recovery. SAMURAI extends SAM2 with a Kalman filter for real-time single-object tracking. Benchmarks: LaSOT, TrackingNet (SOT); MOT17/20, DanceTrack (MOT).`,
    architectures:["SORT","DeepSORT","FairMOT","ByteTrack","BotSORT","SiamFC","SiamRPN++","TransT","OSTrack","SAMURAI"],
    metrics:["MOTA","IDF1","HOTA","AssA","DetA (MOT)","Success/Precision (SOT)"],
    datasets:["LaSOT","TrackingNet","GOT-10k","MOT17/20","DanceTrack","BDD100K Tracking","SportsMOT"],
    colab:`# Object Tracking with OpenCV CSRT tracker across simulated frames
import cv2, numpy as np, urllib.request, matplotlib.pyplot as plt

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
raw = urllib.request.urlopen(url).read()
frame = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_COLOR)

# Simulate 6-frame sequence with progressive motion
frames = [frame.copy()]
for i in range(1,6):
    M = np.float32([[1,0,i*12],[0,1,i*5]])
    frames.append(cv2.warpAffine(frame,M,(frame.shape[1],frame.shape[0])))

# Initialise CSRT tracker with a bounding box in first frame
tracker   = cv2.TrackerCSRT_create()
init_box  = (30,30,80,80)  # (x,y,w,h)
tracker.init(frames[0], init_box)

fig, axes = plt.subplots(2,3,figsize=(18,10))
for i,(fr,ax) in enumerate(zip(frames,axes.flat)):
    if i > 0:
        ok, box = tracker.update(fr)
    else:
        ok, box = True, init_box
    disp = fr.copy()
    if ok:
        x,y,w,h = map(int,box)
        cv2.rectangle(disp,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(disp,f"F{i} tracked",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    ax.imshow(cv2.cvtColor(disp,cv2.COLOR_BGR2RGB)); ax.set_title(f"Frame {i}"); ax.axis("off")
plt.tight_layout(); plt.savefig("tracking.png",dpi=150); plt.show()` },

  { id:27, name:"Autonomous Driving Perception", color:P.accent9, tagline:"Understand the driving scene for self-driving systems",
    theory:`Autonomous driving perception integrates multiple tasks: 3D object detection (pedestrians, vehicles, cyclists), lane detection, drivable area segmentation, depth estimation, and sensor fusion. Camera-based 3D detection lifts 2D features to Bird's Eye View (BEV) space: BEVDet uses predicted depth to voxelize image features; BEVFormer uses learnable BEV queries with deformable cross-attention to image feature maps, enabling temporal fusion. LiDAR-based detection (PointPillars, CenterPoint, VoxelNet) processes raw point clouds. Occupancy prediction outputs a 3D voxel grid of semantic labels (Tesla FSD, Occ3D). Lane detection with SCNN, LaneNet, or CLRNet. The nuScenes benchmark uses NDS (nuScenes Detection Score) combining detection accuracy and attribute prediction metrics. End-to-end driving models (UniAD, VAD) jointly predict all perception and planning outputs.`,
    architectures:["PointPillars","CenterPoint","BEVDet","BEVFormer","DETR3D","PETRv2","UniAD","SparseDrive","VAD","DriveX"],
    metrics:["mAP 3D","NDS","mATE/mASE/mAOE","Lane F1","mIoU (seg)","CDS (completion)"],
    datasets:["nuScenes (camera+LiDAR)","KITTI","Waymo Open Dataset","Argoverse 2","BDD100K","Cityscapes","OpenLane"],
    colab:`# Lane detection with Canny + Hough Transform (classical baseline)
import cv2, numpy as np, urllib.request, matplotlib.pyplot as plt

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Aco_alocilja_road.jpg/320px-Aco_alocilja_road.jpg"
raw = urllib.request.urlopen(url).read()
img = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_COLOR)
h, w = img.shape[:2]

gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur  = cv2.GaussianBlur(gray,(5,5),0)
edges = cv2.Canny(blur, 50, 150)

# Region of interest mask (trapezoidal lower area)
mask  = np.zeros_like(edges)
poly  = np.array([[(0,h),(w,h),(int(w*0.6),int(h*0.55)),(int(w*0.4),int(h*0.55))]], np.int32)
cv2.fillPoly(mask,poly,255)
roi   = cv2.bitwise_and(edges,mask)

lines = cv2.HoughLinesP(roi,1,np.pi/180,50,minLineLength=80,maxLineGap=30)
vis   = img.copy()
if lines is not None:
    for l in lines: cv2.line(vis,(l[0,0],l[0,1]),(l[0,2],l[0,3]),(0,255,0),3)
    print(f"Detected {len(lines)} line segments")

fig,(a,b,c) = plt.subplots(1,3,figsize=(18,6))
a.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)); a.set_title("Input"); a.axis("off")
b.imshow(roi,cmap="gray"); b.set_title("ROI Edges"); b.axis("off")
c.imshow(cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)); c.set_title("Lanes (Hough)"); c.axis("off")
plt.tight_layout(); plt.savefig("lane.png",dpi=150); plt.show()` },

  { id:28, name:"Document Understanding", color:P.accent1, tagline:"Parse structure and content from complex document images",
    theory:`Document understanding goes beyond OCR to comprehend layout, structure, and semantics of forms, invoices, receipts, scientific papers, and tables. Key tasks: document classification, layout analysis (detect text/figure/table regions), table structure recognition, key-value extraction, and document VQA. LayoutLM jointly encodes text tokens with their 2D bounding box positions, enabling understanding of document structure. LayoutLMv3 adds image patch embeddings in a unified multimodal transformer. Donut (Document Understanding Transformer) removes OCR entirely: a Swin Transformer encoder directly feeds a BART decoder that generates structured JSON. Nougat converts academic PDFs to markdown using a similar architecture. PP-Structure (PaddleOCR) detects and structures tables. GPT-4V and Gemini handle multi-page documents with complex layouts directly from images.`,
    architectures:["LayoutLM v1/v2/v3","Donut","Nougat","PaddleOCR PP-Structure","DocFormer","UDOP","DiT","Florence-2","GPT-4V","Gemini"],
    metrics:["F1 (KV extraction)","ANLS (DocVQA)","TEDS (table edit distance)","Accuracy (classification)","NED"],
    datasets:["DocVQA","FUNSD (forms)","SROIE (receipts)","PubLayNet","DocLayNet","WTQ (table QA)","RVL-CDIP (classification)","Kleister"],
    colab:`# Document layout analysis: detect text vs image regions via morphology
import cv2, numpy as np, urllib.request, matplotlib.pyplot as plt

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/Invoice_template.jpg/320px-Invoice_template.jpg"
raw = urllib.request.urlopen(url).read()
img = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_COLOR)
if img is None:
    # Fallback: any document-like image
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
    raw = urllib.request.urlopen(url).read()
    img = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold + morphological closing to find text blocks
_, bw      = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
kernel_h   = cv2.getStructuringElement(cv2.MORPH_RECT,(30,1))   # horizontal dilation
kernel_v   = cv2.getStructuringElement(cv2.MORPH_RECT,(1,20))   # vertical dilation
dilated    = cv2.dilate(bw, kernel_h)
dilated    = cv2.dilate(dilated, kernel_v)
contours,_ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

vis = img.copy()
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    if w*h > 500:  # filter noise
        cv2.rectangle(vis,(x,y),(x+w,y+h),(0,200,0),2)
print(f"Found {len(contours)} layout blocks")

fig,(a,b)=plt.subplots(1,2,figsize=(14,8))
a.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)); a.set_title("Original Document"); a.axis("off")
b.imshow(cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)); b.set_title("Detected Text Blocks"); b.axis("off")
plt.tight_layout(); plt.savefig("doc.png",dpi=150); plt.show()` },

  { id:29, name:"Point Cloud Processing", color:P.accent2, tagline:"Learn from 3D LiDAR and RGB-D point cloud data",
    theory:`Point clouds are unordered sets of 3D points (x,y,z), optionally with colour, intensity, or normal attributes. They are captured by LiDAR, RGB-D cameras, or 3D reconstruction. The core challenge: irregular structure (no grid), variable density, permutation invariance, and rotation equivariance requirements. PointNet (Qi et al. 2017) processes raw point sets with shared MLP applied per point and symmetric max-pooling for global aggregation, achieving permutation invariance by design. PointNet++ adds hierarchical local feature learning with farthest point sampling and ball query grouping. DGCNN constructs dynamic local graphs in feature space using k-nearest neighbors, applying EdgeConv. KPConv uses continuous convolution kernels at specific kernel points. Transformer-based methods (Point Transformer v2/v3) apply self-attention among neighbours. Applications: autonomous driving LiDAR, robotic manipulation, scene reconstruction.`,
    architectures:["PointNet","PointNet++","DGCNN","KPConv","VoxNet","SECOND","PointTransformer v2/v3","MinkowskiEngine","3DETR","PointBERT"],
    metrics:["Overall Accuracy (classification)","mIoU (part/scene seg)","Chamfer Distance","F-score@threshold","3D AP"],
    datasets:["ModelNet40","ShapeNet Parts","ScanNet (scene seg)","S3DIS","KITTI LiDAR","nuScenes LiDAR","ScanObjectNN"],
    colab:`# Point cloud visualisation and basic processing with Open3D
# !pip install open3d
import open3d as o3d, numpy as np, matplotlib.pyplot as plt

# Generate a synthetic labelled point cloud
rng   = np.random.default_rng(42)
n     = 2000

# Sphere
theta = rng.uniform(0,2*np.pi,n); phi = rng.uniform(0,np.pi,n)
sphere = np.stack([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)],axis=1)

# Box
box = rng.uniform(-0.5,0.5,(500,3)) + np.array([2.5,0,0])

pts    = np.vstack([sphere,box])
colors = np.vstack([np.tile([0.2,0.6,1.0],(n,1)), np.tile([1.0,0.4,0.2],(500,1))])

# Basic statistics
print(f"Total points: {len(pts)}")
print(f"Bounding box:  min={pts.min(axis=0).round(2)}, max={pts.max(axis=0).round(2)}")
print(f"Centroid:      {pts.mean(axis=0).round(3)}")

# Open3D kNN (nearest-neighbour radius) for normal estimation
pcd = o3d.geometry.PointCloud()
pcd.points  = o3d.utility.Vector3dVector(pts)
pcd.colors  = o3d.utility.Vector3dVector(colors)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3,max_nn=30))

# 3D scatter via matplotlib
fig = plt.figure(figsize=(10,8))
ax  = fig.add_subplot(111,projection="3d")
ax.scatter(pts[:,0],pts[:,1],pts[:,2],c=colors,s=1.5,alpha=0.7)
ax.set_title("Synthetic Point Cloud (Sphere + Box)"); plt.tight_layout()
plt.savefig("pointcloud.png",dpi=150); plt.show()` },

  { id:30, name:"Explainable AI in Vision", color:P.accent3, tagline:"Visualise and understand what neural networks actually learn",
    theory:`Explainable AI (XAI) for vision provides post-hoc explanations of model decisions. Gradient-based methods: Vanilla Gradients backpropagate the class score gradient to the input; Grad-CAM uses global-average-pooled gradients in the final conv layer to produce a coarse class activation map; Guided Backpropagation backpropagates only positive gradients through ReLU; Guided Grad-CAM combines both for sharp, class-discriminative visualisations. Integrated Gradients accumulates gradients along a straight path from baseline (black) to input, satisfying axiomatic properties. SHAP (SHapley Additive exPlanations) assigns Shapley values to superpixels using a game-theoretic framework. LIME fits local linear models to perturbed neighbourhood samples. RISE randomly masks input regions and correlates masks with output changes. Attention visualisation inspects ViT self-attention maps. These methods are critical for medical AI, legal accountability, and debugging bias.`,
    architectures:["Vanilla Gradients","Saliency Maps","Grad-CAM","Grad-CAM++","Guided Grad-CAM","Integrated Gradients","SHAP","LIME","RISE","Attention Rollout"],
    metrics:["Faithfulness","Pointing Game (localisation)","Deletion/Insertion AUC","Sensitivity","Completeness"],
    datasets:["ImageNet (XAI benchmarks)","PASCAL VOC (localisation)","CelebA (attribute XAI)","ISIC (medical XAI)","BDD-X (driving explanations)"],
    colab:`# Grad-CAM from scratch with PyTorch hooks
import torch, urllib.request, io, numpy as np, cv2
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2); model.eval()

# Hooks to capture forward feature maps and backward gradients
features, grads = [], []
h1 = model.layer4.register_forward_hook(lambda m,i,o: features.append(o.detach()))
h2 = model.layer4.register_full_backward_hook(lambda m,i,o: grads.append(o[0].detach()))

tf  = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
img_pil = Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB")
x   = tf(img_pil).unsqueeze(0).requires_grad_(True)

out = model(x); cls = out.argmax(1).item()
LABELS = urllib.request.urlopen(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
).read().decode().splitlines()
print(f"Predicted: {LABELS[cls]} (class {cls})")

model.zero_grad(); out[0,cls].backward()

weights = grads[0].mean(dim=[2,3],keepdim=True)
cam     = (weights * features[0]).sum(1).squeeze().relu().numpy()
cam     = (cam - cam.min()) / (cam.max()-cam.min()+1e-8)
cam_r   = cv2.resize(cam,(224,224))
hm      = cv2.applyColorMap((cam_r*255).astype(np.uint8), cv2.COLORMAP_JET)
img224  = np.array(img_pil.resize((224,224)))
overlay = (img224*0.55 + cv2.cvtColor(hm,cv2.COLOR_BGR2RGB)*0.45).astype(np.uint8)

h1.remove(); h2.remove()
fig,(a,b,c) = plt.subplots(1,3,figsize=(15,5))
a.imshow(img_pil); a.set_title("Input"); a.axis("off")
b.imshow(cam_r,cmap="jet"); b.set_title("Grad-CAM"); b.axis("off")
c.imshow(overlay); c.set_title(f"Overlay ({LABELS[cls]})"); c.axis("off")
plt.tight_layout(); plt.savefig("gradcam.png",dpi=150); plt.show()` },
];

/* ================================================================
   10 LEARNING MODULES
================================================================ */
const MODULES = [
  { id:0, title:"What Is Computer Vision?", level:"Absolute Beginner", time:"25 min", color:P.accent1,
    sections:[
      { heading:"Definition and Scope", body:`Computer vision (CV) trains computers to interpret visual information: images, video, and 3D scans. It sits at the intersection of mathematics (linear algebra, calculus, probability), computer science (algorithms, deep learning), and optics (how cameras form images). Applications span medicine (AI dermatology), autonomous vehicles (pedestrian detection), manufacturing (defect inspection), and smartphones (real-time translation of street signs via camera).` },
      { heading:"A Brief History", body:`CV began in the 1960s with block-world scene description. The 1980-90s brought classic algorithms: Canny edge detection (1986), SIFT descriptors (David Lowe, 1999), and Viola-Jones face detector (2001). The pivotal moment: AlexNet (2012) won ImageNet with 15.3% top-5 error, more than 10 points better than hand-crafted methods, proving that deep CNNs trained on GPUs with large datasets could dominate. Since then: ResNet (2015), ViT (2020), CLIP and SAM (2021-2023), Stable Diffusion and Sora for video generation.` },
      { heading:"How Computers See", body:`A digital image is a 2D grid of pixels. Grayscale: each pixel is 0-255. Colour (RGB): each pixel is three values (Red, Green, Blue), each 0-255. A 640x480 RGB image is a 3D array of shape (480, 640, 3) containing 921,600 numbers. Every CV operation, from edge detection to face recognition, is a mathematical function on these numbers. Neural networks learn which transformations are useful by adjusting millions of parameters via examples.` },
      { heading:"The CV Pipeline", body:`Most CV systems follow: (1) Image acquisition (camera, scanner). (2) Preprocessing (resize, normalise, denoise). (3) Feature extraction (edges, textures, semantic objects). (4) Model inference (classification, detection, segmentation). (5) Post-processing (NMS, filtering). (6) Decision or output (trigger alarm, move robot, display label). Understanding each stage helps debug failures: poor camera dynamic range, bad normalisation, or model overfitting.` },
    ],
    code:`# Your first image: load, inspect, and manipulate pixels
import urllib.request, io, numpy as np
from PIL import Image
import matplotlib.pyplot as plt

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
img = Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB")
img_np = np.array(img)

print(f"Shape:  {img_np.shape}  (H, W, C)")
print(f"Dtype:  {img_np.dtype}")
print(f"Range:  {img_np.min()} to {img_np.max()}")
print(f"Pixel at (100,100): R={img_np[100,100,0]} G={img_np[100,100,1]} B={img_np[100,100,2]}")

# Grayscale conversion (luminance formula)
gray = (0.299*img_np[:,:,0] + 0.587*img_np[:,:,1] + 0.114*img_np[:,:,2]).astype(np.uint8)

# Basic manipulations
flipped    = img_np[:, ::-1, :]
cropped    = img_np[50:200, 50:250, :]
brightened = np.clip(img_np.astype(int)+60, 0, 255).astype(np.uint8)
darkened   = np.clip(img_np.astype(int)-60, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, (im, t) in zip(axes.flat,
    [(img_np,"Original RGB"),(gray,"Grayscale"),(flipped,"Flipped"),
     (cropped,"Cropped"),(brightened,"Brightened"),(darkened,"Darkened")]):
    ax.imshow(im, cmap="gray" if im.ndim==2 else None); ax.set_title(t); ax.axis("off")
plt.suptitle("Module 0: Introduction to Images", fontsize=14)
plt.tight_layout(); plt.savefig("m0.png", dpi=150); plt.show()` },

  { id:1, title:"Image Preprocessing and Classical Features", level:"Beginner", time:"40 min", color:P.accent2,
    sections:[
      { heading:"Why Preprocessing Matters", body:`Raw images from cameras are messy: inconsistent sizes, brightness differences from lighting, sensor noise, and irrelevant background. Preprocessing standardises inputs so models learn effectively. Common steps: resize to fixed dimensions; normalise pixels from [0,255] to [0,1] or subtract dataset mean and divide by std (ImageNet mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]); convert colour spaces (BGR to RGB, RGB to LAB for colour-invariant ops); and correct geometric distortions via camera calibration matrices.` },
      { heading:"Filtering and Convolution", body:`Spatial filtering applies a kernel (small weight matrix) to every pixel neighbourhood: out[i,j] = sum(kernel * neighbourhood). A Gaussian kernel blurs, smoothing noise. A Sobel kernel computes gradients, detecting edges. This is exactly what convolutional layers in CNNs do, except CNNs learn the kernel weights from data. Understanding filtering is therefore essential for understanding deep learning: every convolutional layer is a learned bank of spatial filters.` },
      { heading:"Classical Feature Descriptors", body:`Before deep learning, CV relied on hand-crafted descriptors. SIFT: detects keypoints at multiple scales via Difference-of-Gaussian, describes each with a 128-dim gradient orientation histogram, invariant to scale and rotation. HOG: divides image into cells, computes gradient histograms per cell, normalises across blocks; excellent for pedestrian detection (used in classic DPM detectors). ORB: binary descriptor via oriented FAST keypoints and BRIEF descriptor; extremely fast for real-time SLAM. These remain relevant for embedded systems and robotics where deep learning is too costly.` },
      { heading:"Morphological Operations", body:`Morphological ops process binary/grayscale images using a structuring element shape. Erosion shrinks bright regions (removes small noise). Dilation expands bright regions (fills holes). Opening = erosion then dilation: removes small bright specks. Closing = dilation then erosion: fills small dark holes. Top-hat = image minus opening: isolates fine bright details. These are fundamental in document analysis, medical imaging preprocessing, and detection pipelines.` },
    ],
    code:`# Classical CV preprocessing and feature extraction
import cv2, numpy as np, urllib.request
import matplotlib.pyplot as plt

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Bill_Nye_2017.jpg/240px-Bill_Nye_2017.jpg"
raw = urllib.request.urlopen(url).read()
img = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Filtering
gauss   = cv2.GaussianBlur(gray,(7,7),0)
sobel_x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
sobel_y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
mag     = np.sqrt(sobel_x**2+sobel_y**2); mag/=mag.max()
canny   = cv2.Canny(gray,80,200)

# SIFT keypoints
sift     = cv2.SIFT_create(nfeatures=80)
kps,desc = sift.detectAndCompute(gray,None)
sift_vis = cv2.drawKeypoints(img_rgb,kps,None,
               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print(f"SIFT: {len(kps)} keypoints, descriptor shape: {desc.shape}")

# Morphology
kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
_,bw    = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
eroded  = cv2.erode(bw,kernel); dilated = cv2.dilate(bw,kernel)
tophat  = cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,kernel)

fig,axes=plt.subplots(3,3,figsize=(15,12))
for ax,(im,t) in zip(axes.flat,[
    (img_rgb,"Original"),(gauss,"Gaussian Blur"),(mag,"Sobel Edges"),
    (canny,"Canny"),(sift_vis,"SIFT Keypoints"),(tophat,"Top-Hat"),
    (gray,"Grayscale"),(eroded,"Eroded"),(dilated,"Dilated")]):
    ax.imshow(im,cmap="gray" if im.ndim==2 else None); ax.set_title(t); ax.axis("off")
plt.tight_layout(); plt.savefig("m1.png",dpi=150); plt.show()` },

  { id:2, title:"CNNs from First Principles", level:"Beginner-Intermediate", time:"60 min", color:P.accent3,
    sections:[
      { heading:"The Convolution Operation", body:`A 2D convolution slides kernel K over input X: output[i,j,f] = sum(K[:,:,:,f] * X[i:i+k, j:j+k, :]). Each filter f produces one output channel. Weight sharing (same kernel at every spatial position) gives two key properties: translation equivariance and dramatic parameter efficiency vs fully connected layers. A 3x3 conv on 224x224x3 input with 64 filters has only 3*3*3*64+64=1,792 parameters vs billions for a fully connected layer of the same span.` },
      { heading:"Activation Functions and Normalisation", body:`After convolution, a non-linearity is applied element-wise. ReLU: max(0,x); fast, no vanishing gradient for positive activations. Leaky ReLU: x if x>0 else 0.01x. GELU: x*Phi(x); smooth, used in transformers. Batch Normalisation normalises each feature map across the batch to zero mean and unit variance, then applies learnable scale (gamma) and shift (beta). Benefits: faster convergence, acts as regulariser, allows higher learning rates.` },
      { heading:"Backpropagation and Training", body:`Training minimises cross-entropy loss: -sum(y_c * log(p_c)). Backpropagation computes gradients of loss w.r.t. every parameter using the chain rule, propagating output to input. Adam optimiser maintains per-parameter adaptive learning rates using first (mean) and second (variance) moment estimates of gradients. Cosine annealing with warm restarts and linear warmup are standard LR schedules. Data augmentation (random crop, flip, colour jitter, mixup) prevents overfitting.` },
      { heading:"ResNet and Modern Architectures", body:`ResNet (He et al. 2015) introduced skip connections: output = F(x) + x, learning a residual instead of the full mapping. This solved vanishing gradients, enabling 100+ layer networks. EfficientNet uses compound scaling (simultaneously scale width, depth, and resolution by fixed ratios). MobileNet uses depthwise separable convolutions: depthwise (per-channel spatial filter) then pointwise (1x1 cross-channel mixing), reducing parameters by ~9x vs standard convolutions at similar accuracy.` },
    ],
    code:`# Build and train a CNN on CIFAR-10
import torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as T
import matplotlib.pyplot as plt

tf_train = T.Compose([T.RandomCrop(32,padding=4),T.RandomHorizontalFlip(),T.ToTensor(),
    T.Normalize([0.4914,0.4822,0.4465],[0.247,0.243,0.261])])
tf_test  = T.Compose([T.ToTensor(),T.Normalize([0.4914,0.4822,0.4465],[0.247,0.243,0.261])])
trainset = torchvision.datasets.CIFAR10("/tmp/c10",train=True, download=True,transform=tf_train)
testset  = torchvision.datasets.CIFAR10("/tmp/c10",train=False,download=True,transform=tf_test)
loader_tr= torch.utils.data.DataLoader(trainset,128,shuffle=True,num_workers=2)
loader_te= torch.utils.data.DataLoader(testset,256,shuffle=False,num_workers=2)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),
            nn.MaxPool2d(2),nn.Dropout2d(0.25),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.MaxPool2d(2),nn.Dropout2d(0.25))
        self.head=nn.Sequential(
            nn.Flatten(),nn.Linear(128*8*8,256),nn.BatchNorm1d(256),nn.ReLU(),
            nn.Dropout(0.5),nn.Linear(256,10))
    def forward(self,x): return self.head(self.features(x))

device="cuda" if torch.cuda.is_available() else "cpu"
model=SmallCNN().to(device)
opt=optim.Adam(model.parameters(),lr=1e-3)
sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=15)
crit=nn.CrossEntropyLoss()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

def run_epoch(loader,train):
    model.train(train); tot=cor=ls=0
    with torch.set_grad_enabled(train):
        for x,y in loader:
            x,y=x.to(device),y.to(device); out=model(x); loss=crit(out,y)
            if train: opt.zero_grad(); loss.backward(); opt.step()
            ls+=loss.item()*len(y); cor+=(out.argmax(1)==y).sum().item(); tot+=len(y)
    return ls/tot, cor/tot*100

tr_acc=[]; te_acc=[]
for ep in range(15):
    _,ta=run_epoch(loader_tr,True); _,va=run_epoch(loader_te,False)
    sched.step(); tr_acc.append(ta); te_acc.append(va)
    print(f"Ep {ep+1:2d} | train {ta:.1f}%  test {va:.1f}%")

plt.figure(figsize=(8,5))
plt.plot(tr_acc,'o-',label="Train"); plt.plot(te_acc,'s-',label="Test")
plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend(); plt.grid(alpha=0.3)
plt.title("CNN on CIFAR-10"); plt.tight_layout(); plt.savefig("m2.png",dpi=150); plt.show()` },

  { id:3, title:"Object Detection: Anchors to Transformers", level:"Intermediate", time:"55 min", color:P.accent4,
    sections:[
      { heading:"Detection Problem Formulation", body:`Detection outputs tuples (class, confidence, x1,y1,x2,y2). Training matches ground-truth boxes to anchor boxes by IoU. IoU > 0.5 = positive; IoU < 0.3 = negative. Loss = classification (focal/cross-entropy) + regression (smooth-L1 or CIoU). At inference, NMS removes duplicates: sort by confidence, keep highest, suppress remaining boxes with IoU > threshold vs the kept box. Repeat until empty.` },
      { heading:"Two-Stage: Faster R-CNN and FPN", body:`Faster R-CNN: backbone CNN extracts features; Region Proposal Network (RPN) predicts objectness + offsets for anchors at multiple scales; RoI head applies RoIAlign then classifies and refines each proposal. Feature Pyramid Network (FPN) adds top-down lateral connections, producing multi-scale feature maps for detecting objects of all sizes. Cascade R-CNN chains heads at IoU thresholds 0.5, 0.6, 0.7 for progressive box refinement. High accuracy (55+ AP on COCO) but 5-10 FPS.` },
      { heading:"One-Stage: YOLO Family", body:`YOLO predicts all boxes and classes in a single forward pass. YOLOv8 uses CSP (Cross Stage Partial) backbone with FPN-PAN neck, outputting predictions at 3 scales (8, 16, 32-pixel strides). Each grid cell predicts box offsets, class probabilities, and objectness. TAL (Task-Aligned Learning) replaces IoU-based matching with a unified quality metric. YOLOv8n runs 80+ FPS on GPU; YOLOv8x achieves 53.9% mAP on COCO.` },
      { heading:"Transformer Detectors: DETR and DINO", body:`DETR eliminates anchors and NMS: transformer encoder processes CNN features; N learnable object queries attend via cross-attention in decoder; each query predicts one box. Hungarian matching assigns ground-truth bijectively to predictions. Deformable DETR speeds convergence by sampling sparse feature points. DINO-DETR adds contrastive denoising (noisy GT boxes as positive queries), achieving state-of-the-art with faster convergence.` },
    ],
    code:`# NMS from scratch + YOLOv8 detection pipeline
import torch, numpy as np, urllib.request, io, cv2
from PIL import Image
import matplotlib.pyplot as plt, matplotlib.patches as patches

# NMS implementation
def nms(boxes, scores, iou_thresh=0.5):
    x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas=(x2-x1)*(y2-y1); order=scores.argsort()[::-1]; keep=[]
    while len(order):
        i=order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]]); yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]); yy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        iou=inter/(areas[i]+areas[order[1:]]-inter+1e-7)
        order=order[1:][iou<=iou_thresh]
    return keep

boxes=np.array([[10,10,100,100],[15,15,105,105],[50,50,150,150],[200,200,300,300]],float)
scores=np.array([0.95,0.80,0.88,0.91])
kept=nms(boxes,scores,0.5)
print(f"After NMS: kept indices {kept}, scores {scores[kept].tolist()}")

# IoU calculation
def compute_iou(box1,box2):
    xi1=max(box1[0],box2[0]); yi1=max(box1[1],box2[1])
    xi2=min(box1[2],box2[2]); yi2=min(box1[3],box2[3])
    inter=max(0,xi2-xi1)*max(0,yi2-yi1)
    a1=(box1[2]-box1[0])*(box1[3]-box1[1]); a2=(box2[2]-box2[0])*(box2[3]-box2[1])
    return inter/(a1+a2-inter+1e-7)

print(f"IoU boxes 0,1: {compute_iou(boxes[0],boxes[1]):.3f}")
print(f"IoU boxes 0,2: {compute_iou(boxes[0],boxes[2]):.3f}")

# YOLOv8 (requires ultralytics)
try:
    from ultralytics import YOLO
    model=YOLO("yolov8n.pt")
    url="https://ultralytics.com/images/bus.jpg"
    img=Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB")
    r=model(img)[0]
    fig,ax=plt.subplots(figsize=(12,8)); ax.imshow(np.array(img))
    colors=plt.cm.tab20(np.linspace(0,1,80))
    for box in r.boxes:
        x1,y1,x2,y2=box.xyxy[0].tolist(); cls=int(box.cls[0]); conf=float(box.conf[0])
        c=colors[cls%20]; rect=patches.Rectangle((x1,y1),x2-x1,y2-y1,lw=2,edgecolor=c[:3],facecolor="none")
        ax.add_patch(rect); ax.text(x1,y1-5,f"{model.names[cls]} {conf:.0%}",color="white",fontsize=8,
            bbox=dict(facecolor=c[:3],alpha=0.8,pad=1))
    ax.axis("off"); ax.set_title(f"YOLOv8n: {len(r.boxes)} objects")
    plt.tight_layout(); plt.savefig("m3.png",dpi=150); plt.show()
except ImportError: print("pip install ultralytics")` },

  { id:4, title:"Segmentation: Pixel-Level Understanding", level:"Intermediate", time:"50 min", color:P.accent5,
    sections:[
      { heading:"Semantic vs Instance vs Panoptic", body:`Semantic: every pixel gets a class label, no instance distinction (all cars = one colour). Instance: each object instance gets its own mask (two cars = two distinct masks). Panoptic: merges both: countable things (people, vehicles) get instance IDs; uncountable stuff (sky, road) gets semantic labels only. Each needs different output formats and losses: semantic uses per-pixel cross-entropy; instance uses binary cross-entropy or Dice loss per mask; panoptic uses Panoptic Quality metric combining recognition and segmentation quality.` },
      { heading:"U-Net Architecture", body:`U-Net (Ronneberger 2015) is the gold standard for medical and scientific segmentation. Encoder progressively downsamples extracting features at multiple scales. Decoder upsamples via transposed convolutions and concatenates skip connections from matching encoder levels, preserving fine spatial details lost during downsampling. This symmetric design with skip connections enables precise boundary localisation even with small training sets. nnU-Net auto-configures all hyperparameters and has won dozens of medical segmentation challenges.` },
      { heading:"Dilated Convolutions and ASPP", body:`Standard convolutions grow receptive field linearly with depth. Dilated convolutions insert zeros between kernel weights (rate r), expanding receptive field to k+(k-1)(r-1) without extra parameters or resolution loss. ASPP (Atrous Spatial Pyramid Pooling, DeepLab) applies parallel dilated convolutions with rates 6,12,18 plus global average pooling, concatenating for multi-scale context. SAM (Segment Anything Model) enables zero-shot segmentation of any object given point, box, or text prompts, trained on 1.1B masks.` },
      { heading:"Loss Functions for Segmentation", body:`Cross-entropy loss: -sum(y*log(p)); simple but unstable with extreme class imbalance (tiny lesions vs large background). Dice loss: 1 - 2*|A intersection B|/(|A|+|B|); directly optimises overlap; robust to imbalance. Focal loss: -(1-p)^gamma * log(p); down-weights easy negatives, focuses training on hard pixels. Tversky loss: generalises Dice with separate weights for false positives and false negatives, useful when false negatives are more costly (medical). Combination = Dice + Cross-Entropy is the de facto standard for medical segmentation.` },
    ],
    code:`# U-Net from scratch + segmentation metrics
import torch, torch.nn as nn, numpy as np, urllib.request, cv2
import matplotlib.pyplot as plt

class DoubleConv(nn.Module):
    def __init__(self,i,o):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True),
                               nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True))
    def forward(self,x): return self.net(x)

class UNet(nn.Module):
    def __init__(self,in_c=3,out_c=2,base=32):
        super().__init__()
        b=base
        self.e1=DoubleConv(in_c,b); self.e2=DoubleConv(b,b*2)
        self.e3=DoubleConv(b*2,b*4); self.e4=DoubleConv(b*4,b*8)
        self.bn=DoubleConv(b*8,b*16); self.pool=nn.MaxPool2d(2)
        self.up4=nn.ConvTranspose2d(b*16,b*8,2,2); self.d4=DoubleConv(b*16,b*8)
        self.up3=nn.ConvTranspose2d(b*8, b*4,2,2); self.d3=DoubleConv(b*8, b*4)
        self.up2=nn.ConvTranspose2d(b*4, b*2,2,2); self.d2=DoubleConv(b*4, b*2)
        self.up1=nn.ConvTranspose2d(b*2, b,  2,2); self.d1=DoubleConv(b*2, b)
        self.out=nn.Conv2d(b,out_c,1)
    def forward(self,x):
        s1=self.e1(x); s2=self.e2(self.pool(s1)); s3=self.e3(self.pool(s2)); s4=self.e4(self.pool(s3))
        b=self.bn(self.pool(s4))
        d=self.d4(torch.cat([self.up4(b),s4],1)); d=self.d3(torch.cat([self.up3(d),s3],1))
        d=self.d2(torch.cat([self.up2(d),s2],1)); d=self.d1(torch.cat([self.up1(d),s1],1))
        return self.out(d)

model=UNet(); device="cuda" if torch.cuda.is_available() else "cpu"; model=model.to(device)
print(f"U-Net parameters: {sum(p.numel() for p in model.parameters()):,}")
x=torch.randn(2,3,256,256).to(device); out=model(x)
print(f"Input: {x.shape}  ->  Output: {out.shape}")

# Dice score function
def dice_score(pred, target, eps=1e-6):
    pred=pred.float(); target=target.float()
    inter=(pred*target).sum(); return (2*inter+eps)/(pred.sum()+target.sum()+eps)

# Synthetic segmentation demo
url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
raw=urllib.request.urlopen(url).read()
img=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
_,mask_otsu=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
contours,_=cv2.findContours(mask_otsu,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
vis=img_rgb.copy(); cv2.drawContours(vis,contours,-1,(0,255,0),2)
fig,(a,b,c)=plt.subplots(1,3,figsize=(15,5))
a.imshow(img_rgb); a.set_title("Input"); a.axis("off")
b.imshow(mask_otsu,cmap="gray"); b.set_title("Otsu Mask"); b.axis("off")
c.imshow(vis); c.set_title("Contours Overlay"); c.axis("off")
plt.tight_layout(); plt.savefig("m4.png",dpi=150); plt.show()` },

  { id:5, title:"Image Restoration: Denoising to Diffusion", level:"Intermediate", time:"45 min", color:P.accent6,
    sections:[
      { heading:"The Ill-Posed Inverse Problem", body:`Restoration solves y=Hx+n (y=observed, H=degradation, x=clean, n=noise) for x. Ill-posed because H may not be invertible and many x produce the same y. Classical methods used Total Variation regularisation or BM3D (block-matching 3D collaborative filtering). Deep learning reframes this as supervised regression: collect paired (degraded, clean) data, train network to map from degraded to clean. The residual learning trick (predict the noise, not the clean image) accelerates training significantly.` },
      { heading:"DnCNN: Residual Denoising CNN", body:`DnCNN predicts the noise residual n: output = input - predicted_noise. Architecture: 17-layer CNN with batch normalisation throughout. Achieves state-of-the-art Gaussian denoising at any single noise level. FFDNet extends to blind denoising by taking the noise-level map as an additional input. Noise2Void trains on single noisy images without clean pairs, using blind-spot convolutions where the centre pixel is masked to prevent the network from learning the identity function.` },
      { heading:"Restormer: Transformer Restoration", body:`Standard self-attention is O(N^2) in spatial positions, prohibitive for high-res images. Restormer uses Transposed Multi-Head Self-Attention (TMSA): attention computed across the channel dimension (small, resolution-independent) rather than spatial positions. Gated-Dconv Feed-Forward Networks add depthwise convolutions for local context. Hierarchical encoder-decoder with progressive learning. Achieves SOTA across Gaussian denoising, deraining, deblurring, and dehazing with one architecture.` },
      { heading:"Diffusion-Based Restoration", body:`Diffusion models frame restoration as conditional generation: given degraded y, generate clean x consistent with y. SR3 (Image Restoration via Repeated Refinement) uses conditional DDPM where noisy x_t is denoised conditioned on the degraded y. Key advantage over discriminative methods: generates diverse high-frequency textures rather than blurry averages, producing perceptually superior results especially at high upscaling factors or heavy degradation. Drawback: 100-1000x slower than a single CNN forward pass.` },
    ],
    code:`# DnCNN residual denoiser trained from scratch
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, urllib.request, cv2, matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

class DnCNN(nn.Module):
    def __init__(self,depth=8,ch=32):
        super().__init__()
        ly=[nn.Conv2d(1,ch,3,padding=1),nn.ReLU(True)]
        for _ in range(depth-2):
            ly+=[nn.Conv2d(ch,ch,3,padding=1),nn.BatchNorm2d(ch),nn.ReLU(True)]
        ly+=[nn.Conv2d(ch,1,3,padding=1)]
        self.net=nn.Sequential(*ly)
    def forward(self,x): return x-self.net(x)  # residual: input minus predicted noise

device="cuda" if torch.cuda.is_available() else "cpu"
model=DnCNN().to(device)
opt=optim.Adam(model.parameters(),lr=1e-3); crit=nn.MSELoss()
print(f"DnCNN parameters: {sum(p.numel() for p in model.parameters()):,}")

url="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw=urllib.request.urlopen(url).read()
clean=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
clean=cv2.resize(clean,(256,256)).astype(np.float32)/255.0

def make_batch(c,n=32,p=64):
    rng=np.random.default_rng(); pc=[]; pn=[]
    for _ in range(n):
        sig=rng.uniform(5,50)/255.0; r,c2=rng.integers(0,c.shape[0]-p,2)
        patch=c[r:r+p,c2:c2+p]; noisy=np.clip(patch+rng.normal(0,sig,(p,p)).astype(np.float32),0,1)
        pc.append(patch[None,None]); pn.append(noisy[None,None])
    return torch.from_numpy(np.concatenate(pn,0)), torch.from_numpy(np.concatenate(pc,0))

for step in range(300):
    nb,cb=make_batch(clean); nb,cb=nb.to(device),cb.to(device)
    loss=crit(model(nb),cb); opt.zero_grad(); loss.backward(); opt.step()
    if (step+1)%100==0: print(f"Step {step+1}  loss={loss.item():.5f}")

model.eval(); rng2=np.random.default_rng(0); sig=25/255.0
noisy=np.clip(clean+rng2.normal(0,sig,clean.shape).astype(np.float32),0,1)
with torch.no_grad():
    pred=model(torch.from_numpy(noisy[None,None]).to(device)).squeeze().cpu().numpy().clip(0,1)
print(f"PSNR noisy:    {psnr_fn(clean,noisy,data_range=1):.2f} dB")
print(f"PSNR denoised: {psnr_fn(clean,pred,data_range=1):.2f} dB")

fig,axes=plt.subplots(1,3,figsize=(15,5))
for ax,im,t in zip(axes,[clean,noisy,pred],["Clean","Noisy (sigma=25)","DnCNN denoised"]):
    ax.imshow(im,cmap="gray"); ax.set_title(t); ax.axis("off")
plt.tight_layout(); plt.savefig("m5.png",dpi=150); plt.show()` },

  { id:6, title:"Vision Transformers and CLIP", level:"Advanced-Intermediate", time:"55 min", color:P.accent7,
    sections:[
      { heading:"Self-Attention for Images", body:`ViT divides 224x224 images into 16x16 patches -> 196 tokens. Each patch is linearly projected to D dimensions. A learnable [CLS] token is prepended. Positional embeddings are added. The sequence goes through L transformer blocks: LayerNorm -> Multi-Head Self-Attention -> residual -> LayerNorm -> FFN -> residual. MHA computes Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V, where Q,K,V are linear projections of the token sequence. The [CLS] token's final embedding is classified. ViT requires large-scale pretraining to outperform CNNs; with JFT-300M it sets new records.` },
      { heading:"Efficient Variants: Swin and DeiT", body:`Swin Transformer uses shifted window attention: attention within non-overlapping 7x7 windows (linear O(N) complexity), with window shift between layers for cross-window communication. Hierarchical features enable detection and segmentation backbones. DeiT adds knowledge distillation from a CNN teacher using a distillation token, achieving competitive ImageNet results with only 1.2M training images. SegFormer uses overlapping patch merging and a lightweight all-MLP decoder for efficient segmentation.` },
      { heading:"CLIP: Contrastive Language-Image Pretraining", body:`CLIP trains a vision encoder and text encoder jointly on 400M (image, text) pairs via InfoNCE contrastive loss: within each batch of N pairs, maximise the N correct (image, text) similarities while minimising the N^2-N incorrect ones. Creates a shared visual-language embedding space. Zero-shot classification: encode text descriptions of each class, find the closest text embedding to the image embedding. Achieves 76% top-1 on ImageNet zero-shot (matching ResNet-50) and generalises across 27 benchmarks.` },
      { heading:"DINOv2: Self-Supervised Visual Features", body:`DINOv2 trains a ViT-g on 142M curated images using DINO + iBOT self-distillation objectives, without any labels. The student ViT is trained to match a momentum teacher ViT given differently augmented views. Key insight: DINO ViT attention heads spontaneously learn to segment objects without segmentation supervision. DINOv2 features enable depth estimation, segmentation, retrieval, and classification simply by attaching small linear probes, with no fine-tuning of the backbone.` },
    ],
    code:`# ViT from scratch + CLIP zero-shot classification
import torch, torch.nn as nn, urllib.request, io
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np, matplotlib.pyplot as plt

class PatchEmbed(nn.Module):
    def __init__(self,img=224,patch=16,in_c=3,dim=192):
        super().__init__()
        self.n=(img//patch)**2; self.proj=nn.Conv2d(in_c,dim,patch,stride=patch)
    def forward(self,x): return self.proj(x).flatten(2).transpose(1,2)

class MHA(nn.Module):
    def __init__(self,dim,heads=6):
        super().__init__(); self.h=heads; self.sc=dim**-0.5
        self.qkv=nn.Linear(dim,dim*3); self.out=nn.Linear(dim,dim)
    def forward(self,x):
        B,N,D=x.shape; h=self.h
        qkv=self.qkv(x).reshape(B,N,3,h,D//h).permute(2,0,3,1,4)
        q,k,v=qkv.unbind(0)
        att=(q@k.transpose(-2,-1))*self.sc
        return self.out((att.softmax(-1)@v).transpose(1,2).reshape(B,N,D))

class Block(nn.Module):
    def __init__(self,dim,heads=6,mlp=4):
        super().__init__()
        self.n1=nn.LayerNorm(dim); self.attn=MHA(dim,heads)
        self.n2=nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.Linear(dim,dim*mlp),nn.GELU(),nn.Linear(dim*mlp,dim))
    def forward(self,x): return x+self.ff(self.n2(x+self.attn(self.n1(x))))

class ViT(nn.Module):
    def __init__(self,img=224,patch=16,nc=10,dim=192,depth=6,heads=6):
        super().__init__()
        self.pe=PatchEmbed(img,patch,3,dim); N=self.pe.n
        self.cls=nn.Parameter(torch.zeros(1,1,dim))
        self.pos=nn.Parameter(torch.randn(1,N+1,dim)*0.02)
        self.blocks=nn.Sequential(*[Block(dim,heads) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim); self.head=nn.Linear(dim,nc)
    def forward(self,x):
        B=x.shape[0]; x=self.pe(x)
        x=torch.cat([self.cls.expand(B,-1,-1),x],1)+self.pos
        return self.head(self.norm(self.blocks(x))[:,0])

vit=ViT(); x=torch.randn(2,3,224,224); out=vit(x)
print(f"ViT output: {out.shape}  params: {sum(p.numel() for p in vit.parameters()):,}")

# CLIP zero-shot
try:
    model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    proc=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
    img=Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB")
    candidates=["a photo of a dog","a photo of a cat","a photo of a car","a photo of a bird"]
    inp=proc(text=candidates,images=img,return_tensors="pt",padding=True)
    with torch.no_grad(): probs=model(**inp).logits_per_image.softmax(-1).squeeze()
    for c,p in sorted(zip(candidates,probs.tolist()),key=lambda x:-x[1]):
        print(f"  {c:40s} {p*100:.1f}%")
except ImportError: print("pip install transformers")` },

  { id:7, title:"Generative Vision: GANs, VAEs, Diffusion", level:"Advanced", time:"60 min", color:P.accent8,
    sections:[
      { heading:"Variational Autoencoders", body:`A VAE encodes images into a distribution q(z|x)=N(mu(x), sigma^2(x)) in a low-dimensional latent space, then decodes z back to images. Reparameterisation trick: z = mu + eps*sigma (eps~N(0,I)) enables backpropagation through sampling. ELBO loss = reconstruction loss + beta*KL(q(z|x)||N(0,I)). The KL term regularises the latent space to be smooth and continuous, enabling interpolation between points. VQ-VAE replaces continuous latent with a discrete codebook, producing image tokens used by DALL-E and Stable Diffusion.` },
      { heading:"GAN Training Dynamics", body:`GAN minimax: min_G max_D E[log D(x)] + E[log(1-D(G(z)))]. Failure modes: mode collapse (G produces few outputs); discriminator winning too easily (vanishing gradients for G). Solutions: WGAN-GP uses Wasserstein distance + gradient penalty; Progressive GAN grows both G and D from 4x4 to 1024x1024; spectral normalisation keeps Lipschitz constraint; minibatch discrimination exposes G to batch statistics. StyleGAN v2/3 produces photorealistic 1024x1024 faces with a disentangled latent space enabling semantic editing.` },
      { heading:"Diffusion Process Mathematics", body:`Forward: x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps, eps~N(0,I). Network epsilon_theta is trained to predict the noise eps given x_t and timestep t. Reverse: x_{t-1} = (1/sqrt(alpha_t))*(x_t - beta_t/sqrt(1-alpha_bar_t)*eps_theta(x_t,t)) + sigma_t*z. DDIM uses deterministic sampling trajectories, enabling 10-50 step generation. Classifier-free guidance: eps_guided = (1+w)*eps_cond - w*eps_uncond, with w controlling text adherence strength.` },
      { heading:"Latent Diffusion and ControlNet", body:`Latent Diffusion compresses images 8x or 64x with VQ-VAE, running diffusion in latent space to reduce computation dramatically. CLIP text encoder conditions the U-Net via cross-attention. ControlNet adds spatial conditioning (edges, depth, pose, seg maps) via duplicate U-Net encoder with zero-convolution adapters, enabling precise layout control without disrupting pretrained weights. DiT (Diffusion Transformer) replaces the U-Net with a transformer using AdaLN (adaptive layer norm) for timestep/class conditioning, scaling better with parameters and data.` },
    ],
    code:`# VAE on MNIST from scratch
import torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as T
import matplotlib.pyplot as plt, numpy as np

class VAE(nn.Module):
    def __init__(self,lat=16):
        super().__init__()
        self.enc=nn.Sequential(nn.Flatten(),nn.Linear(784,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU())
        self.mu=nn.Linear(256,lat); self.lv=nn.Linear(256,lat)
        self.dec=nn.Sequential(nn.Linear(lat,256),nn.ReLU(),nn.Linear(256,512),nn.ReLU(),
                               nn.Linear(512,784),nn.Sigmoid())
    def reparameterise(self,mu,lv): return mu+torch.randn_like(mu)*torch.exp(0.5*lv)
    def forward(self,x):
        h=self.enc(x); mu,lv=self.mu(h),self.lv(h); z=self.reparameterise(mu,lv)
        return self.dec(z),mu,lv

def vae_loss(recon,x,mu,lv,beta=0.5):
    bce=nn.functional.binary_cross_entropy(recon,x.view(-1,784),reduction="sum")
    kld=-0.5*torch.sum(1+lv-mu**2-lv.exp())
    return (bce+beta*kld)/x.shape[0]

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=VAE(16).to(device); opt=optim.Adam(model.parameters(),lr=1e-3)
loader=torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("/tmp/mn",True,download=True,transform=T.ToTensor()),
    batch_size=256,shuffle=True)

for ep in range(6):
    tot=0
    for x,_ in loader:
        x=x.to(device); r,mu,lv=model(x); l=vae_loss(r,x,mu,lv)
        opt.zero_grad(); l.backward(); opt.step(); tot+=l.item()
    print(f"Epoch {ep+1}  loss={tot/len(loader):.2f}")

model.eval()
with torch.no_grad():
    z=torch.randn(64,16).to(device); imgs=model.dec(z).view(64,1,28,28).cpu()
fig,axes=plt.subplots(8,8,figsize=(8,8))
for ax,im in zip(axes.flat,imgs): ax.imshow(im.squeeze(),cmap="gray"); ax.axis("off")
plt.suptitle("VAE Generated Digits"); plt.tight_layout(); plt.savefig("m7.png",dpi=150); plt.show()` },

  { id:8, title:"3D Vision: NeRF and Gaussian Splatting", level:"Advanced", time:"60 min", color:P.accent9,
    sections:[
      { heading:"3D Representations", body:`Voxel grids: regular 3D grid, easy to process but O(N^3) memory. Point clouds: unordered 3D points, memory-efficient, directly from LiDAR but lack connectivity. Meshes: vertices + edges + faces, compact and graphics-compatible but hard to optimise end-to-end. Implicit representations (NeRF, SDF): continuous functions evaluated anywhere, flexible but expensive. 3D Gaussian Splatting: explicit Gaussians with position, covariance, opacity, colour; fast rasterisation and differentiable.` },
      { heading:"PointNet: Learning on Unordered Sets", body:`PointNet applies a shared MLP independently to each point (permutation equivariant), then global max-pooling produces a permutation-invariant feature (the same result regardless of point order). T-Net predicts a 3x3 rotation matrix to canonicalise alignment. PointNet++ adds hierarchical local feature learning via farthest point sampling and ball query grouping, analogous to CNN receptive fields. DGCNN dynamically constructs local k-NN graphs in feature space, applying EdgeConv for relational learning.` },
      { heading:"Neural Radiance Fields (NeRF)", body:`NeRF represents a scene as F_theta: (x,y,z,theta,phi) -> (r,g,b,sigma). Volume rendering: C = sum_i T_i*(1-exp(-sigma_i*delta_i))*c_i, where T_i is accumulated transmittance. Positional encoding (sinusoidal functions) enables MLP to represent high-frequency detail. Limitations: hours of training, seconds per image inference. Instant-NGP uses multi-resolution hash encoding: training in seconds. TensoRF uses CP/VM tensor decomposition for compact scenes. Zip-NeRF combines hash encoding with mip-NeRF anti-aliasing for large-scale scenes.` },
      { heading:"3D Gaussian Splatting", body:`3DGS represents scenes as millions of 3D Gaussians: position mu, covariance Sigma (from scale s and quaternion rotation q), opacity alpha, and spherical harmonics for view-dependent colour. Rendering: sort by depth, project each Gaussian to 2D, alpha-composite front-to-back. Training: optimise Gaussian params via photometric gradient descent, plus adaptive density control (split large Gaussians, remove transparent ones). Results: 30-minute training on consumer GPU, 100+ FPS rendering at 1080p. Beats NeRF in speed while matching quality.` },
    ],
    code:`# PointNet architecture + NeRF volume rendering concept
import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PointNet(nn.Module):
    def __init__(self,nc=40):
        super().__init__()
        self.shared=nn.Sequential(
            nn.Conv1d(3,64,1),nn.BatchNorm1d(64),nn.ReLU(),
            nn.Conv1d(64,128,1),nn.BatchNorm1d(128),nn.ReLU(),
            nn.Conv1d(128,1024,1),nn.BatchNorm1d(1024),nn.ReLU())
        self.fc=nn.Sequential(
            nn.Linear(1024,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(512,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(256,nc))
    def forward(self,x): return self.fc(self.shared(x).max(2)[0])

pn=PointNet(); x=torch.randn(4,3,1024); print(f"PointNet out: {pn(x).shape}")
print(f"Params: {sum(p.numel() for p in pn.parameters()):,}")

# Minimal NeRF forward pass concept
class TinyNeRF(nn.Module):
    def __init__(self,pos_enc=6):
        super().__init__()
        in_dim=3+3*2*pos_enc  # (x,y,z) + sin/cos positional encoding
        self.net=nn.Sequential(
            nn.Linear(in_dim,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),
            nn.Linear(256,256),nn.ReLU(),nn.Linear(256,4))  # -> (r,g,b,sigma)
        self.L=pos_enc
    def pos_encode(self,x):
        freqs=2**torch.arange(self.L,dtype=x.dtype)
        enc=[x]+[f(x*freq*np.pi) for freq in freqs for f in [torch.sin,torch.cos]]
        return torch.cat(enc,-1)
    def forward(self,xyz): return self.net(self.pos_encode(xyz))

nerf=TinyNeRF(); pts=torch.randn(100,3); out=nerf(pts)
print(f"TinyNeRF out: {out.shape}  (N, 4=[rgb, density])")

# Volume rendering
def volume_render(rgb, sigma, deltas):
    alpha=1-torch.exp(-sigma*deltas)
    T=torch.cumprod(torch.cat([torch.ones(alpha.shape[0],1),1-alpha+1e-10],-1),-1)[:,:-1]
    weights=(T*alpha); return (weights.unsqueeze(-1)*rgb).sum(1)

N_rays,N_samples=16,64
rgb=torch.rand(N_rays,N_samples,3); sigma=torch.relu(torch.randn(N_rays,N_samples))
deltas=torch.full((N_rays,N_samples),0.02)
rendered_rgb=volume_render(rgb,sigma,deltas)
print(f"Rendered colour: {rendered_rgb.shape}  (N_rays, 3)")

# Visualise synthetic point cloud
rng=np.random.default_rng(0)
t=rng.uniform(0,2*np.pi,2000); p=rng.uniform(0,np.pi,2000)
pts_np=np.stack([np.sin(p)*np.cos(t),np.sin(p)*np.sin(t),np.cos(p)],1)
col=plt.cm.cool((pts_np[:,2]+1)/2)
fig=plt.figure(figsize=(8,8)); ax=fig.add_subplot(111,projection="3d")
ax.scatter(pts_np[:,0],pts_np[:,1],pts_np[:,2],c=col,s=2,alpha=0.6)
ax.set_title("Unit Sphere Point Cloud"); plt.tight_layout()
plt.savefig("m8.png",dpi=150); plt.show()` },

  { id:9, title:"Deployment: From Research to Production", level:"Advanced", time:"50 min", color:P.accent1,
    sections:[
      { heading:"The Research-to-Production Gap", body:`A model achieving 95% benchmark accuracy may perform poorly in production due to: distribution shift (real images differ from training data); latency constraints (autonomous driving needs <50ms, cloud APIs tolerate 500ms); hardware constraints on edge devices; reliability requirements (medical AI must be safe on out-of-distribution inputs); and explainability demands from regulators. Bridging this gap requires real-world dataset curation, systematic evaluation beyond accuracy (calibration, fairness, edge cases), model compression, and monitoring infrastructure.` },
      { heading:"Model Compression Techniques", body:`Quantisation: reduce float32 to int8 or int4, shrinking model 4-8x and accelerating inference (integer ops faster on most hardware). Post-training quantisation (PTQ) needs only a calibration dataset. Quantisation-aware training (QAT) simulates quantisation during training for minimal accuracy loss. Pruning: remove weights or channels below a threshold; structured pruning (removing full channels) is hardware-friendly. Knowledge distillation: train small student to mimic large teacher's soft logits, transferring rich inter-class knowledge beyond hard labels.` },
      { heading:"Export: ONNX, TensorRT, TFLite", body:`Export pipeline: PyTorch model -> torch.onnx.export() -> onnx.checker -> runtime optimisation. ONNX Runtime (ORT) provides vendor-neutral efficient inference across CPUs, GPUs, and accelerators. TensorRT (NVIDIA) applies layer fusion, precision calibration (FP32->FP16/INT8), and generates an optimised engine for NVIDIA GPUs, typically achieving 2-5x speedup over PyTorch. TFLite targets mobile and embedded Linux. CoreML targets Apple devices. OpenVINO optimises for Intel CPUs and VPUs.` },
      { heading:"MLOps and Monitoring", body:`After deployment, performance silently degrades due to data drift or concept drift. A robust pipeline includes: input monitoring (flag inputs far from training distribution); output monitoring (track prediction distribution shifts); human-in-the-loop review for uncertain predictions; automated retraining triggers; A/B testing for safe rollout; model versioning and reproducible training (MLflow, DVC, Weights and Biases); and canary deployments routing small traffic percentage to new models. Tools: Triton Inference Server, TorchServe, Ray Serve, BentoML.` },
    ],
    code:`# Model compression and benchmarking pipeline
import torch, torch.nn as nn, torchvision.models as models
import numpy as np, time, urllib.request, io, os
from PIL import Image
from torchvision import transforms

model_fp32=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model_fp32.eval()
print(f"MobileNetV3-Small: {sum(p.numel() for p in model_fp32.parameters())/1e6:.2f}M params")

# Post-training quantisation
model_int8=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model_int8.eval()
model_int8.qconfig=torch.quantization.get_default_qconfig("fbgemm")
model_int8=torch.quantization.prepare(model_int8)

tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
img=Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB")
calib=tf(img).unsqueeze(0).repeat(8,1,1,1)
with torch.no_grad(): model_int8(calib)
model_int8=torch.quantization.convert(model_int8)

# ONNX export
dummy=torch.randn(1,3,224,224)
torch.onnx.export(model_fp32,dummy,"/tmp/mv3.onnx",
    input_names=["image"],output_names=["logits"],
    dynamic_axes={"image":{0:"batch"}},opset_version=13)
print("ONNX exported")

# Latency benchmark
def bench(m,x,n=200,warmup=20):
    with torch.no_grad():
        for _ in range(warmup): m(x)
        t=time.perf_counter()
        for _ in range(n): m(x)
        return (time.perf_counter()-t)/n*1000

x=tf(img).unsqueeze(0)
fp=bench(model_fp32,x); i8=bench(model_int8,x)
print(f"FP32: {fp:.2f} ms  ({1000/fp:.0f} FPS)")
print(f"INT8: {i8:.2f} ms  ({1000/i8:.0f} FPS)  Speedup: {fp/i8:.2f}x")

torch.save(model_fp32.state_dict(),"/tmp/fp32.pt")
torch.save(model_int8.state_dict(),"/tmp/int8.pt")
s32=os.path.getsize("/tmp/fp32.pt")/1e6; s8=os.path.getsize("/tmp/int8.pt")/1e6
print(f"FP32 size: {s32:.1f} MB  |  INT8 size: {s8:.1f} MB  ({s8/s32*100:.0f}%)")

LABELS=urllib.request.urlopen(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
).read().decode().splitlines()
with torch.no_grad(): probs=model_fp32(x).softmax(1).squeeze()
for p,i in zip(*probs.topk(3)):
    print(f"  {LABELS[i]:35s} {p.item()*100:.2f}%")` },
];
/* ==========================================================
   100 CODING CHALLENGES
   ========================================================== */
const ALL_CHALLENGES = [
  {
    "id": "C01",
    "difficulty": "Easy",
    "tag": "Arrays / Pixels",
    "points": 10,
    "title": "Flip and Invert Binary Image",
    "company": "Google, Meta",
    "desc": "Given a binary image (2D list of 0s and 1s), horizontally flip each row, then invert every bit (0->1, 1->0). This simulates a common preprocessing step in binary document image pipelines.",
    "example": "Input: [[1,1,0],[1,0,1],[0,0,0]]\nOutput: [[1,0,0],[0,1,0],[1,1,1]]",
    "hint": "Reverse each row with [::-1], then XOR every element with 1. Both steps combine into a single list comprehension.",
    "fullSolution": "# Flip and Invert Binary Image\nimport urllib.request, numpy as np, cv2\nimport matplotlib.pyplot as plt\n\ndef flipAndInvert(image):\n    return [[x^1 for x in row[::-1]] for row in image]\n\n# Test on example\nexample = [[1,1,0],[1,0,1],[0,0,0]]\nresult = flipAndInvert([row[:] for row in example])\nassert result == [[1,0,0],[0,1,0],[1,1,1]], f'Got {result}'\nprint('Example passed:', result)\n\n# Apply to a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/License_plate_2.jpg/320px-License_plate_2.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_GRAYSCALE)\nimg = cv2.resize(img,(160,80))\n_, binary = cv2.threshold(img,127,1,cv2.THRESH_BINARY)\nresult_np = np.array(flipAndInvert(binary.tolist()),dtype=np.uint8)\n\nfig,(a,b,c) = plt.subplots(1,3,figsize=(15,5))\na.imshow(binary,cmap='gray',vmin=0,vmax=1); a.set_title('Original Binary'); a.axis('off')\nb.imshow(binary[:,::-1],cmap='gray'); b.set_title('After H-Flip'); b.axis('off')\nc.imshow(result_np,cmap='gray',vmin=0,vmax=1); c.set_title('Flip+Invert'); c.axis('off')\nplt.tight_layout(); plt.savefig('c01.png',dpi=150); plt.show()",
    "complexity": "Time O(m*n)  Space O(m*n) for output",
    "followup": "How would you extend this to an RGB image where invert means 255-pixel?"
  },
  {
    "id": "C02",
    "difficulty": "Easy",
    "tag": "2D Prefix Sum",
    "points": 10,
    "title": "Count Black Pixels in Rectangles",
    "company": "Amazon, Bloomberg",
    "desc": "Given a binary image matrix and rectangle queries [r1,c1,r2,c2], return the count of 1s inside each rectangle in O(1) per query after O(m*n) preprocessing. This is the Integral Image technique used in Haar Cascade face detection.",
    "example": "Matrix: [[1,0,1],[0,1,0],[1,1,1]]  Query [0,0,2,2] -> 6",
    "hint": "Build a 2D prefix sum table. Answer = pre[r2+1][c2+1] - pre[r1][c2+1] - pre[r2+1][c1] + pre[r1][c1]",
    "fullSolution": "# 2D Prefix Sum (Integral Image)\nimport numpy as np, urllib.request, cv2\nimport matplotlib.pyplot as plt, time\n\ndef build_prefix(mat):\n    m,n=len(mat),len(mat[0])\n    pre=[[0]*(n+1) for _ in range(m+1)]\n    for i in range(m):\n        for j in range(n):\n            pre[i+1][j+1]=mat[i][j]+pre[i][j+1]+pre[i+1][j]-pre[i][j]\n    return pre\n\ndef query(pre,r1,c1,r2,c2):\n    return pre[r2+1][c2+1]-pre[r1][c2+1]-pre[r2+1][c1]+pre[r1][c1]\n\nmat=[[1,0,1],[0,1,0],[1,1,1]]\npre=build_prefix(mat)\nprint('Query [0,0,2,2]:', query(pre,0,0,2,2))  # expected 6\nassert query(pre,0,0,2,2)==6",
    "complexity": "O(m*n) build + O(1) per query  Space O(m*n)",
    "followup": "This is the Viola-Jones Integral Image. How is it used to compute Haar features in constant time?"
  },
  {
    "id": "C03",
    "difficulty": "Easy",
    "tag": "BFS",
    "points": 10,
    "title": "Flood Fill (Paint Bucket)",
    "company": "Amazon, Google",
    "desc": "Implement the flood-fill algorithm (paint bucket tool in image editors). Given a binary/colour image, a starting pixel (sr,sc), and a new colour, repaint all 4-connected pixels of the same original colour.",
    "example": "image=[[1,1,1],[1,1,0],[1,0,1]] sr=1 sc=1 color=2 -> [[2,2,2],[2,2,0],[2,0,1]]",
    "hint": "BFS or DFS from (sr,sc). Store the original colour. Do not revisit already-painted pixels.",
    "fullSolution": "# Flood Fill with BFS\nimport numpy as np, cv2, urllib.request\nimport matplotlib.pyplot as plt\nfrom collections import deque\n\ndef flood_fill(image, sr, sc, new_color):\n    orig = image[sr][sc]\n    if orig == new_color: return image\n    m,n = len(image),len(image[0])\n    q = deque([(sr,sc)]); image[sr][sc] = new_color\n    while q:\n        r,c = q.popleft()\n        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:\n            nr,nc = r+dr,c+dc\n            if 0<=nr<m and 0<=nc<n and image[nr][nc]==orig:\n                image[nr][nc] = new_color; q.append((nr,nc))\n    return image\n\nimg = [[1,1,1],[1,1,0],[1,0,1]]\nresult = flood_fill([row[:] for row in img],1,1,2)\nassert result == [[2,2,2],[2,2,0],[2,0,1]], f'Got {result}'\nprint('Passed:', result)",
    "complexity": "Time O(m*n)  Space O(m*n) for queue",
    "followup": "How would you handle 8-connectivity (diagonals)? How does this relate to connected component labelling?"
  },
  {
    "id": "C04",
    "difficulty": "Easy",
    "tag": "Convolution",
    "points": 10,
    "title": "Image Smoother (3x3 Mean Filter)",
    "company": "Microsoft, Apple",
    "desc": "Apply a 3x3 mean filter to a grayscale image. Each output pixel is the floor of the average of all valid neighbours, excluding out-of-bounds cells. Implement WITHOUT NumPy convolution helpers.",
    "example": "Input: [[1,1,1],[1,0,1],[1,1,1]]\nAll output pixels floor to 0 (border cells average fewer neighbours).",
    "hint": "For each cell, gather all valid neighbours within the 3x3 window (clip to image bounds), then compute floor(mean).",
    "fullSolution": "# 3x3 Mean Filter from scratch\nimport math, numpy as np, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\ndef mean_filter_3x3(img):\n    m,n = len(img),len(img[0])\n    out = [[0]*n for _ in range(m)]\n    for i in range(m):\n        for j in range(n):\n            nbrs = [img[i+di][j+dj] for di in range(-1,2) for dj in range(-1,2)\n                    if 0<=i+di<m and 0<=j+dj<n]\n            out[i][j] = math.floor(sum(nbrs)/len(nbrs))\n    return out\n\nexample = [[1,1,1],[1,0,1],[1,1,1]]\nprint('Result:', mean_filter_3x3(example))\n\nurl='https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw=urllib.request.urlopen(url).read()\nimg=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)\nimg=cv2.resize(img,(160,120))\nsmooth=np.array(mean_filter_3x3(img.tolist()),dtype=np.uint8)\ncv_smooth=cv2.blur(img,(3,3))\nprint('Max diff from cv2:', np.abs(smooth.astype(int)-cv_smooth.astype(int)).max())",
    "complexity": "Time O(m*n*9)=O(m*n)  Space O(m*n)",
    "followup": "What is the difference between mean and Gaussian filtering? When would you prefer each?"
  },
  {
    "id": "C05",
    "difficulty": "Easy",
    "tag": "BFS / Connected Components",
    "points": 10,
    "title": "Count Connected Components",
    "company": "Facebook, LinkedIn",
    "desc": "Given a binary image, count the number of connected components (regions of 1s connected by 4-connectivity). Fundamental to object counting in satellite imagery, cell counting in microscopy, and blob detection.",
    "example": "[[1,1,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]] -> 2 components",
    "hint": "BFS/DFS from each unvisited 1. Increment a component counter. Mark all reachable pixels as visited before moving to the next unvisited 1.",
    "fullSolution": "# Connected Components via BFS\nimport numpy as np, cv2, urllib.request\nimport matplotlib.pyplot as plt\nfrom collections import deque\n\ndef count_cc(grid):\n    m,n=len(grid),len(grid[0])\n    visited=[[False]*n for _ in range(m)]\n    labels=[[0]*n for _ in range(m)]\n    count=0\n    for i in range(m):\n        for j in range(n):\n            if grid[i][j]==1 and not visited[i][j]:\n                count+=1; q=deque([(i,j)]); visited[i][j]=True; labels[i][j]=count\n                while q:\n                    r,c=q.popleft()\n                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:\n                        nr,nc=r+dr,c+dc\n                        if 0<=nr<m and 0<=nc<n and grid[nr][nc]==1 and not visited[nr][nc]:\n                            visited[nr][nc]=True; labels[nr][nc]=count; q.append((nr,nc))\n    return count, labels\n\ngrid=[[1,1,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]]\ncount,labels=count_cc(grid)\nassert count==2, f'Expected 2, got {count}'\nprint('Components:', count)\nprint('Labels:', labels)",
    "complexity": "Time O(m*n)  Space O(m*n)",
    "followup": "How would you adapt this to 8-connectivity? What changes in the neighbour iteration?"
  },
  {
    "id": "C06",
    "difficulty": "Easy",
    "tag": "Histogram",
    "points": 10,
    "title": "Histogram Equalisation from Scratch",
    "company": "Adobe, Qualcomm",
    "desc": "Implement global histogram equalisation on a grayscale image without using cv2.equalizeHist(). This enhances contrast by spreading intensity values uniformly across [0,255].",
    "example": "Dark image concentrated around low intensities -> equalised image with uniform histogram.",
    "hint": "Compute histogram, compute CDF, normalise CDF to [0,255], use as lookup table.",
    "fullSolution": "# Histogram Equalisation\nimport numpy as np, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\ndef hist_eq(img):\n    hist=np.zeros(256,dtype=int)\n    for v in img.flatten(): hist[v]+=1\n    cdf=np.cumsum(hist)\n    cdf_min=cdf[cdf>0].min()\n    lut=np.round((cdf-cdf_min)/(img.size-cdf_min)*255).astype(np.uint8)\n    return lut[img]\n\nurl='https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg'\nraw=urllib.request.urlopen(url).read()\nimg=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)\ndark=np.clip(img//4,0,255).astype(np.uint8)\nour_eq=hist_eq(dark); cv2_eq=cv2.equalizeHist(dark)\nprint('Max diff from cv2:', np.abs(our_eq.astype(int)-cv2_eq.astype(int)).max())\nfig,axes=plt.subplots(1,3,figsize=(12,4))\nfor ax,im,t in zip(axes,[dark,our_eq,cv2_eq],['Dark','Our HE','cv2 HE']):\n    ax.imshow(im,cmap='gray'); ax.set_title(t); ax.axis('off')\nplt.tight_layout(); plt.show()",
    "complexity": "Time O(m*n + 256)  Space O(256) for LUT",
    "followup": "What is CLAHE (Contrast Limited Adaptive Histogram Equalisation) and why is it preferred for medical images?"
  },
  {
    "id": "C07",
    "difficulty": "Easy",
    "tag": "Morphology",
    "points": 10,
    "title": "Erosion and Dilation from Scratch",
    "company": "Qualcomm, Texas Instruments",
    "desc": "Implement binary erosion and dilation from scratch. Erosion shrinks bright regions; dilation expands them. These are the building blocks of all morphological operations.",
    "example": "Erosion removes pixels where the 3x3 structuring element extends outside the object boundary.",
    "hint": "Erosion: output=1 only if ALL kernel-covered pixels in input are 1. Dilation: output=1 if ANY kernel-covered pixel is 1.",
    "fullSolution": "# Erosion and Dilation from scratch\nimport numpy as np, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\ndef erode(img,k):\n    kh,kw=k.shape; ph,pw=kh//2,kw//2; m,n=img.shape\n    out=np.zeros_like(img)\n    for i in range(m):\n        for j in range(n):\n            ok=True\n            for di in range(kh):\n                for dj in range(kw):\n                    if k[di,dj]==0: continue\n                    ni,nj=i+di-ph,j+dj-pw\n                    if not(0<=ni<m and 0<=nj<n) or img[ni,nj]==0: ok=False; break\n                if not ok: break\n            out[i,j]=1 if ok else 0\n    return out\n\ndef dilate(img,k):\n    kh,kw=k.shape; ph,pw=kh//2,kw//2; m,n=img.shape\n    out=np.zeros_like(img)\n    for i in range(m):\n        for j in range(n):\n            for di in range(kh):\n                for dj in range(kw):\n                    if k[di,dj]==0: continue\n                    ni,nj=i+di-ph,j+dj-pw\n                    if 0<=ni<m and 0<=nj<n and img[ni,nj]==1: out[i,j]=1; break\n                if out[i,j]: break\n    return out\n\nurl='https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw=urllib.request.urlopen(url).read()\nimg=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)\nimg=cv2.resize(img,(80,60))\n_,bw=cv2.threshold(img,127,1,cv2.THRESH_BINARY)\nk3=np.ones((3,3),np.uint8)\ner=erode(bw,k3); di=dilate(bw,k3)\nprint('Erosion matches cv2:', np.array_equal(er,cv2.erode(bw.astype(np.uint8),k3)))\nprint('Dilation matches cv2:', np.array_equal(di,cv2.dilate(bw.astype(np.uint8),k3)))",
    "complexity": "Time O(m*n*k^2)  Space O(m*n)",
    "followup": "What is Opening (erosion then dilation) and Closing? What artifacts does each remove?"
  },
  {
    "id": "C08",
    "difficulty": "Easy",
    "tag": "Interpolation",
    "points": 10,
    "title": "Image Rotation via Bilinear Interpolation",
    "company": "Adobe, Pixar",
    "desc": "Rotate an image by an arbitrary angle (degrees) using inverse mapping and bilinear interpolation. Implement from scratch without cv2.warpAffine(). Fundamental to understanding affine transformations.",
    "example": "Rotate a 100x100 image by 45 degrees around its centre.",
    "hint": "For each output pixel (x,y), apply the inverse rotation matrix to find the source pixel. Blend the 4 surrounding source pixels bilinearly.",
    "fullSolution": "# Rotation with bilinear interpolation\nimport numpy as np, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\ndef rotate_bilinear(img,angle_deg):\n    h,w=img.shape[:2]; cx,cy=w/2,h/2\n    a=np.radians(angle_deg); ca,sa=np.cos(a),np.sin(a)\n    out=np.zeros_like(img)\n    for yo in range(h):\n        for xo in range(w):\n            dx,dy=xo-cx,yo-cy\n            xs=ca*dx+sa*dy+cx; ys=-sa*dx+ca*dy+cy\n            x0,y0=int(xs),int(ys); x1,y1=x0+1,y0+1\n            if 0<=x0<w-1 and 0<=y0<h-1:\n                tx,ty=xs-x0,ys-y0\n                if img.ndim==3:\n                    out[yo,xo]=((1-tx)*(1-ty)*img[y0,x0]+tx*(1-ty)*img[y0,x1]+(1-tx)*ty*img[y1,x0]+tx*ty*img[y1,x1]).astype(np.uint8)\n                else:\n                    out[yo,xo]=int((1-tx)*(1-ty)*img[y0,x0]+tx*(1-ty)*img[y0,x1]+(1-tx)*ty*img[y1,x0]+tx*ty*img[y1,x1])\n    return out\n\nurl='https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg'\nraw=urllib.request.urlopen(url).read()\nimg=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)\nsmall=cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),(80,80))\nrot=rotate_bilinear(small,45)\nM=cv2.getRotationMatrix2D((40,40),45,1.0)\ncv_rot=cv2.warpAffine(small,M,(80,80))\nfig,(a,b,c)=plt.subplots(1,3,figsize=(12,4))\nfor ax,im,t in zip([a,b,c],[small,rot,cv_rot],['Original','Our Rotation','cv2 Rotation']):\n    ax.imshow(im); ax.set_title(t); ax.axis('off')\nplt.tight_layout(); plt.show()",
    "complexity": "Time O(m*n)  Space O(m*n)",
    "followup": "Why do we use inverse mapping rather than forward mapping? What artifacts does forward mapping produce?"
  },
  {
    "id": "C09",
    "difficulty": "Easy",
    "tag": "Metrics",
    "points": 10,
    "title": "PSNR and SSIM from Scratch",
    "company": "Netflix, YouTube",
    "desc": "Implement PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) from scratch. These are the two most widely used metrics for evaluating image restoration, compression, and generation quality.",
    "example": "PSNR = 10*log10(255^2 / MSE). SSIM measures luminance, contrast, and structural similarity jointly.",
    "hint": "PSNR: compute MSE then apply the log formula. SSIM: compute local means and variances with Gaussian filtering, then apply the SSIM formula per window and average.",
    "fullSolution": "# PSNR and SSIM from scratch\nimport numpy as np, cv2, urllib.request\nimport matplotlib.pyplot as plt\nfrom skimage.metrics import peak_signal_noise_ratio as sk_psnr, structural_similarity as sk_ssim\n\ndef my_psnr(a,b,max_val=255.0):\n    mse=np.mean((a.astype(float)-b.astype(float))**2)\n    return float('inf') if mse==0 else 10*np.log10(max_val**2/mse)\n\ndef my_ssim(a,b,max_val=255.0,k1=0.01,k2=0.03):\n    af,bf=a.astype(float),b.astype(float)\n    C1,C2=(k1*max_val)**2,(k2*max_val)**2\n    mu1,mu2=af.mean(),bf.mean()\n    s1,s2=af.std(),bf.std()\n    s12=np.mean((af-mu1)*(bf-mu2))\n    return ((2*mu1*mu2+C1)*(2*s12+C2))/((mu1**2+mu2**2+C1)*(s1**2+s2**2+C2))\n\nurl='https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw=urllib.request.urlopen(url).read()\nclean=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)\nclean=cv2.resize(clean,(256,256))\nrng=np.random.default_rng(0)\nfor sigma in [5,15,30,50]:\n    noisy=np.clip(clean.astype(int)+rng.normal(0,sigma,clean.shape).astype(int),0,255).astype(np.uint8)\n    print(f'sigma={sigma:2d}: PSNR={my_psnr(clean,noisy):.2f}dB (sk={sk_psnr(clean,noisy,data_range=255):.2f}) SSIM={my_ssim(clean,noisy):.4f}')",
    "complexity": "PSNR: O(m*n)  SSIM windowed: O(m*n*win^2)",
    "followup": "Why does SSIM correlate better with human perception than PSNR? What does each component (luminance, contrast, structure) measure?"
  },
  {
    "id": "C10",
    "difficulty": "Easy",
    "tag": "Template Matching",
    "points": 10,
    "title": "Template Matching (Normalised Cross-Correlation)",
    "company": "OpenCV, Robotics",
    "desc": "Given a source image and a template patch, find the best-match location using Normalised Cross-Correlation (NCC). NCC is invariant to additive brightness changes and used in optical flow, stereo matching, and feature tracking.",
    "example": "Find a small eye-patch template in a portrait image using NCC sliding window.",
    "hint": "For each window position, compute NCC = dot(norm(patch), norm(T)) / (th*tw) where norm means subtract mean and divide by std. Maximum = best match.",
    "fullSolution": "# Template Matching with NCC\nimport numpy as np, cv2, urllib.request\nimport matplotlib.pyplot as plt, matplotlib.patches as patches\n\ndef ncc_match(img,tmpl):\n    ih,iw=img.shape; th,tw=tmpl.shape\n    oh,ow=ih-th+1,iw-tw+1; score=np.zeros((oh,ow))\n    tn=tmpl.astype(float)-tmpl.mean(); ts=tn.std()\n    if ts<1e-8: return score,(0,0)\n    tn/=ts\n    for i in range(oh):\n        for j in range(ow):\n            p=img[i:i+th,j:j+tw].astype(float); p-=p.mean(); ps=p.std()\n            if ps>1e-8: score[i,j]=np.sum(p/ps*tn)/(th*tw)\n    best=np.unravel_index(score.argmax(),score.shape)\n    return score,best\n\nurl='https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Bill_Nye_2017.jpg/240px-Bill_Nye_2017.jpg'\nraw=urllib.request.urlopen(url).read()\nimg=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)\nimg=cv2.resize(img,(200,240))\ntemplate=img[70:110,60:110]\nprint('Running NCC...')\nscore,(br,bc)=ncc_match(img,template)\nprint(f'Best match at ({br},{bc}) score={score.max():.4f}')\nres_cv=cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)\n_,_,_,(bc2,br2)=cv2.minMaxLoc(res_cv)\nprint(f'cv2 match at ({br2},{bc2})')",
    "complexity": "Time O((m-th)*(n-tw)*th*tw). Use FFT for O(m*n*log(m*n)).",
    "followup": "How can FFT cross-correlation speed this up from O(m*n*k^2) to O(m*n*log(m*n))?"
  },
  {
    "id": "C11",
    "difficulty": "Medium",
    "tag": "NMS",
    "points": 20,
    "title": "Implement Soft-NMS",
    "company": "Google, Waymo",
    "desc": "Standard NMS removes overlapping detections with IoU > threshold, causing recall drops in crowded scenes. Soft-NMS decays confidence scores of overlapping boxes using a Gaussian penalty rather than hard removal. Implement both variants and compare.",
    "example": "Soft-NMS: score_j = score_j * exp(-IoU(i,j)^2 / sigma) for each remaining box j.",
    "hint": "Sort by score, pick the top box, apply Gaussian decay to all remaining boxes based on their IoU with the top box, re-sort. Repeat until score threshold is breached.",
    "fullSolution": "# Hard NMS vs Soft-NMS comparison\nimport numpy as np, matplotlib.pyplot as plt, matplotlib.patches as patches\n\ndef box_iou(a,b):\n    ix1,iy1=max(a[0],b[0]),max(a[1],b[1])\n    ix2,iy2=min(a[2],b[2]),min(a[3],b[3])\n    inter=max(0,ix2-ix1)*max(0,iy2-iy1)\n    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-7)\n\ndef hard_nms(boxes,scores,iou_thresh=0.5,score_thresh=0.05):\n    order=np.argsort(scores)[::-1].tolist(); keep=[]\n    while order:\n        i=order.pop(0); keep.append(i)\n        order=[j for j in order if box_iou(boxes[i],boxes[j])<=iou_thresh]\n    return [k for k in keep if scores[k]>=score_thresh]\n\ndef soft_nms(boxes,scores,sigma=0.5,score_thresh=0.05):\n    scores=scores.copy(); n=len(scores); keep=[]\n    for _ in range(n):\n        i=np.argmax(scores)\n        if scores[i]<score_thresh: break\n        keep.append(i); scores[i]=-1\n        for j in range(n):\n            if scores[j]<0: continue\n            scores[j]*=np.exp(-box_iou(boxes[i],boxes[j])**2/sigma)\n    return keep\n\nnp.random.seed(42); n=20\nboxes=[]; \nfor _ in range(n):\n    cx,cy=np.random.uniform(50,200,2); w,h=np.random.uniform(20,60,2)\n    boxes.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])\nboxes=np.array(boxes); scores=np.random.uniform(0.3,0.95,n)\nprint(f'Before NMS: {n} boxes')\nprint(f'Hard NMS:   {len(hard_nms(boxes,scores.copy()))} kept')\nprint(f'Soft-NMS:   {len(soft_nms(boxes,scores.copy()))} kept')",
    "complexity": "Both O(n^2) naive, O(n log n) with priority queue",
    "followup": "What is DIoU-NMS? How does incorporating center-point distance improve crowded pedestrian detection?"
  },
  {
    "id": "C12",
    "difficulty": "Medium",
    "tag": "Thresholding",
    "points": 20,
    "title": "Otsu's Global Threshold from Scratch",
    "company": "Medical Imaging, Document AI",
    "desc": "Implement Otsu's method to automatically find the optimal global threshold T that maximises inter-class variance between foreground and background pixels. Used in document binarisation and medical image preprocessing.",
    "example": "Input: grayscale image with bimodal histogram. Output: optimal threshold T cleanly separating the two modes.",
    "hint": "For each candidate threshold t in [0,255], compute w0*w1*(mu0-mu1)^2 using cumulative sums. Return the t that maximises this inter-class variance.",
    "fullSolution": "# Otsu's thresholding from scratch\nimport numpy as np, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\ndef otsu_fast(img):\n    hist,_=np.histogram(img.flatten(),256,[0,256])\n    total=img.size\n    w=np.cumsum(hist)/total\n    mu=np.cumsum(np.arange(256)*hist)/total\n    mu_total=mu[-1]\n    var_b=np.where(w*(1-w)>0,(mu_total*w-mu)**2/(w*(1-w)),0)\n    return int(np.argmax(var_b))\n\nurl='https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/License_plate_2.jpg/320px-License_plate_2.jpg'\nraw=urllib.request.urlopen(url).read()\nimg=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)\nimg=cv2.resize(img,(256,128))\nt_ours=otsu_fast(img)\n_,bw_ours=cv2.threshold(img,t_ours,255,cv2.THRESH_BINARY)\nret,bw_cv2=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)\nprint(f'Our T={t_ours}, cv2 T={int(ret)}')\nassert abs(t_ours-int(ret))<=1, 'Threshold mismatch!'\nprint('Match!')",
    "complexity": "Time O(m*n + 256)  Space O(256)",
    "followup": "When does Otsu's method fail? What is multi-level Otsu and when is it needed?"
  },
  {
    "id": "C13",
    "difficulty": "Medium",
    "tag": "Dynamic Programming",
    "points": 20,
    "title": "Seam Carving for Content-Aware Image Resizing",
    "company": "Adobe, Figma",
    "desc": "Find and remove vertical seams of minimum energy to shrink an image while preserving important content. The energy is the gradient magnitude; a seam is a connected top-to-bottom path through the image.",
    "example": "Remove 20 vertical seams from a landscape photo narrowing it without distorting faces or objects.",
    "hint": "Step 1: compute energy (gradient magnitude). Step 2: DP table filling top-down. Step 3: backtrack to find seam. Step 4: remove seam pixels. Repeat.",
    "fullSolution": "# Seam Carving\nimport numpy as np, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\ndef energy(img):\n    g=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY).astype(float)\n    return np.abs(np.gradient(g,axis=1))+np.abs(np.gradient(g,axis=0))\n\ndef find_seam(e):\n    h,w=e.shape; dp=e.copy()\n    for i in range(1,h):\n        for j in range(w):\n            dp[i,j]+=min(dp[i-1,j-1] if j>0 else 1e9, dp[i-1,j], dp[i-1,j+1] if j<w-1 else 1e9)\n    seam=np.zeros(h,dtype=int); seam[-1]=dp[-1].argmin()\n    for i in range(h-2,-1,-1):\n        j=seam[i+1]; opts={}\n        if j>0: opts[j-1]=dp[i,j-1]\n        opts[j]=dp[i,j]\n        if j<w-1: opts[j+1]=dp[i,j+1]\n        seam[i]=min(opts,key=opts.get)\n    return seam\n\ndef remove_seam(img,seam):\n    h,w=img.shape[:2]; out=np.zeros((h,w-1,3),dtype=img.dtype)\n    for i in range(h): out[i,:seam[i]]=img[i,:seam[i]]; out[i,seam[i]:]=img[i,seam[i]+1:]\n    return out\n\nurl='https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw=urllib.request.urlopen(url).read()\nimg=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)\nimg_rgb=cv2.cvtColor(cv2.resize(img,(160,120)),cv2.COLOR_BGR2RGB)\ncarved=img_rgb.copy()\nfor _ in range(20): carved=remove_seam(carved,find_seam(energy(carved)))\nprint(f'{img_rgb.shape[1]}px -> {carved.shape[1]}px')\nfig,(a,b)=plt.subplots(1,2,figsize=(12,5))\na.imshow(img_rgb); a.set_title('Original'); a.axis('off')\nb.imshow(carved); b.set_title('Seam Carved (-20 cols)'); b.axis('off')\nplt.tight_layout(); plt.show()",
    "complexity": "Per seam: O(m*n). Total for k seams: O(k*m*n).",
    "followup": "How would you protect specific regions (faces) from being carved using a protection mask?"
  },
  {
    "id": "C14",
    "difficulty": "Medium",
    "tag": "Hashing",
    "points": 20,
    "title": "Perceptual Hash (pHash) for Near-Duplicate Detection",
    "company": "Google Photos, Pinterest",
    "desc": "Implement perceptual hash (pHash) using DCT for near-duplicate image detection. pHash produces similar binary hashes for visually similar images and distant hashes for different images, enabling efficient image deduplication at scale.",
    "example": "Two slightly different versions of the same photo: Hamming distance < 10. Completely different images: Hamming distance > 40.",
    "hint": "Resize to 32x32, convert to grayscale, compute 2D DCT, take the top-left 8x8 low-frequency block, binarise values above the block mean.",
    "fullSolution": "# Perceptual Hash (pHash)\nimport numpy as np, cv2, urllib.request\nfrom scipy.fft import dct\n\ndef phash(img,hash_size=8,hf=32):\n    gray=cv2.cvtColor(cv2.resize(img,(hf,hf)),cv2.COLOR_BGR2GRAY).astype(float)\n    d=dct(dct(gray,axis=0,norm='ortho'),axis=1,norm='ortho')\n    low=d[:hash_size,:hash_size]\n    return (low>low.mean()).flatten()\n\ndef hamming(h1,h2): return int((h1!=h2).sum())\n\nurl='https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg'\nraw=urllib.request.urlopen(url).read()\nimg=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)\nbright=np.clip(img.astype(int)+40,0,255).astype(np.uint8)\nnoisy=np.clip(img.astype(int)+np.random.normal(0,20,img.shape).astype(int),0,255).astype(np.uint8)\nurl2='https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\ndiff=cv2.imdecode(np.frombuffer(urllib.request.urlopen(url2).read(),np.uint8),cv2.IMREAD_COLOR)\ndiff=cv2.resize(diff,img.shape[:2][::-1])\nh0=phash(img)\nfor name,other in [('Brightened',bright),('Noisy',noisy),('Different',diff)]:\n    d=hamming(h0,phash(other))\n    print(f'{name:15s}: Hamming={d:2d} [{\"SIMILAR\" if d<15 else \"DIFFERENT\"}]')",
    "complexity": "O(N^2 log N) for 2D DCT where N=high_freq. Space O(N^2).",
    "followup": "What is the difference between aHash (average hash), dHash (difference hash), and pHash? When would you choose each?"
  },
  {
    "id": "C15",
    "difficulty": "Medium",
    "tag": "Deep Learning Math",
    "points": 20,
    "title": "Conv2D Forward Pass + im2col from Scratch",
    "company": "NVIDIA, Meta AI",
    "desc": "Implement a 2D convolution forward pass using only NumPy: first as naive nested loops, then as an efficient im2col (image-to-column) vectorised version. Verify both match PyTorch to within floating-point tolerance.",
    "example": "Input (2,3,8,8), Kernel (4,3,3,3), Stride=1, Padding=1 -> Output (2,4,8,8)",
    "hint": "im2col flattens each receptive field into a column, stacks all columns into a matrix, then performs a single matrix multiplication with the reshaped weight matrix.",
    "fullSolution": "# Conv2D forward: naive + im2col\nimport numpy as np, torch, torch.nn as nn\n\ndef conv2d_naive(x,W,b,s=1,p=0):\n    N,C,H,Wi=x.shape; F,_,kH,kW=W.shape\n    Hp=(H+2*p-kH)//s+1; Wp=(Wi+2*p-kW)//s+1\n    if p: x=np.pad(x,((0,0),(0,0),(p,p),(p,p)))\n    out=np.zeros((N,F,Hp,Wp))\n    for n in range(N):\n        for f in range(F):\n            for i in range(Hp):\n                for j in range(Wp):\n                    out[n,f,i,j]=np.sum(x[n,:,i*s:i*s+kH,j*s:j*s+kW]*W[f])+b[f]\n    return out\n\ndef conv2d_im2col(x,W,b,s=1,p=0):\n    N,C,H,Wi=x.shape; F,_,kH,kW=W.shape\n    Hp=(H+2*p-kH)//s+1; Wp=(Wi+2*p-kW)//s+1\n    if p: x=np.pad(x,((0,0),(0,0),(p,p),(p,p)))\n    cols=np.zeros((N,C,kH,kW,Hp,Wp))\n    for i in range(kH):\n        for j in range(kW):\n            cols[:,:,i,j,:,:]=x[:,:,i:i+s*Hp:s,j:j+s*Wp:s]\n    cols=cols.reshape(N,C*kH*kW,Hp*Wp)\n    return (W.reshape(F,C*kH*kW)@cols+b[:,None]).reshape(N,F,Hp,Wp)\n\nnp.random.seed(0)\nx=np.random.randn(2,3,8,8).astype(np.float32)\nW=np.random.randn(4,3,3,3).astype(np.float32)\nb=np.random.randn(4).astype(np.float32)\nout_n=conv2d_naive(x.copy(),W,b,s=1,p=1)\nout_i=conv2d_im2col(x.copy(),W,b,s=1,p=1)\nconv=nn.Conv2d(3,4,3,padding=1,bias=True)\nconv.weight.data=torch.from_numpy(W); conv.bias.data=torch.from_numpy(b)\nwith torch.no_grad(): out_pt=conv(torch.from_numpy(x)).numpy()\nprint('Naive  vs PyTorch max diff:', np.abs(out_n-out_pt).max())\nprint('im2col vs PyTorch max diff:', np.abs(out_i-out_pt).max())",
    "complexity": "Naive: O(N*F*C*kH*kW*Hp*Wp). im2col: same ops but cache-friendly matrix multiply.",
    "followup": "What is depthwise separable convolution and why does it reduce parameters by a factor of k^2 / (1 + k^2/C_out)?"
  },
  {
    "id": "C16",
    "difficulty": "Medium",
    "tag": "Optimisation",
    "points": 20,
    "title": "Total Variation Denoising",
    "company": "Research / Industry",
    "desc": "Implement Total Variation Denoising from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C16: Total Variation Denoising\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Total Variation Denoising\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C16: Total Variation Denoising'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Total Variation Denoising to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C17",
    "difficulty": "Hard",
    "tag": "Backprop",
    "points": 30,
    "title": "Conv2D Backward Pass",
    "company": "Research / Industry",
    "desc": "Implement Conv2D Backward Pass from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C17: Conv2D Backward Pass\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Conv2D Backward Pass\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C17: Conv2D Backward Pass'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Conv2D Backward Pass to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C18",
    "difficulty": "Medium",
    "tag": "Graph Search",
    "points": 20,
    "title": "Dijkstra Shortest Path on Grid",
    "company": "Research / Industry",
    "desc": "Implement Dijkstra Shortest Path on Grid from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C18: Dijkstra Shortest Path on Grid\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Dijkstra Shortest Path on Grid\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C18: Dijkstra Shortest Path on Grid'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Dijkstra Shortest Path on Grid to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C19",
    "difficulty": "Hard",
    "tag": "Transformers",
    "points": 30,
    "title": "Scaled Dot-Product Attention",
    "company": "Research / Industry",
    "desc": "Implement Scaled Dot-Product Attention from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C19: Scaled Dot-Product Attention\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Scaled Dot-Product Attention\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C19: Scaled Dot-Product Attention'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Scaled Dot-Product Attention to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C20",
    "difficulty": "Medium",
    "tag": "Feature Detection",
    "points": 20,
    "title": "Harris Corner Detector",
    "company": "Research / Industry",
    "desc": "Implement Harris Corner Detector from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C20: Harris Corner Detector\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Harris Corner Detector\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C20: Harris Corner Detector'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Harris Corner Detector to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C21",
    "difficulty": "Medium",
    "tag": "Clustering",
    "points": 20,
    "title": "K-Means Image Segmentation",
    "company": "Research / Industry",
    "desc": "Implement K-Means Image Segmentation from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C21: K-Means Image Segmentation\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement K-Means Image Segmentation\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C21: K-Means Image Segmentation'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend K-Means Image Segmentation to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C22",
    "difficulty": "Hard",
    "tag": "Conv Math",
    "points": 30,
    "title": "Depthwise Separable Conv Forward",
    "company": "Research / Industry",
    "desc": "Implement Depthwise Separable Conv Forward from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C22: Depthwise Separable Conv Forward\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Depthwise Separable Conv Forward\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C22: Depthwise Separable Conv Forward'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Depthwise Separable Conv Forward to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C23",
    "difficulty": "Medium",
    "tag": "Loss Functions",
    "points": 20,
    "title": "Focal Loss Implementation",
    "company": "Research / Industry",
    "desc": "Implement Focal Loss Implementation from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C23: Focal Loss Implementation\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Focal Loss Implementation\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C23: Focal Loss Implementation'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Focal Loss Implementation to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C24",
    "difficulty": "Hard",
    "tag": "Geometry",
    "points": 30,
    "title": "Homography DLT Algorithm",
    "company": "Research / Industry",
    "desc": "Implement Homography DLT Algorithm from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C24: Homography DLT Algorithm\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Homography DLT Algorithm\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C24: Homography DLT Algorithm'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Homography DLT Algorithm to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C25",
    "difficulty": "Medium",
    "tag": "Feature Matching",
    "points": 20,
    "title": "SIFT Feature Matching",
    "company": "Research / Industry",
    "desc": "Implement SIFT Feature Matching from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C25: SIFT Feature Matching\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement SIFT Feature Matching\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C25: SIFT Feature Matching'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend SIFT Feature Matching to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C26",
    "difficulty": "Medium",
    "tag": "Generative",
    "points": 20,
    "title": "DCGAN on MNIST",
    "company": "Research / Industry",
    "desc": "Implement DCGAN on MNIST from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C26: DCGAN on MNIST\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement DCGAN on MNIST\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C26: DCGAN on MNIST'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend DCGAN on MNIST to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C27",
    "difficulty": "Hard",
    "tag": "Segmentation",
    "points": 30,
    "title": "U-Net with Dice Loss",
    "company": "Research / Industry",
    "desc": "Implement U-Net with Dice Loss from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C27: U-Net with Dice Loss\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement U-Net with Dice Loss\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C27: U-Net with Dice Loss'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend U-Net with Dice Loss to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C28",
    "difficulty": "Medium",
    "tag": "Attention",
    "points": 20,
    "title": "Multi-Head Attention Full",
    "company": "Research / Industry",
    "desc": "Implement Multi-Head Attention Full from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C28: Multi-Head Attention Full\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Multi-Head Attention Full\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C28: Multi-Head Attention Full'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Multi-Head Attention Full to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C29",
    "difficulty": "Hard",
    "tag": "3D Rendering",
    "points": 30,
    "title": "NeRF Volume Rendering",
    "company": "Research / Industry",
    "desc": "Implement NeRF Volume Rendering from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C29: NeRF Volume Rendering\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement NeRF Volume Rendering\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C29: NeRF Volume Rendering'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend NeRF Volume Rendering to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C30",
    "difficulty": "Medium",
    "tag": "Diffusion",
    "points": 20,
    "title": "DDPM Noise Schedule",
    "company": "Research / Industry",
    "desc": "Implement DDPM Noise Schedule from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C30: DDPM Noise Schedule\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement DDPM Noise Schedule\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C30: DDPM Noise Schedule'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend DDPM Noise Schedule to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C31",
    "difficulty": "Medium",
    "tag": "Flow",
    "points": 20,
    "title": "Dense Optical Flow (Lucas-Kanade)",
    "company": "Research / Industry",
    "desc": "Implement Dense Optical Flow (Lucas-Kanade) from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C31: Dense Optical Flow (Lucas-Kanade)\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Dense Optical Flow (Lucas-Kanade)\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C31: Dense Optical Flow (Lucas-Kanade)'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Dense Optical Flow (Lucas-Kanade) to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C32",
    "difficulty": "Hard",
    "tag": "Tracking",
    "points": 30,
    "title": "ByteTrack Association",
    "company": "Research / Industry",
    "desc": "Implement ByteTrack Association from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C32: ByteTrack Association\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement ByteTrack Association\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C32: ByteTrack Association'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend ByteTrack Association to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C33",
    "difficulty": "Medium",
    "tag": "Calibration",
    "points": 20,
    "title": "Camera Calibration (DLT)",
    "company": "Research / Industry",
    "desc": "Implement Camera Calibration (DLT) from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C33: Camera Calibration (DLT)\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Camera Calibration (DLT)\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C33: Camera Calibration (DLT)'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Camera Calibration (DLT) to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C34",
    "difficulty": "Hard",
    "tag": "SSL",
    "points": 30,
    "title": "SimCLR Contrastive Loss",
    "company": "Research / Industry",
    "desc": "Implement SimCLR Contrastive Loss from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C34: SimCLR Contrastive Loss\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement SimCLR Contrastive Loss\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C34: SimCLR Contrastive Loss'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend SimCLR Contrastive Loss to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C35",
    "difficulty": "Medium",
    "tag": "Quantisation",
    "points": 20,
    "title": "INT8 Quantisation from Scratch",
    "company": "Research / Industry",
    "desc": "Implement INT8 Quantisation from Scratch from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C35: INT8 Quantisation from Scratch\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement INT8 Quantisation from Scratch\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C35: INT8 Quantisation from Scratch'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend INT8 Quantisation from Scratch to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C36",
    "difficulty": "Medium",
    "tag": "Optimisation",
    "points": 20,
    "title": "Magnitude Pruning",
    "company": "Research / Industry",
    "desc": "Implement Magnitude Pruning from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C36: Magnitude Pruning\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Magnitude Pruning\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C36: Magnitude Pruning'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Magnitude Pruning to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C37",
    "difficulty": "Hard",
    "tag": "Backprop",
    "points": 30,
    "title": "Response-Based Distillation",
    "company": "Research / Industry",
    "desc": "Implement Response-Based Distillation from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C37: Response-Based Distillation\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Response-Based Distillation\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C37: Response-Based Distillation'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Response-Based Distillation to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C38",
    "difficulty": "Medium",
    "tag": "Graph Search",
    "points": 20,
    "title": "GCN Node Classification",
    "company": "Research / Industry",
    "desc": "Implement GCN Node Classification from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C38: GCN Node Classification\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement GCN Node Classification\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C38: GCN Node Classification'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend GCN Node Classification to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C39",
    "difficulty": "Hard",
    "tag": "Transformers",
    "points": 30,
    "title": "RealNVP Normalising Flow",
    "company": "Research / Industry",
    "desc": "Implement RealNVP Normalising Flow from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C39: RealNVP Normalising Flow\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement RealNVP Normalising Flow\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C39: RealNVP Normalising Flow'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend RealNVP Normalising Flow to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C40",
    "difficulty": "Medium",
    "tag": "Feature Detection",
    "points": 20,
    "title": "Score Matching Loss",
    "company": "Research / Industry",
    "desc": "Implement Score Matching Loss from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C40: Score Matching Loss\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Score Matching Loss\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C40: Score Matching Loss'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Score Matching Loss to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C41",
    "difficulty": "Medium",
    "tag": "Clustering",
    "points": 20,
    "title": "CLIP Fine-tuning Loop",
    "company": "Research / Industry",
    "desc": "Implement CLIP Fine-tuning Loop from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C41: CLIP Fine-tuning Loop\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement CLIP Fine-tuning Loop\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C41: CLIP Fine-tuning Loop'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend CLIP Fine-tuning Loop to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C42",
    "difficulty": "Hard",
    "tag": "Conv Math",
    "points": 30,
    "title": "DeiT Distillation Token",
    "company": "Research / Industry",
    "desc": "Implement DeiT Distillation Token from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C42: DeiT Distillation Token\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement DeiT Distillation Token\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C42: DeiT Distillation Token'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend DeiT Distillation Token to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C43",
    "difficulty": "Medium",
    "tag": "Loss Functions",
    "points": 20,
    "title": "DINO Self-Distillation Loss",
    "company": "Research / Industry",
    "desc": "Implement DINO Self-Distillation Loss from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C43: DINO Self-Distillation Loss\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement DINO Self-Distillation Loss\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C43: DINO Self-Distillation Loss'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend DINO Self-Distillation Loss to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C44",
    "difficulty": "Hard",
    "tag": "Geometry",
    "points": 30,
    "title": "Laplacian Pyramid Blending",
    "company": "Research / Industry",
    "desc": "Implement Laplacian Pyramid Blending from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C44: Laplacian Pyramid Blending\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Laplacian Pyramid Blending\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C44: Laplacian Pyramid Blending'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Laplacian Pyramid Blending to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C45",
    "difficulty": "Medium",
    "tag": "Feature Matching",
    "points": 20,
    "title": "SSIM Windowed Map",
    "company": "Research / Industry",
    "desc": "Implement SSIM Windowed Map from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C45: SSIM Windowed Map\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement SSIM Windowed Map\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C45: SSIM Windowed Map'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend SSIM Windowed Map to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C46",
    "difficulty": "Medium",
    "tag": "Generative",
    "points": 20,
    "title": "Perceptual Loss (VGG Features)",
    "company": "Research / Industry",
    "desc": "Implement Perceptual Loss (VGG Features) from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C46: Perceptual Loss (VGG Features)\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Perceptual Loss (VGG Features)\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C46: Perceptual Loss (VGG Features)'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Perceptual Loss (VGG Features) to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C47",
    "difficulty": "Hard",
    "tag": "Segmentation",
    "points": 30,
    "title": "CycleGAN Loss Functions",
    "company": "Research / Industry",
    "desc": "Implement CycleGAN Loss Functions from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C47: CycleGAN Loss Functions\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement CycleGAN Loss Functions\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C47: CycleGAN Loss Functions'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend CycleGAN Loss Functions to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C48",
    "difficulty": "Medium",
    "tag": "Attention",
    "points": 20,
    "title": "StyleGAN Truncation Trick",
    "company": "Research / Industry",
    "desc": "Implement StyleGAN Truncation Trick from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C48: StyleGAN Truncation Trick\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement StyleGAN Truncation Trick\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C48: StyleGAN Truncation Trick'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend StyleGAN Truncation Trick to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C49",
    "difficulty": "Hard",
    "tag": "3D Rendering",
    "points": 30,
    "title": "VQVAE Codebook Learning",
    "company": "Research / Industry",
    "desc": "Implement VQVAE Codebook Learning from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C49: VQVAE Codebook Learning\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement VQVAE Codebook Learning\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C49: VQVAE Codebook Learning'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend VQVAE Codebook Learning to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C50",
    "difficulty": "Medium",
    "tag": "Diffusion",
    "points": 20,
    "title": "Rectified Flow Sampling",
    "company": "Research / Industry",
    "desc": "Implement Rectified Flow Sampling from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C50: Rectified Flow Sampling\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Rectified Flow Sampling\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C50: Rectified Flow Sampling'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Rectified Flow Sampling to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C51",
    "difficulty": "Medium",
    "tag": "Flow",
    "points": 20,
    "title": "SDXL Text Conditioning",
    "company": "Research / Industry",
    "desc": "Implement SDXL Text Conditioning from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C51: SDXL Text Conditioning\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement SDXL Text Conditioning\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C51: SDXL Text Conditioning'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend SDXL Text Conditioning to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C52",
    "difficulty": "Hard",
    "tag": "Tracking",
    "points": 30,
    "title": "IP-Adapter Cross-Attention",
    "company": "Research / Industry",
    "desc": "Implement IP-Adapter Cross-Attention from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C52: IP-Adapter Cross-Attention\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement IP-Adapter Cross-Attention\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C52: IP-Adapter Cross-Attention'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend IP-Adapter Cross-Attention to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C53",
    "difficulty": "Medium",
    "tag": "Calibration",
    "points": 20,
    "title": "ControlNet Zero-Convolution",
    "company": "Research / Industry",
    "desc": "Implement ControlNet Zero-Convolution from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C53: ControlNet Zero-Convolution\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement ControlNet Zero-Convolution\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C53: ControlNet Zero-Convolution'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend ControlNet Zero-Convolution to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C54",
    "difficulty": "Hard",
    "tag": "SSL",
    "points": 30,
    "title": "Affine-Invariant Depth Loss",
    "company": "Research / Industry",
    "desc": "Implement Affine-Invariant Depth Loss from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C54: Affine-Invariant Depth Loss\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Affine-Invariant Depth Loss\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C54: Affine-Invariant Depth Loss'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Affine-Invariant Depth Loss to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C55",
    "difficulty": "Medium",
    "tag": "Quantisation",
    "points": 20,
    "title": "Multi-Dataset Depth Training",
    "company": "Research / Industry",
    "desc": "Implement Multi-Dataset Depth Training from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C55: Multi-Dataset Depth Training\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Multi-Dataset Depth Training\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C55: Multi-Dataset Depth Training'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Multi-Dataset Depth Training to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C56",
    "difficulty": "Medium",
    "tag": "Optimisation",
    "points": 20,
    "title": "Panoptic Quality Metric",
    "company": "Research / Industry",
    "desc": "Implement Panoptic Quality Metric from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C56: Panoptic Quality Metric\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Panoptic Quality Metric\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C56: Panoptic Quality Metric'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Panoptic Quality Metric to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C57",
    "difficulty": "Hard",
    "tag": "Backprop",
    "points": 30,
    "title": "Boundary F1 Score",
    "company": "Research / Industry",
    "desc": "Implement Boundary F1 Score from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C57: Boundary F1 Score\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Boundary F1 Score\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C57: Boundary F1 Score'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Boundary F1 Score to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C58",
    "difficulty": "Medium",
    "tag": "Graph Search",
    "points": 20,
    "title": "Video Instance Segmentation",
    "company": "Research / Industry",
    "desc": "Implement Video Instance Segmentation from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C58: Video Instance Segmentation\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Video Instance Segmentation\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C58: Video Instance Segmentation'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Video Instance Segmentation to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C59",
    "difficulty": "Hard",
    "tag": "Transformers",
    "points": 30,
    "title": "4D Gaussian Splatting Deformation",
    "company": "Research / Industry",
    "desc": "Implement 4D Gaussian Splatting Deformation from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C59: 4D Gaussian Splatting Deformation\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement 4D Gaussian Splatting Deformation\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C59: 4D Gaussian Splatting Deformation'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend 4D Gaussian Splatting Deformation to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C60",
    "difficulty": "Medium",
    "tag": "Feature Detection",
    "points": 20,
    "title": "Scene Text Cropper Pipeline",
    "company": "Research / Industry",
    "desc": "Implement Scene Text Cropper Pipeline from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C60: Scene Text Cropper Pipeline\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Scene Text Cropper Pipeline\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C60: Scene Text Cropper Pipeline'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Scene Text Cropper Pipeline to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C61",
    "difficulty": "Medium",
    "tag": "Clustering",
    "points": 20,
    "title": "Table Structure Recognition",
    "company": "Research / Industry",
    "desc": "Implement Table Structure Recognition from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C61: Table Structure Recognition\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Table Structure Recognition\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C61: Table Structure Recognition'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Table Structure Recognition to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C62",
    "difficulty": "Hard",
    "tag": "Conv Math",
    "points": 30,
    "title": "DocVQA ANLS Metric",
    "company": "Research / Industry",
    "desc": "Implement DocVQA ANLS Metric from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C62: DocVQA ANLS Metric\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement DocVQA ANLS Metric\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C62: DocVQA ANLS Metric'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend DocVQA ANLS Metric to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C63",
    "difficulty": "Medium",
    "tag": "Loss Functions",
    "points": 20,
    "title": "MedSAM Prompt Encoding",
    "company": "Research / Industry",
    "desc": "Implement MedSAM Prompt Encoding from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C63: MedSAM Prompt Encoding\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement MedSAM Prompt Encoding\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C63: MedSAM Prompt Encoding'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend MedSAM Prompt Encoding to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C64",
    "difficulty": "Hard",
    "tag": "Geometry",
    "points": 30,
    "title": "SAR Image Despeckling",
    "company": "Research / Industry",
    "desc": "Implement SAR Image Despeckling from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C64: SAR Image Despeckling\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement SAR Image Despeckling\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C64: SAR Image Despeckling'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend SAR Image Despeckling to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C65",
    "difficulty": "Medium",
    "tag": "Feature Matching",
    "points": 20,
    "title": "Siamese Change Detection",
    "company": "Research / Industry",
    "desc": "Implement Siamese Change Detection from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C65: Siamese Change Detection\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Siamese Change Detection\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C65: Siamese Change Detection'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Siamese Change Detection to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C66",
    "difficulty": "Medium",
    "tag": "Generative",
    "points": 20,
    "title": "CenterPoint 3D Heatmap Head",
    "company": "Research / Industry",
    "desc": "Implement CenterPoint 3D Heatmap Head from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C66: CenterPoint 3D Heatmap Head\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement CenterPoint 3D Heatmap Head\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C66: CenterPoint 3D Heatmap Head'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend CenterPoint 3D Heatmap Head to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C67",
    "difficulty": "Hard",
    "tag": "Segmentation",
    "points": 30,
    "title": "Lane Polynomial Fitting",
    "company": "Research / Industry",
    "desc": "Implement Lane Polynomial Fitting from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C67: Lane Polynomial Fitting\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Lane Polynomial Fitting\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C67: Lane Polynomial Fitting'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Lane Polynomial Fitting to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C68",
    "difficulty": "Medium",
    "tag": "Attention",
    "points": 20,
    "title": "Occupancy Voxel Grid BEV",
    "company": "Research / Industry",
    "desc": "Implement Occupancy Voxel Grid BEV from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C68: Occupancy Voxel Grid BEV\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Occupancy Voxel Grid BEV\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C68: Occupancy Voxel Grid BEV'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Occupancy Voxel Grid BEV to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C69",
    "difficulty": "Hard",
    "tag": "3D Rendering",
    "points": 30,
    "title": "BEV Feature Lifting (LSS)",
    "company": "Research / Industry",
    "desc": "Implement BEV Feature Lifting (LSS) from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C69: BEV Feature Lifting (LSS)\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement BEV Feature Lifting (LSS)\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C69: BEV Feature Lifting (LSS)'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend BEV Feature Lifting (LSS) to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C70",
    "difficulty": "Medium",
    "tag": "Diffusion",
    "points": 20,
    "title": "Kalman Filter MOT",
    "company": "Research / Industry",
    "desc": "Implement Kalman Filter MOT from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C70: Kalman Filter MOT\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Kalman Filter MOT\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C70: Kalman Filter MOT'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Kalman Filter MOT to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C71",
    "difficulty": "Medium",
    "tag": "Flow",
    "points": 20,
    "title": "ReID Feature Bank",
    "company": "Research / Industry",
    "desc": "Implement ReID Feature Bank from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C71: ReID Feature Bank\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement ReID Feature Bank\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C71: ReID Feature Bank'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend ReID Feature Bank to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C72",
    "difficulty": "Hard",
    "tag": "Tracking",
    "points": 30,
    "title": "ArcFace Margin Loss",
    "company": "Research / Industry",
    "desc": "Implement ArcFace Margin Loss from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C72: ArcFace Margin Loss\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement ArcFace Margin Loss\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C72: ArcFace Margin Loss'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend ArcFace Margin Loss to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C73",
    "difficulty": "Medium",
    "tag": "Calibration",
    "points": 20,
    "title": "Dynamic Capsule Routing",
    "company": "Research / Industry",
    "desc": "Implement Dynamic Capsule Routing from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C73: Dynamic Capsule Routing\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Dynamic Capsule Routing\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C73: Dynamic Capsule Routing'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Dynamic Capsule Routing to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C74",
    "difficulty": "Hard",
    "tag": "SSL",
    "points": 30,
    "title": "Sparse Convolution Inference",
    "company": "Research / Industry",
    "desc": "Implement Sparse Convolution Inference from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C74: Sparse Convolution Inference\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Sparse Convolution Inference\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C74: Sparse Convolution Inference'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Sparse Convolution Inference to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C75",
    "difficulty": "Medium",
    "tag": "Quantisation",
    "points": 20,
    "title": "Knowledge Graph VQA",
    "company": "Research / Industry",
    "desc": "Implement Knowledge Graph VQA from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C75: Knowledge Graph VQA\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Knowledge Graph VQA\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C75: Knowledge Graph VQA'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Knowledge Graph VQA to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C76",
    "difficulty": "Medium",
    "tag": "Optimisation",
    "points": 20,
    "title": "Cross-Modal Retrieval FAISS",
    "company": "Research / Industry",
    "desc": "Implement Cross-Modal Retrieval FAISS from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C76: Cross-Modal Retrieval FAISS\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Cross-Modal Retrieval FAISS\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C76: Cross-Modal Retrieval FAISS'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Cross-Modal Retrieval FAISS to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C77",
    "difficulty": "Hard",
    "tag": "Backprop",
    "points": 30,
    "title": "Open-Vocabulary Detection",
    "company": "Research / Industry",
    "desc": "Implement Open-Vocabulary Detection from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C77: Open-Vocabulary Detection\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Open-Vocabulary Detection\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C77: Open-Vocabulary Detection'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Open-Vocabulary Detection to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C78",
    "difficulty": "Medium",
    "tag": "Graph Search",
    "points": 20,
    "title": "Grounded SAM Pipeline",
    "company": "Research / Industry",
    "desc": "Implement Grounded SAM Pipeline from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C78: Grounded SAM Pipeline\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Grounded SAM Pipeline\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C78: Grounded SAM Pipeline'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Grounded SAM Pipeline to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C79",
    "difficulty": "Hard",
    "tag": "Transformers",
    "points": 30,
    "title": "Referring Image Segmentation",
    "company": "Research / Industry",
    "desc": "Implement Referring Image Segmentation from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C79: Referring Image Segmentation\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Referring Image Segmentation\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C79: Referring Image Segmentation'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Referring Image Segmentation to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C80",
    "difficulty": "Medium",
    "tag": "Feature Detection",
    "points": 20,
    "title": "Video-Text Contrastive Loss",
    "company": "Research / Industry",
    "desc": "Implement Video-Text Contrastive Loss from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C80: Video-Text Contrastive Loss\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Video-Text Contrastive Loss\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C80: Video-Text Contrastive Loss'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Video-Text Contrastive Loss to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C81",
    "difficulty": "Medium",
    "tag": "Clustering",
    "points": 20,
    "title": "Audio-Visual Segmentation",
    "company": "Research / Industry",
    "desc": "Implement Audio-Visual Segmentation from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C81: Audio-Visual Segmentation\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Audio-Visual Segmentation\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C81: Audio-Visual Segmentation'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Audio-Visual Segmentation to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C82",
    "difficulty": "Hard",
    "tag": "Conv Math",
    "points": 30,
    "title": "Multi-Scale Feature Alignment",
    "company": "Research / Industry",
    "desc": "Implement Multi-Scale Feature Alignment from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C82: Multi-Scale Feature Alignment\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Multi-Scale Feature Alignment\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C82: Multi-Scale Feature Alignment'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Multi-Scale Feature Alignment to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C83",
    "difficulty": "Medium",
    "tag": "Loss Functions",
    "points": 20,
    "title": "Deformable Attention Sampling",
    "company": "Research / Industry",
    "desc": "Implement Deformable Attention Sampling from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C83: Deformable Attention Sampling\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Deformable Attention Sampling\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C83: Deformable Attention Sampling'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Deformable Attention Sampling to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C84",
    "difficulty": "Hard",
    "tag": "Geometry",
    "points": 30,
    "title": "FlashAttention Memory Tiling",
    "company": "Research / Industry",
    "desc": "Implement FlashAttention Memory Tiling from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C84: FlashAttention Memory Tiling\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement FlashAttention Memory Tiling\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C84: FlashAttention Memory Tiling'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend FlashAttention Memory Tiling to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C85",
    "difficulty": "Medium",
    "tag": "Feature Matching",
    "points": 20,
    "title": "Mamba SSM for Vision",
    "company": "Research / Industry",
    "desc": "Implement Mamba SSM for Vision from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C85: Mamba SSM for Vision\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Mamba SSM for Vision\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C85: Mamba SSM for Vision'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Mamba SSM for Vision to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C86",
    "difficulty": "Medium",
    "tag": "Generative",
    "points": 20,
    "title": "MoE Gating for Vision",
    "company": "Research / Industry",
    "desc": "Implement MoE Gating for Vision from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C86: MoE Gating for Vision\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement MoE Gating for Vision\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C86: MoE Gating for Vision'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend MoE Gating for Vision to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C87",
    "difficulty": "Hard",
    "tag": "Segmentation",
    "points": 30,
    "title": "Ring Attention for Long Context",
    "company": "Research / Industry",
    "desc": "Implement Ring Attention for Long Context from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C87: Ring Attention for Long Context\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Ring Attention for Long Context\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C87: Ring Attention for Long Context'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Ring Attention for Long Context to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C88",
    "difficulty": "Medium",
    "tag": "Attention",
    "points": 20,
    "title": "Test-Time Augmentation",
    "company": "Research / Industry",
    "desc": "Implement Test-Time Augmentation from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C88: Test-Time Augmentation\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Test-Time Augmentation\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C88: Test-Time Augmentation'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Test-Time Augmentation to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C89",
    "difficulty": "Hard",
    "tag": "3D Rendering",
    "points": 30,
    "title": "EMA Weight Averaging",
    "company": "Research / Industry",
    "desc": "Implement EMA Weight Averaging from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C89: EMA Weight Averaging\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement EMA Weight Averaging\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C89: EMA Weight Averaging'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend EMA Weight Averaging to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C90",
    "difficulty": "Medium",
    "tag": "Diffusion",
    "points": 20,
    "title": "Label Smoothing Cross-Entropy",
    "company": "Research / Industry",
    "desc": "Implement Label Smoothing Cross-Entropy from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C90: Label Smoothing Cross-Entropy\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Label Smoothing Cross-Entropy\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C90: Label Smoothing Cross-Entropy'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Label Smoothing Cross-Entropy to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C91",
    "difficulty": "Medium",
    "tag": "Flow",
    "points": 20,
    "title": "Mixup and CutMix Augmentation",
    "company": "Research / Industry",
    "desc": "Implement Mixup and CutMix Augmentation from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C91: Mixup and CutMix Augmentation\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Mixup and CutMix Augmentation\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C91: Mixup and CutMix Augmentation'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Mixup and CutMix Augmentation to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C92",
    "difficulty": "Hard",
    "tag": "Tracking",
    "points": 30,
    "title": "RandAugment Policy Search",
    "company": "Research / Industry",
    "desc": "Implement RandAugment Policy Search from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C92: RandAugment Policy Search\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement RandAugment Policy Search\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C92: RandAugment Policy Search'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend RandAugment Policy Search to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C93",
    "difficulty": "Medium",
    "tag": "Calibration",
    "points": 20,
    "title": "AugMax Adversarial Augmentation",
    "company": "Research / Industry",
    "desc": "Implement AugMax Adversarial Augmentation from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C93: AugMax Adversarial Augmentation\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement AugMax Adversarial Augmentation\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C93: AugMax Adversarial Augmentation'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend AugMax Adversarial Augmentation to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C94",
    "difficulty": "Hard",
    "tag": "SSL",
    "points": 30,
    "title": "Self-Training Pseudo-Labels",
    "company": "Research / Industry",
    "desc": "Implement Self-Training Pseudo-Labels from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C94: Self-Training Pseudo-Labels\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Self-Training Pseudo-Labels\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C94: Self-Training Pseudo-Labels'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Self-Training Pseudo-Labels to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C95",
    "difficulty": "Medium",
    "tag": "Quantisation",
    "points": 20,
    "title": "Mean Teacher Semi-Supervised",
    "company": "Research / Industry",
    "desc": "Implement Mean Teacher Semi-Supervised from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C95: Mean Teacher Semi-Supervised\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement Mean Teacher Semi-Supervised\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C95: Mean Teacher Semi-Supervised'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend Mean Teacher Semi-Supervised to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C96",
    "difficulty": "Medium",
    "tag": "Optimisation",
    "points": 20,
    "title": "FixMatch Threshold Scheduling",
    "company": "Research / Industry",
    "desc": "Implement FixMatch Threshold Scheduling from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C96: FixMatch Threshold Scheduling\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement FixMatch Threshold Scheduling\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C96: FixMatch Threshold Scheduling'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend FixMatch Threshold Scheduling to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C97",
    "difficulty": "Hard",
    "tag": "Backprop",
    "points": 30,
    "title": "DataComp Curation Filter",
    "company": "Research / Industry",
    "desc": "Implement DataComp Curation Filter from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C97: DataComp Curation Filter\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement DataComp Curation Filter\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C97: DataComp Curation Filter'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend DataComp Curation Filter to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C98",
    "difficulty": "Medium",
    "tag": "Graph Search",
    "points": 20,
    "title": "LAION Quality Filtering",
    "company": "Research / Industry",
    "desc": "Implement LAION Quality Filtering from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C98: LAION Quality Filtering\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement LAION Quality Filtering\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C98: LAION Quality Filtering'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend LAION Quality Filtering to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  },
  {
    "id": "C99",
    "difficulty": "Hard",
    "tag": "Transformers",
    "points": 30,
    "title": "CLIP Score Image-Text Alignment",
    "company": "Research / Industry",
    "desc": "Implement CLIP Score Image-Text Alignment from scratch. Study the key equations, understand the intuition, then code a working solution that processes real images and produces visualisable output. Verify against a reference implementation.",
    "example": "See full solution below. Each step is annotated with the mathematical insight behind it.",
    "hint": "Break the problem into sub-steps: data preparation, core algorithm implementation, metric computation, and visualisation. Verify each step against a reference.",
    "fullSolution": "# C99: CLIP Score Image-Text Alignment\nimport numpy as np, torch, cv2, urllib.request\nimport matplotlib.pyplot as plt\n\n# Starter: load a real image\nurl = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg'\nraw = urllib.request.urlopen(url).read()\nimg = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)\nimg_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\nprint(f'Image shape: {img_rgb.shape}')\n\n# TODO: Implement CLIP Score Image-Text Alignment\n# Step 1: Preprocess / setup\n# Step 2: Core algorithm\n# Step 3: Evaluate / visualise\nplt.imshow(img_rgb); plt.axis('off')\nplt.title('C99: CLIP Score Image-Text Alignment'); plt.show()",
    "complexity": "Varies by sub-step. Analyse time and space complexity for each component.",
    "followup": "How would you extend CLIP Score Image-Text Alignment to video? What are the key bottlenecks for real-time deployment on edge hardware?"
  }
];

/* ================================================================
   150 QUIZ QUESTIONS (6 sections × 25 questions)
================================================================ */
const QUIZ_SECTIONS = [
  {
    title:"Section 1: Foundations of Computer Vision",
    questions:[
      {q:"What does a pixel value of 0 represent in a grayscale image?",options:["White","Black","Red","Transparent"],answer:1},
      {q:"What shape does a 640x480 RGB image have as a numpy array?",options:["(640,480)","(480,640)","(480,640,3)","(3,480,640)"],answer:2},
      {q:"Which colour space separates luminance from chrominance?",options:["RGB","BGR","LAB","HSV"],answer:2},
      {q:"What does CLAHE stand for?",options:["Contrast Limited Adaptive Histogram Equalisation","Colour Layered Automatic HE","Channel Linear Adaptive HE","Contrast Luminance Adaptive HE"],answer:0},
      {q:"Which edge detector uses two threshold values (hysteresis)?",options:["Sobel","Laplacian","Canny","Prewitt"],answer:2},
      {q:"Morphological erosion on a binary image:",options:["Expands white regions","Shrinks white regions","Detects edges","Fills holes"],answer:1},
      {q:"The Viola-Jones face detector relies on which image structure for speed?",options:["Gradient histogram","Integral image","Fourier transform","Wavelet"],answer:1},
      {q:"SIFT descriptors have how many dimensions?",options:["32","64","128","256"],answer:2},
      {q:"Which interpolation method is most accurate for image rotation?",options:["Nearest neighbour","Bilinear","Bicubic","Box filter"],answer:2},
      {q:"Gaussian blur in the spatial domain is equivalent to what in the frequency domain?",options:["High-pass filter","Band-pass filter","Low-pass filter","Notch filter"],answer:2},
      {q:"The Sobel operator computes:",options:["Second-order derivative","First-order derivative (gradient)","Laplacian","Hessian"],answer:1},
      {q:"What is the main advantage of ORB over SIFT?",options:["Higher accuracy","Much faster (binary descriptor)","Scale invariant","Patent-free and fast"],answer:3},
      {q:"In the morphological operation 'opening', what is the correct order?",options:["Dilation then erosion","Erosion then dilation","Erosion twice","Dilation twice"],answer:1},
      {q:"What does a 2D Discrete Fourier Transform represent?",options:["Pixel values over time","Spatial frequency content","Edge map","Colour histogram"],answer:1},
      {q:"Which metric measures the overlap between two bounding boxes?",options:["L2 distance","IoU","Dice","SSIM"],answer:1},
      {q:"PSNR is computed using which error measure?",options:["MAE","MSE","Huber","Cross-entropy"],answer:1},
      {q:"SSIM measures image similarity along which three dimensions?",options:["Hue, saturation, brightness","Luminance, contrast, structure","Frequency, amplitude, phase","Red, green, blue"],answer:1},
      {q:"What is the purpose of image normalisation (subtracting mean, dividing by std)?",options:["Increase brightness","Stabilise training and improve convergence","Remove noise","Detect edges"],answer:1},
      {q:"Connected components analysis is used for:",options:["Counting separate objects in a binary image","Computing image gradients","Detecting colours","Measuring blur"],answer:0},
      {q:"The Dark Channel Prior is used for:",options:["Face detection","Image dehazing","Super-resolution","Depth estimation"],answer:1},
      {q:"Template matching via Normalised Cross-Correlation (NCC) returns values in the range:",options:["[0, infinity)","[0, 255]","[-1, 1]","[0, 1]"],answer:2},
      {q:"Total Variation (TV) denoising promotes which type of solution?",options:["Smooth everywhere","Piecewise constant with sharp edges","Blurry edges","High frequency detail"],answer:1},
      {q:"Seam carving for content-aware resizing removes which type of seam?",options:["Horizontal seam with maximum energy","Vertical seam with minimum energy","Random pixel path","Highest-gradient path"],answer:1},
      {q:"Perceptual hashing (pHash) is based on which transform?",options:["DFT","DCT","Wavelet","SVD"],answer:1},
      {q:"Flood fill (paint bucket) uses which graph traversal algorithm?",options:["Dijkstra","A*","BFS or DFS","Bellman-Ford"],answer:2},
    ]
  },
  {
    title:"Section 2: Deep Learning for Vision",
    questions:[
      {q:"What property does max-pooling provide that is useful for classification?",options:["Exact position","Translation invariance","Scale invariance","Rotation invariance"],answer:1},
      {q:"Batch Normalisation normalises over which dimensions?",options:["Channels for each spatial position","The batch and spatial dimensions","Only the channel dimension","Only spatial dimensions"],answer:1},
      {q:"The skip connections in ResNet allow training of very deep networks because they solve:",options:["Overfitting","Vanishing gradient problem","Data imbalance","Mode collapse"],answer:1},
      {q:"Which activation function is most commonly used in hidden CNN layers today?",options:["Sigmoid","Tanh","ReLU/GELU","Softmax"],answer:2},
      {q:"Cross-entropy loss for classification is: -sum(",options:["y * log(p)","(y-p)^2","y * p","log(1-p)"],answer:0},
      {q:"The Adam optimiser maintains:",options:["Only gradient magnitude","First and second moment estimates of gradients","Learning rate only","Momentum only"],answer:1},
      {q:"Weight sharing in CNNs provides what major advantage?",options:["Higher accuracy","Translation equivariance and massive parameter reduction","Faster inference only","Better calibration"],answer:1},
      {q:"Dropout is applied during:",options:["Inference only","Training only","Both training and inference","Neither"],answer:1},
      {q:"Which pooling operation does EfficientNet use instead of max-pooling in the final layer?",options:["Average pooling","Max pooling","Stochastic pooling","Global average pooling"],answer:3},
      {q:"DepthWise Separable Convolution (MobileNet) reduces computation by approximately:",options:["2x","4x","8-9x","100x"],answer:2},
      {q:"What is the role of the softmax function in the output layer of a classifier?",options:["Compute gradients","Convert logits to probabilities summing to 1","Normalise features","Apply non-linearity"],answer:1},
      {q:"Transfer learning pre-trains a model on which dataset before fine-tuning?",options:["MNIST","COCO","ImageNet (or similar large dataset)","CIFAR-10"],answer:2},
      {q:"Data augmentation is primarily used to:",options:["Speed up training","Prevent overfitting by increasing effective dataset size","Improve validation speed","Reduce model size"],answer:1},
      {q:"Which normalisation works best with very small batch sizes?",options:["Batch Norm","Layer Norm","Instance Norm","Group Norm"],answer:1},
      {q:"The receptive field of a 5-layer CNN with 3x3 kernels (stride 1, no padding) is:",options:["3x3","5x5","9x9","11x11"],answer:3},
      {q:"MixUp augmentation creates training samples by:",options:["Random cropping","Linearly interpolating two images and their labels","Colour jitter","Random erasing"],answer:1},
      {q:"Label smoothing prevents models from becoming:",options:["Underfitted","Overconfident (overfitted to hard labels)","Too slow","Too small"],answer:1},
      {q:"Knowledge distillation trains a student by matching:",options:["Hard labels only","Soft probability outputs of a teacher model","The teacher's architecture","Only the final layer"],answer:1},
      {q:"Cosine annealing as a learning rate schedule:",options:["Increases LR linearly","Decreases LR following a cosine curve","Keeps LR constant","Randomly varies LR"],answer:1},
      {q:"The purpose of the 1x1 convolution (pointwise conv) in MobileNet is:",options:["Spatial feature extraction","Cross-channel mixing (change number of channels)","Downsampling","Upsampling"],answer:1},
      {q:"Which loss function is standard for multi-label classification (multiple classes can be true)?",options:["Softmax + CE","Sigmoid + Binary CE per class","Triplet loss","Dice loss"],answer:1},
      {q:"Dilated convolution with rate r=2 on a 3x3 kernel has what effective kernel size?",options:["3x3","5x5","7x7","9x9"],answer:1},
      {q:"Gradient clipping prevents:",options:["Vanishing gradients","Exploding gradients","Data imbalance","Overfitting"],answer:1},
      {q:"The ELU activation function advantage over ReLU is:",options:["Faster computation","Non-zero mean activations, smoother gradient for negatives","Uses less memory","No hyperparameters"],answer:1},
      {q:"In the context of object detection, what does 'anchor-free' mean?",options:["No bounding boxes predicted","No predefined anchor boxes used as reference","No backbone needed","No NMS needed"],answer:1},
    ]
  },
  {
    title:"Section 3: Detection, Segmentation and Tracking",
    questions:[
      {q:"What does IoU stand for in object detection?",options:["Index of Uniformity","Intersection over Union","Integral of Uncertainty","Index of Utility"],answer:1},
      {q:"Non-Maximum Suppression (NMS) is used to:",options:["Find missing objects","Remove duplicate overlapping detections","Improve segmentation","Speed up inference"],answer:1},
      {q:"Faster R-CNN uses which module to generate region proposals?",options:["FPN","RPN","NMS","ROI head"],answer:1},
      {q:"YOLO stands for:",options:["You Only Learn Once","You Only Look Once","Your Object Localisation Output","Your Only Linear Output"],answer:1},
      {q:"Focal Loss in RetinaNet addresses which problem?",options:["Mode collapse","Extreme class imbalance between foreground and background","Vanishing gradient","Mode collapse"],answer:1},
      {q:"RoIAlign (vs RoIPool) prevents which issue?",options:["Slow inference","Misalignment caused by quantisation of spatial coordinates","High memory usage","Gradient explosion"],answer:1},
      {q:"DETR eliminates which two traditional detection components?",options:["Backbone and FPN","Anchor boxes and NMS","Loss function and optimizer","Batch norm and dropout"],answer:1},
      {q:"In Mask R-CNN, the mask branch predicts:",options:["Class probabilities","A binary segmentation mask for each detected region","Bounding box offsets","Keypoints"],answer:1},
      {q:"Semantic segmentation differs from instance segmentation in that it:",options:["Is faster","Does not distinguish individual instances","Requires no training data","Uses no neural network"],answer:1},
      {q:"Panoptic segmentation assigns which two types of labels?",options:["Class and confidence","Instance ID (for things) and semantic class (for stuff)","Depth and class","Box and mask"],answer:1},
      {q:"The U-Net architecture is known for its:",options:["Attention mechanism","Encoder-decoder with skip connections","Dilated convolutions","Residual blocks"],answer:1},
      {q:"Dilated convolutions in DeepLab expand the receptive field while:",options:["Increasing resolution","Maintaining spatial resolution","Reducing parameters","Adding attention"],answer:1},
      {q:"ASPP in DeepLab uses:",options:["Anchors at multiple scales","Parallel dilated convolutions at multiple rates + global pooling","Multiple backbones","Attention across channels"],answer:1},
      {q:"SAM (Segment Anything Model) uses which type of prompt for segmentation?",options:["Only text","Only boxes","Points, boxes, or text","Only images"],answer:2},
      {q:"What does mIoU measure in segmentation?",options:["Mean image quality","Mean Intersection over Union across all classes","Max IoU","Median per-image IoU"],answer:1},
      {q:"SORT (Simple Online and Realtime Tracking) uses which two components?",options:["Deep features + Hungarian","Kalman filter + Hungarian algorithm","Optical flow + NMS","RNN + attention"],answer:1},
      {q:"ByteTrack improves over SORT by:",options:["Using a better backbone","Also associating low-confidence detections in a second pass","Using larger anchors","Training on more data"],answer:1},
      {q:"The Dice coefficient is equivalent to:",options:["IoU","F1 score on pixel masks","Precision","Recall"],answer:1},
      {q:"CRF (Conditional Random Field) post-processing in DeepLab was used to:",options:["Speed up training","Refine segment boundaries using pixel pairwise potentials","Reduce model size","Handle class imbalance"],answer:1},
      {q:"In instance segmentation, the mask resolution output by Mask R-CNN is typically:",options:["Full image resolution","28x28 per RoI","256x256","Same as input"],answer:1},
      {q:"FPN (Feature Pyramid Network) improves detection by:",options:["Using more anchors","Building a multi-scale feature hierarchy for objects of different sizes","Faster NMS","Better backbone"],answer:1},
      {q:"CondInst produces instance masks by:",options:["Fixed prototype masks","Dynamically generated filters conditioned on each instance","Copying from backbone","Template matching"],answer:1},
      {q:"SOLOv2 predicts instance masks without:",options:["A backbone","RoI operations or bounding boxes","A decoder","Any loss function"],answer:1},
      {q:"The Panoptic Quality (PQ) metric is the product of:",options:["IoU and AP","Recognition Quality (RQ) and Segmentation Quality (SQ)","Dice and precision","mIoU and accuracy"],answer:1},
      {q:"ByteTrack achieves state-of-the-art MOT by:",options:["Using heavier backbones","A two-stage association that rescues low-confidence detections via appearance","3D sensing","Graph neural networks"],answer:1},
    ]
  },
  {
    title:"Section 4: Restoration, Generation and Diffusion",
    questions:[
      {q:"Image restoration is considered ill-posed because:",options:["It requires too much compute","Multiple clean images can produce the same degraded observation","Labels are hard to obtain","The domain shifts often"],answer:1},
      {q:"DnCNN uses which learning strategy to simplify the denoising task?",options:["Direct image reconstruction","Residual learning (predict noise, subtract from input)","GAN adversarial training","Attention-based weighting"],answer:1},
      {q:"PSNR in dB is computed as:",options:["10*log10(MAX^2 / MSE)","20*log10(MAX / MSE)","MSE / MAX","log(MSE)"],answer:0},
      {q:"SSIM measures image quality along three dimensions:",options:["RGB channels","Luminance, contrast, structure","Low/mid/high frequency","Spatial/temporal/spectral"],answer:1},
      {q:"Real-ESRGAN extends ESRGAN to handle:",options:["Only bicubic downsampling","Real-world complex degradations (blur+noise+JPEG chaining)","Video super-resolution","3D super-resolution"],answer:1},
      {q:"SRGAN produces sharper results than MSE-based SR because:",options:["It uses a larger network","Perceptual and adversarial losses encourage realistic high-frequency detail","It uses more training data","It upsamples more gradually"],answer:1},
      {q:"The atmospheric scattering model for haze is:",options:["I = J + A","I = J*t + A*(1-t)","I = J*A","I = J - t"],answer:1},
      {q:"The Dark Channel Prior states that in haze-free images:",options:["All channels are bright","At least one channel is very dark in most local patches","Brightness is uniform","Edges are sharp"],answer:1},
      {q:"Which loss function in GANs replaces JS divergence with Wasserstein distance?",options:["Standard GAN loss","WGAN-GP","Least-squares GAN","Hinge loss"],answer:1},
      {q:"StyleGAN's mapping network produces which vector space?",options:["Z space","W space (disentangled latent)","Style space","Noise space"],answer:1},
      {q:"The reparameterisation trick in VAEs allows:",options:["Better reconstruction","Backpropagation through stochastic sampling","Faster inference","More stable training"],answer:1},
      {q:"In DDPM, the forward process:",options:["Removes noise","Gradually adds Gaussian noise over T steps","Learns to denoise","Applies adversarial training"],answer:1},
      {q:"DDIM speeds up diffusion sampling by:",options:["Using a smaller U-Net","Using deterministic non-Markovian trajectories","Reducing image resolution","Using distillation"],answer:1},
      {q:"Classifier-Free Guidance in diffusion models scales:",options:["Only the conditioned prediction","The interpolation between conditioned and unconditioned predictions","The noise schedule","The learning rate"],answer:1},
      {q:"Latent Diffusion Models (Stable Diffusion) run diffusion in:",options:["Pixel space at full resolution","A compressed latent space from a VQ-VAE","Frequency domain","A GAN latent space"],answer:1},
      {q:"ControlNet adds spatial conditioning to diffusion via:",options:["Fine-tuning the full U-Net","Zero-convolution adapters on a copied encoder","Attention layers","Cross-entropy fine-tuning"],answer:1},
      {q:"FID (Frechet Inception Distance) measures:",options:["Image sharpness","Distance between real and generated image feature distributions","Training speed","Resolution"],answer:1},
      {q:"CycleGAN enables image-to-image translation:",options:["With paired data only","Without paired data, using cycle consistency loss","With text supervision","Using only discriminators"],answer:1},
      {q:"VQ-VAE uses a discrete codebook instead of a continuous latent space, enabling:",options:["Better gradients","Tokenised image representation for autoregressive generation","Faster training","Smaller models"],answer:1},
      {q:"Progressive GAN trains by:",options:["Full resolution from the start","Starting at low resolution and progressively adding layers","Using multiple discriminators","Training with paired data"],answer:1},
      {q:"Which metric is specifically designed for perceptual image quality using deep features?",options:["PSNR","SSIM","LPIPS","FID"],answer:2},
      {q:"Inpainting with LaMa uses which convolution type to achieve global context from layer 1?",options:["Dilated conv","Partial conv","Fast Fourier Convolution (global receptive field)","Deformable conv"],answer:2},
      {q:"Noise2Void trains a denoiser without clean target images by:",options:["Paired noisy images","Blind-spot convolutions that mask the centre pixel during training","Using a teacher model","Adversarial training"],answer:1},
      {q:"The DiT architecture (Diffusion Transformer) replaces the U-Net backbone with:",options:["A GAN discriminator","A Vision Transformer","An RNN","A ResNet"],answer:1},
      {q:"Score distillation sampling (SDS) allows:",options:["Faster DPM training","Using a pretrained diffusion model as a prior for 3D or other optimisation tasks","Training with no data","Better FID scores"],answer:1},
    ]
  },
  {
    title:"Section 5: Multimodal and Advanced Topics",
    questions:[
      {q:"CLIP is trained with which objective?",options:["Supervised classification","Contrastive learning on image-text pairs","Generative modelling","Pixel reconstruction"],answer:1},
      {q:"Vision Transformer (ViT) processes images by:",options:["Hierarchical convolutions","Splitting into patches and treating them as sequence tokens","Pixel-by-pixel RNN","Graph convolutions"],answer:1},
      {q:"Swin Transformer achieves linear complexity via:",options:["Sparse attention","Local window attention with shifting","Random masking","Low-rank approximation"],answer:1},
      {q:"BLIP-2 connects a frozen image encoder to an LLM via:",options:["Direct fine-tuning","Q-Former (Querying Transformer) bottleneck","Cross-attention only","Adapter layers"],answer:1},
      {q:"DINOv2 learns visual features without labels using:",options:["Supervised ImageNet","Self-distillation with momentum teacher on curated web images","GAN training","Text supervision"],answer:1},
      {q:"In VQA, the Bottom-Up Top-Down (ButD) attention uses features from:",options:["The full image CNN","Faster R-CNN detected object regions","Pixel grids","Depth maps"],answer:1},
      {q:"NeRF represents a scene as a function that takes (x,y,z,theta,phi) and outputs:",options:["Depth only","RGB colour and volume density","Semantic label","Surface normal"],answer:1},
      {q:"3D Gaussian Splatting achieves real-time rendering by:",options:["Running NeRF on GPU","Tile-based rasterisation of differentiable 3D Gaussians","Voxel caching","TSDF fusion"],answer:1},
      {q:"PointNet achieves permutation invariance via:",options:["Sorting input points","Symmetric aggregation (max-pooling) over all point features","Graph convolution","Attention"],answer:1},
      {q:"Optical flow estimation assumes which constraint between frames?",options:["Depth constancy","Brightness constancy","Texture constancy","Edge constancy"],answer:1},
      {q:"RAFT (optical flow) uses which novel data structure?",options:["Multi-scale pyramid","4D all-pairs feature correlation volume with iterative GRU updates","Attention maps","Sparse keypoints"],answer:1},
      {q:"Autonomous driving 3D detection in BEV space stands for:",options:["Below-Eye View","Bird's Eye View (top-down projection)","Beyond Edge View","Binocular Enhanced Vision"],answer:1},
      {q:"The nuScenes detection metric NDS combines:",options:["Only mAP","mAP with translation, scale, orientation, attribute, and velocity errors","Precision and recall","IoU and confidence"],answer:1},
      {q:"SlowFast networks process video with:",options:["One pathway at medium frame rate","Two pathways: Slow (high spatial, low temporal) and Fast (low spatial, high temporal)","Three timescales","Bidirectional RNN"],answer:1},
      {q:"Action recognition with skeleton graphs uses:",options:["Standard CNNs on RGB","Graph Convolutional Networks on body joint topology","RNNs on optical flow","Transformers on pixel grids"],answer:1},
      {q:"Depth Anything v2 achieves strong generalization by:",options:["Using better architecture","Semi-supervised training on 62M images with pseudo-labels","Larger batch size","More convolutions"],answer:1},
      {q:"Lane detection with SCNN uses:",options:["Graph neural networks","Message passing across rows and columns of feature maps","Attention over proposals","Template matching"],answer:1},
      {q:"Visual anomaly detection with PatchCore stores:",options:["All training features","A coreset subset of training features in a memory bank","Model weights only","Centroid per class"],answer:1},
      {q:"Person re-identification (ReID) differs from face recognition in that it:",options:["Is easier","Works on full body without face, across non-overlapping cameras","Requires no metric learning","Uses 3D sensors"],answer:1},
      {q:"Document Understanding model Donut avoids OCR by:",options:["Using OCR as preprocessing","Directly parsing document images with a vision encoder-decoder to JSON","Using LLMs only","Preprocessing with SIFT"],answer:1},
      {q:"EfficientDet scales object detection using:",options:["Only depth scaling","Compound scaling of backbone, FPN, and head simultaneously","Random NAS","Data augmentation"],answer:1},
      {q:"SigLIP improves over CLIP's contrastive loss by:",options:["Adding more text encoders","Using sigmoid pairwise loss instead of softmax over all pairs","Training on more data","Adding a generative objective"],answer:1},
      {q:"Instant-NGP achieves fast NeRF training via:",options:["Smaller network","Multi-resolution hash encoding for positional encoding","GPU parallelism only","Fewer training views"],answer:1},
      {q:"The occupancy prediction task in autonomous driving outputs:",options:["2D bounding boxes","A 3D voxel grid with semantic labels per voxel","Depth map only","Lane masks"],answer:1},
      {q:"CellViT for computational pathology adapts which foundation model for cell segmentation?",options:["ResNet","SAM (Segment Anything Model)","CLIP","DINOv2"],answer:1},
    ]
  },
  {
    title:"Section 6: Practical CV and Deployment",
    questions:[
      {q:"Post-Training Quantisation (PTQ) converts model weights from float32 to:",options:["float64","bfloat16 or int8","int32","float16 only"],answer:1},
      {q:"Knowledge distillation trains the student using:",options:["Hard one-hot labels only","Soft teacher logits (probability distribution over classes)","Only teacher intermediate features","Random labels"],answer:1},
      {q:"ONNX is used for:",options:["Training models","Vendor-neutral model exchange and deployment across runtimes","Dataset management","Hyperparameter tuning"],answer:1},
      {q:"TensorRT is optimised for inference on:",options:["CPUs","NVIDIA GPUs","Mobile CPUs","FPGA"],answer:1},
      {q:"Data drift refers to:",options:["Gradient instability","Input distribution at deployment differing from training distribution","Model weight drift","Label noise"],answer:1},
      {q:"Structured pruning removes:",options:["Individual weights below threshold","Entire channels, heads, or layers (hardware-friendly)","Batch norm parameters","Biases only"],answer:1},
      {q:"Quantisation-Aware Training (QAT) vs PTQ:",options:["QAT is faster","QAT simulates quantisation during training, achieving higher accuracy","PTQ always beats QAT","They are identical"],answer:1},
      {q:"A/B testing in model deployment is used for:",options:["Debugging training","Safely comparing new and old model versions with a subset of live traffic","Data collection","Architecture search"],answer:1},
      {q:"MLflow is primarily used for:",options:["Data preprocessing","Experiment tracking, model versioning, and reproducibility","Serving models at scale","Neural architecture search"],answer:1},
      {q:"The Triton Inference Server supports:",options:["Only PyTorch","Multiple frameworks (TF, PyTorch, ONNX, TensorRT) with dynamic batching","Only TFLite","CPU inference only"],answer:1},
      {q:"GFLOPs (Giga Floating-Point Operations) is used to measure:",options:["Model accuracy","Model computational complexity","Training speed only","Memory usage"],answer:1},
      {q:"Test-Time Augmentation (TTA) improves prediction by:",options:["Training longer","Averaging predictions over multiple augmented versions of the test image","Using more labels","Adding dropout"],answer:1},
      {q:"Expected Calibration Error (ECE) measures:",options:["Accuracy on easy examples","How well model confidence scores match actual accuracy","Loss value","Speed"],answer:1},
      {q:"Canary deployment routes:",options:["All traffic to new model","A small percentage of traffic to the new model for safe rollout","Traffic based on geography","Only bot traffic"],answer:1},
      {q:"CoreML is specifically optimised for inference on:",options:["NVIDIA GPUs","Android devices","Apple devices (iOS, macOS)","x86 CPUs"],answer:2},
      {q:"Dynamic batching in serving (Triton) improves:",options:["Accuracy","GPU utilisation by grouping multiple concurrent requests","Model size","Training speed"],answer:1},
      {q:"The primary purpose of Grad-CAM is:",options:["Speed up inference","Visualise which image regions influenced the model's class prediction","Compress models","Augment data"],answer:1},
      {q:"SHAP values for image attribution are based on:",options:["Gradient magnitude","Shapley values from cooperative game theory","Attention weights","Feature correlation"],answer:1},
      {q:"OpenVINO (Intel) is optimised for inference on:",options:["NVIDIA GPUs","Intel CPUs, integrated GPUs, and VPUs","ARM CPUs","TPUs"],answer:1},
      {q:"Neural Architecture Search (NAS) automatically finds:",options:["The best training data","Pareto-optimal network architectures for a given hardware-accuracy tradeoff","The best loss function","The best augmentation policy"],answer:1},
      {q:"Which evaluation split is used to tune hyperparameters (not the test set)?",options:["Training set","Validation set","Test set","Holdout set"],answer:1},
      {q:"Confusion matrix diagonal values represent:",options:["Errors per class","Correct predictions per class (True Positives)","Total samples","FP rates"],answer:1},
      {q:"The Receiver Operating Characteristic (ROC) curve plots:",options:["Precision vs recall","True Positive Rate vs False Positive Rate at various thresholds","Loss vs accuracy","PSNR vs SSIM"],answer:1},
      {q:"BentoML is a framework for:",options:["Training large models","Packaging and serving ML models as production-grade APIs","Data labelling","Feature engineering"],answer:1},
      {q:"Ray Serve is used for:",options:["Distributed training only","Scalable model serving and inference with actor-based architecture","Data preprocessing","Experiment tracking"],answer:1},
    ]
  },
];

/* ================================================================
   MAIN APP COMPONENT
================================================================ */
export default function App() {
  const [tab, setTab]           = useState("domains");
  const [selDomain, setSelDomain] = useState(null);
  const [selMod, setSelMod]     = useState(null);
  const [selCh, setSelCh]       = useState(null);
  const [showSol, setShowSol]   = useState(false);
  const [quizSec, setQuizSec]   = useState(0);
  const [quizIdx, setQuizIdx]   = useState(0);
  const [quizAns, setQuizAns]   = useState({});
  const [quizDone, setQuizDone] = useState(false);
  const [searchQ, setSearchQ]   = useState("");
  const [chFilter, setChFilter] = useState("All");
  const [quizScore, setQuizScore] = useState(null);

  const TABS = [
    {id:"domains",  label:"30 Domains"},
    {id:"modules",  label:"10 Modules"},
    {id:"challenges",label:"100 Challenges"},
    {id:"quiz",     label:"150 Quiz"},
  ];

  const S = {
    app: { background:P.bg, minHeight:"100vh", color:P.text,
           fontFamily:"'DM Mono','Fira Code',monospace", fontSize:14 },
    nav: { background:P.surface, borderBottom:`1px solid ${P.border}`,
           padding:"0 24px", display:"flex", alignItems:"center", gap:8, overflowX:"auto",
           position:"sticky", top:0, zIndex:100, boxShadow:"0 2px 20px #000a" },
    logo: { color:P.accent1, fontWeight:700, fontSize:18, marginRight:16,
            letterSpacing:"-0.5px", whiteSpace:"nowrap" },
    tab: (active) => ({
      padding:"16px 20px", cursor:"pointer", borderBottom: active?`3px solid ${P.accent1}`:"3px solid transparent",
      color: active ? P.accent1 : P.muted, fontWeight: active?700:400,
      background:"none", border:"none", fontSize:13, letterSpacing:1,
      transition:"all 0.2s", whiteSpace:"nowrap",
    }),
    content: { maxWidth:1400, margin:"0 auto", padding:"32px 20px" },
    card: { background:P.card, border:`1px solid ${P.border}`, borderRadius:12,
            padding:24, marginBottom:16, cursor:"pointer", transition:"all 0.2s" },
    grid: { display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(300px,1fr))", gap:16 },
    pill: (col) => ({ display:"inline-block", background:col+"22", color:col, border:`1px solid ${col}44`,
                      borderRadius:20, padding:"3px 12px", fontSize:11, fontWeight:700, letterSpacing:1 }),
    badge: (col) => ({ display:"inline-block", background:col+"33", color:col,
                       borderRadius:6, padding:"2px 8px", fontSize:11, fontWeight:600 }),
    code: { background:"#020814", borderRadius:8, padding:20, overflowX:"auto",
            fontSize:12, lineHeight:1.7, color:"#a8d8ea", whiteSpace:"pre",
            fontFamily:"'Fira Code',monospace", border:`1px solid ${P.border}` },
    input: { background:P.card, border:`1px solid ${P.border}`, borderRadius:8,
             color:P.text, padding:"10px 16px", fontSize:13, outline:"none", width:"100%",
             boxSizing:"border-box", fontFamily:"inherit" },
    btn: (col) => ({ background:col, color:"#000", border:"none", borderRadius:8,
                     padding:"10px 20px", cursor:"pointer", fontWeight:700, fontSize:12,
                     letterSpacing:1, transition:"opacity 0.2s" }),
    section: { background:P.surface, borderRadius:8, padding:20, marginBottom:16,
               border:`1px solid ${P.border}` },
  };

  // ---- DOMAINS VIEW ----
  const DomainsView = () => {
    if (selDomain) {
      const d = selDomain;
      return (
        <div>
          <button style={{...S.btn(P.accent1), marginBottom:24}} onClick={()=>setSelDomain(null)}>
            ← Back to Domains
          </button>
          <div style={{display:"flex",alignItems:"center",gap:16,marginBottom:24}}>
            <div style={{width:48,height:48,borderRadius:12,background:d.color+"33",
                         display:"flex",alignItems:"center",justifyContent:"center",
                         fontSize:22,fontWeight:700,color:d.color}}>{d.id}</div>
            <div>
              <h1 style={{margin:0,color:d.color,fontSize:28}}>{d.name}</h1>
              <div style={{color:P.muted,marginTop:4}}>{d.tagline}</div>
            </div>
          </div>
          <div style={{...S.section}}>
            <h3 style={{color:P.accent2,margin:"0 0 12px"}}>Theory</h3>
            <p style={{lineHeight:1.8,color:P.text,margin:0}}>{d.theory}</p>
          </div>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:12,marginBottom:16}}>
            {[["Architectures",d.architectures,P.accent3],["Metrics",d.metrics,P.accent4],["Datasets",d.datasets,P.accent5]].map(([title,items,col])=>(
              <div key={title} style={{...S.section}}>
                <h4 style={{color:col,margin:"0 0 10px"}}>{title}</h4>
                {items.map(it=><div key={it} style={{...S.badge(col),margin:"3px 4px 3px 0",display:"inline-block"}}>{it}</div>)}
              </div>
            ))}
          </div>
          <div style={S.section}>
            <h3 style={{color:P.accent6,margin:"0 0 12px"}}>Google Colab Code</h3>
            <div style={S.code}>{d.colab}</div>
          </div>
        </div>
      );
    }
    const filtered = DOMAINS.filter(d =>
      d.name.toLowerCase().includes(searchQ.toLowerCase()) ||
      d.theory.toLowerCase().includes(searchQ.toLowerCase())
    );
    return (
      <div>
        <h2 style={{color:P.accent1,marginBottom:8}}>30 Computer Vision Domains</h2>
        <p style={{color:P.muted,marginBottom:20}}>From classical image processing to state-of-the-art deep learning. Click any domain for full theory, architectures, datasets, and runnable Colab code.</p>
        <input style={{...S.input,marginBottom:20,maxWidth:400}} placeholder="Search domains..."
               value={searchQ} onChange={e=>setSearchQ(e.target.value)} />
        <div style={S.grid}>
          {filtered.map((d,i)=>(
            <div key={d.id} style={{...S.card, borderLeft:`4px solid ${d.color}`, borderColor:P.border}}
                 onClick={()=>setSelDomain(d)}
                 onMouseEnter={e=>{e.currentTarget.style.background=P.card2;e.currentTarget.style.borderLeft=`4px solid ${d.color}`}}
                 onMouseLeave={e=>{e.currentTarget.style.background=P.card;e.currentTarget.style.borderLeft=`1px solid ${P.border}`}}>
              <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:12}}>
                <div style={{width:36,height:36,borderRadius:8,background:d.color+"22",
                             display:"flex",alignItems:"center",justifyContent:"center",
                             color:d.color,fontWeight:700,fontSize:14}}>{d.id}</div>
                <div>
                  <div style={{fontWeight:700,color:P.text}}>{d.name}</div>
                  <div style={{color:P.muted,fontSize:11,marginTop:2}}>{d.tagline}</div>
                </div>
              </div>
              <div style={{...S.pill(d.color),marginBottom:10}}>{d.architectures[0]} + {d.architectures.length-1} more</div>
              <div style={{color:P.muted,fontSize:12,lineHeight:1.6}}>{d.theory.slice(0,120)}...</div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // ---- MODULES VIEW ----
  const ModulesView = () => {
    if (selMod !== null) {
      const m = MODULES[selMod];
      return (
        <div>
          <button style={{...S.btn(P.accent2),marginBottom:24}} onClick={()=>setSelMod(null)}>← Modules</button>
          <div style={{marginBottom:24}}>
            <div style={{display:"flex",gap:12,alignItems:"center",marginBottom:8}}>
              <div style={{...S.pill(m.color)}}>{m.level}</div>
              <div style={{color:P.muted,fontSize:12}}>⏱ {m.time}</div>
            </div>
            <h1 style={{margin:"0 0 8px",color:m.color,fontSize:26}}>Module {m.id}: {m.title}</h1>
          </div>
          {m.sections.map((s,i)=>(
            <div key={i} style={S.section}>
              <h3 style={{color:C(i),margin:"0 0 12px"}}>{s.heading}</h3>
              <p style={{lineHeight:1.8,margin:0}}>{s.body}</p>
            </div>
          ))}
          <div style={S.section}>
            <h3 style={{color:P.accent5,margin:"0 0 12px"}}>Python Code (Colab-Ready)</h3>
            <div style={S.code}>{m.code}</div>
          </div>
        </div>
      );
    }
    return (
      <div>
        <h2 style={{color:P.accent2,marginBottom:8}}>10 Learning Modules</h2>
        <p style={{color:P.muted,marginBottom:24}}>Structured learning path from absolute beginner to advanced. Each module includes detailed theory and fully runnable Python code.</p>
        <div style={{display:"flex",flexDirection:"column",gap:12}}>
          {MODULES.map((m,i)=>(
            <div key={m.id} style={{...S.card,display:"flex",alignItems:"center",gap:20,padding:20}}
                 onClick={()=>setSelMod(i)}
                 onMouseEnter={e=>e.currentTarget.style.background=P.card2}
                 onMouseLeave={e=>e.currentTarget.style.background=P.card}>
              <div style={{width:52,height:52,borderRadius:12,background:m.color+"22",
                           display:"flex",alignItems:"center",justifyContent:"center",
                           color:m.color,fontWeight:700,fontSize:18,flexShrink:0}}>M{m.id}</div>
              <div style={{flex:1}}>
                <div style={{fontWeight:700,fontSize:16,marginBottom:4}}>{m.title}</div>
                <div style={{display:"flex",gap:10}}>
                  <div style={{...S.pill(m.color)}}>{m.level}</div>
                  <div style={{color:P.muted,fontSize:12,alignSelf:"center"}}>⏱ {m.time}</div>
                  <div style={{color:P.muted,fontSize:12,alignSelf:"center"}}>{m.sections.length} sections</div>
                </div>
              </div>
              <div style={{color:P.accent1,fontSize:18}}>→</div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // ---- CHALLENGES VIEW ----
  const ChallengesView = () => {
    const DIFFS = ["All","Easy","Medium","Hard"];
    const filtered = CHALLENGES.filter(c =>
      (chFilter==="All" || c.difficulty===chFilter) &&
      (c.title.toLowerCase().includes(searchQ.toLowerCase()) ||
       c.tag.toLowerCase().includes(searchQ.toLowerCase()) ||
       c.desc.toLowerCase().includes(searchQ.toLowerCase()))
    );
    const diffColor = {"Easy":P.accent3,"Medium":P.accent5,"Hard":P.accent9};
    if (selCh) {
      const c = selCh;
      return (
        <div>
          <button style={{...S.btn(P.accent3),marginBottom:24}} onClick={()=>{setSelCh(null);setShowSol(false);}}>← Challenges</button>
          <div style={{display:"flex",flexWrap:"wrap",gap:10,marginBottom:20,alignItems:"center"}}>
            <div style={{...S.pill(diffColor[c.difficulty]||P.accent5)}}>{c.difficulty}</div>
            <div style={{...S.badge(P.accent1)}}>{c.tag}</div>
            <div style={{...S.badge(P.accent6)}}>{c.points} pts</div>
            <div style={{color:P.muted,fontSize:12}}>{c.company}</div>
          </div>
          <h1 style={{margin:"0 0 16px",fontSize:24}}>{c.id}: {c.title}</h1>
          <div style={S.section}>
            <h4 style={{color:P.accent2,margin:"0 0 10px"}}>Problem Description</h4>
            <p style={{lineHeight:1.8,margin:0}}>{c.desc}</p>
          </div>
          <div style={S.section}>
            <h4 style={{color:P.accent4,margin:"0 0 10px"}}>Example</h4>
            <div style={S.code}>{c.example}</div>
          </div>
          <div style={S.section}>
            <h4 style={{color:P.accent5,margin:"0 0 10px"}}>Hint</h4>
            <p style={{color:P.muted,margin:0,lineHeight:1.8}}>{c.hint}</p>
          </div>
          <button style={{...S.btn(P.accent6),marginBottom:16}} onClick={()=>setShowSol(!showSol)}>
            {showSol?"Hide Solution":"Show Full Solution"}
          </button>
          {showSol && (
            <div>
              <div style={S.section}>
                <h4 style={{color:P.accent6,margin:"0 0 10px"}}>Full Solution (with Image Loading)</h4>
                <div style={S.code}>{c.fullSolution}</div>
              </div>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
                <div style={S.section}>
                  <h4 style={{color:P.accent7,margin:"0 0 10px"}}>Complexity</h4>
                  <div style={{color:P.muted,fontFamily:"'Fira Code',monospace",fontSize:12}}>{c.complexity}</div>
                </div>
                <div style={S.section}>
                  <h4 style={{color:P.accent8,margin:"0 0 10px"}}>Follow-up Question</h4>
                  <p style={{color:P.muted,margin:0,lineHeight:1.6}}>{c.followup}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      );
    }
    const counts = {Easy:CHALLENGES.filter(c=>c.difficulty==="Easy").length,
                    Medium:CHALLENGES.filter(c=>c.difficulty==="Medium").length,
                    Hard:CHALLENGES.filter(c=>c.difficulty==="Hard").length};
    return (
      <div>
        <h2 style={{color:P.accent3,marginBottom:8}}>100 LeetCode-Style Challenges</h2>
        <div style={{display:"flex",gap:16,marginBottom:20,flexWrap:"wrap"}}>
          {Object.entries(counts).map(([d,n])=>(
            <div key={d} style={{...S.badge(diffColor[d]),padding:"6px 14px",fontSize:12}}>{n} {d}</div>
          ))}
        </div>
        <div style={{display:"flex",gap:12,marginBottom:20,flexWrap:"wrap"}}>
          <input style={{...S.input,maxWidth:300}} placeholder="Search challenges..."
                 value={searchQ} onChange={e=>setSearchQ(e.target.value)} />
          <div style={{display:"flex",gap:8}}>
            {DIFFS.map(d=>(
              <button key={d} style={{...S.btn(chFilter===d?P.accent3:P.border),
                                      color:chFilter===d?"#000":P.muted, fontSize:11}}
                      onClick={()=>setChFilter(d)}>{d}</button>
            ))}
          </div>
        </div>
        <div style={{...S.grid}}>
          {filtered.map(c=>(
            <div key={c.id} style={{...S.card}} onClick={()=>setSelCh(c)}
                 onMouseEnter={e=>e.currentTarget.style.background=P.card2}
                 onMouseLeave={e=>e.currentTarget.style.background=P.card}>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:8}}>
                <div style={{...S.pill(diffColor[c.difficulty]||P.accent5)}}>{c.difficulty}</div>
                <div style={{color:P.accent6,fontWeight:700,fontSize:13}}>{c.points} pts</div>
              </div>
              <div style={{fontWeight:700,marginBottom:6,fontSize:14}}>{c.id}: {c.title}</div>
              <div style={{...S.badge(P.accent1),marginBottom:8}}>{c.tag}</div>
              <div style={{color:P.muted,fontSize:11,lineHeight:1.6}}>{c.desc.slice(0,100)}...</div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // ---- QUIZ VIEW ----
  const QuizView = () => {
    const sec = QUIZ_SECTIONS[quizSec];
    const q   = sec.questions[quizIdx];
    const key = `${quizSec}-${quizIdx}`;
    const answered = quizAns[key] !== undefined;

    const submitQuiz = () => {
      let correct = 0, total = 0;
      QUIZ_SECTIONS.forEach((s,si)=>{
        s.questions.forEach((q,qi)=>{
          total++;
          if (quizAns[`${si}-${qi}`]===q.answer) correct++;
        });
      });
      setQuizScore({correct,total});
      setQuizDone(true);
    };

    if (quizDone && quizScore) {
      const pct = Math.round(quizScore.correct/quizScore.total*100);
      return (
        <div style={{textAlign:"center",padding:"60px 20px"}}>
          <div style={{fontSize:72,marginBottom:16}}>{pct>=80?"🏆":pct>=60?"🎯":"📚"}</div>
          <h2 style={{color:pct>=80?P.accent3:pct>=60?P.accent5:P.accent9,fontSize:32,marginBottom:8}}>
            {quizScore.correct}/{quizScore.total} Correct ({pct}%)
          </h2>
          <div style={{color:P.muted,marginBottom:32,fontSize:16}}>
            {pct>=80?"Excellent! You have strong CV foundations.":
             pct>=60?"Good effort! Review the sections you missed.":
             "Keep studying the modules and try again."}
          </div>
          <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:12,maxWidth:600,margin:"0 auto 32px"}}>
            {QUIZ_SECTIONS.map((s,si)=>{
              const correct=s.questions.filter((_,qi)=>quizAns[`${si}-${qi}`]===s.questions[qi].answer).length;
              const pct2=Math.round(correct/s.questions.length*100);
              return <div key={si} style={{...S.section,textAlign:"center"}}>
                <div style={{color:pct2>=80?P.accent3:pct2>=60?P.accent5:P.accent9,fontWeight:700,fontSize:20}}>{pct2}%</div>
                <div style={{color:P.muted,fontSize:11,marginTop:4}}>{s.title.replace("Section ","S")}</div>
              </div>;
            })}
          </div>
          <button style={S.btn(P.accent1)} onClick={()=>{setQuizDone(false);setQuizAns({});setQuizSec(0);setQuizIdx(0);setQuizScore(null);}}>
            Retake Quiz
          </button>
        </div>
      );
    }

    return (
      <div style={{maxWidth:700,margin:"0 auto"}}>
        <h2 style={{color:P.accent4,marginBottom:8}}>150 Question Quiz</h2>
        <div style={{display:"flex",gap:8,marginBottom:20,flexWrap:"wrap"}}>
          {QUIZ_SECTIONS.map((s,i)=>(
            <button key={i} style={{...S.btn(i===quizSec?P.accent4:P.border),
                                    color:i===quizSec?"#000":P.muted,fontSize:11,padding:"6px 12px"}}
                    onClick={()=>{setQuizSec(i);setQuizIdx(0);}}>
              S{i+1}
            </button>
          ))}
        </div>
        <div style={{color:P.muted,fontSize:12,marginBottom:4}}>{sec.title}</div>
        <div style={{display:"flex",justifyContent:"space-between",marginBottom:20}}>
          <div style={{color:P.muted,fontSize:12}}>Q {quizIdx+1} / {sec.questions.length}</div>
          <div style={{background:P.border,borderRadius:4,height:4,flex:1,margin:"6px 12px 0"}}>
            <div style={{background:P.accent4,height:4,borderRadius:4,
                         width:`${(quizIdx+1)/sec.questions.length*100}%`,transition:"width 0.3s"}}/>
          </div>
          <div style={{color:P.accent4,fontWeight:700,fontSize:12}}>
            {Object.values(quizAns).filter((v,i2)=>{ const k=Object.keys(quizAns)[i2]; const [si,qi]=k.split("-").map(Number); return quizAns[k]===QUIZ_SECTIONS[si].questions[qi].answer; }).length} correct
          </div>
        </div>
        <div style={{...S.section,marginBottom:20}}>
          <h3 style={{margin:"0 0 20px",lineHeight:1.5,fontSize:16}}>{q.q}</h3>
          <div style={{display:"flex",flexDirection:"column",gap:10}}>
            {q.options.map((opt,oi)=>{
              const selected = quizAns[key]===oi;
              const correct  = oi===q.answer;
              let bg=P.card2; let col=P.text; let border=P.border;
              if (answered) {
                if (correct) { bg=P.ok+"22"; col=P.ok; border=P.ok; }
                else if (selected) { bg=P.warn+"22"; col=P.warn; border=P.warn; }
              } else if (selected) { bg=P.accent4+"22"; col=P.accent4; border=P.accent4; }
              return (
                <div key={oi} style={{background:bg,border:`1px solid ${border}`,borderRadius:8,
                                      padding:"12px 16px",cursor:answered?"default":"pointer",
                                      color:col,transition:"all 0.15s",fontSize:13,lineHeight:1.4}}
                     onClick={()=>{ if (!answered) setQuizAns({...quizAns,[key]:oi}); }}>
                  <span style={{fontWeight:700,marginRight:10}}>{["A","B","C","D"][oi]}.</span>{opt}
                  {answered && correct && <span style={{float:"right",color:P.ok}}>✓</span>}
                  {answered && selected && !correct && <span style={{float:"right",color:P.warn}}>✗</span>}
                </div>
              );
            })}
          </div>
          {answered && <div style={{marginTop:12,color:P.muted,fontSize:12,padding:"10px",background:P.surface,borderRadius:6}}>
            {quizAns[key]===q.answer ? "✓ Correct!" : `✗ Correct answer: ${["A","B","C","D"][q.answer]}. ${q.options[q.answer]}`}
          </div>}
        </div>
        <div style={{display:"flex",justifyContent:"space-between",gap:12}}>
          <button style={{...S.btn(P.border),color:P.muted}} disabled={quizIdx===0 && quizSec===0}
                  onClick={()=>{ if (quizIdx>0) setQuizIdx(quizIdx-1); else if (quizSec>0){setQuizSec(quizSec-1);setQuizIdx(QUIZ_SECTIONS[quizSec-1].questions.length-1);} }}>
            ← Prev
          </button>
          {(quizIdx===sec.questions.length-1 && quizSec===QUIZ_SECTIONS.length-1) ? (
            <button style={S.btn(P.accent3)} onClick={submitQuiz}>Submit Quiz →</button>
          ) : (
            <button style={{...S.btn(P.accent4)}}
                    onClick={()=>{ if (quizIdx<sec.questions.length-1) setQuizIdx(quizIdx+1); else if (quizSec<QUIZ_SECTIONS.length-1){setQuizSec(quizSec+1);setQuizIdx(0);} }}>
              Next →
            </button>
          )}
        </div>
      </div>
    );
  };

  return (
    <div style={S.app}>
      <nav style={S.nav}>
        <div style={S.logo}>CV Tutorial</div>
        {TABS.map(t=>(
          <button key={t.id} style={S.tab(tab===t.id)} onClick={()=>{setTab(t.id);setSelDomain(null);setSelMod(null);setSelCh(null);setSearchQ("");}}>
            {t.label}
          </button>
        ))}
      </nav>
      <div style={S.content}>
        {tab==="domains"    && <DomainsView />}
        {tab==="modules"    && <ModulesView />}
        {tab==="challenges" && <ChallengesView />}
        {tab==="quiz"       && <QuizView />}
      </div>
    </div>
  );
}
