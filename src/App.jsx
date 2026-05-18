import { useState, useRef, useEffect } from "react";

const P = {
  // Core background system
  bg:        "#0f1115",
  surface:   "#171b21",
  card:      "#1f252c",
  card2:     "#2a3138",

  // Brand colours
  amber:     "#D59C10",
  amberSoft: "#E5B93C",
  amberDeep: "#B88208",

  graphite:  "#33383D",
  graphite2: "#454C52",
  graphite3: "#5A636B",

  // Accent system
  accent1:   "#D59C10",
  accent2:   "#E5B93C",
  accent3:   "#B88208",
  accent4:   "#6C7A89",
  accent5:   "#8FA3B0",

  // Text system
  text:      "#F5F7FA",
  textSoft:  "#D6D9DD",
  muted:     "#9AA4AE",

  // Borders and states
  border:    "#3A424B",
  borderSoft:"#2B3138",

  ok:        "#C8A227",
  warn:      "#E0A100",
  error:     "#C96B5C",
};

const CL = [
  P.accent1,
  P.accent2,
  P.accent3,
  P.accent4,
  P.accent5
];

const C = i => CL[i % CL.length];

/* ================================================================
   30 CV DOMAINS
================================================================ */
const DOMAINS = [
{
  id:1,
  name:"Image Classification",
  color:P.accent1,
  tagline:"Teach a computer how to recognize and name what it sees in an image",

  theory:`
Image classification is one of the most fundamental tasks in computer vision. The goal is simple: given one image, the computer assigns one main label to the entire image.

For example:
- a dog image should be classified as dog
- a car image should be classified as car
- a diseased tomato leaf image may be classified as tomato late blight
- a chest X-ray image may be classified as pneumonia or normal

A human can look at a picture and understand it quickly. A computer cannot do this naturally. To a computer, an image is only a grid of numbers.

A pixel is the smallest visible unit of an image. Every digital image is made from pixels.

For a grayscale image:
- 0 means black
- 255 means white
- values between 0 and 255 represent different shades of gray

For an RGB image, each pixel has three values:
- red
- green
- blue

Example:
[255, 0, 0] means red
[0, 255, 0] means green
[0, 0, 255] means blue
[255, 255, 255] means white
[0, 0, 0] means black

A grayscale image has shape:
height × width

An RGB image has shape:
height × width × 3

For example, an image with shape (224, 224, 3) means:
- 224 pixels tall
- 224 pixels wide
- 3 color channels

Image classification teaches a model to find useful patterns inside these numbers.

In older computer vision, engineers manually designed features such as edges, corners, gradients, and textures. Popular methods included SIFT, SURF, HOG, and traditional classifiers such as SVMs or k-NN.

Modern deep learning models learn these features automatically.

A convolutional neural network usually learns in stages:
- early layers learn edges and simple patterns
- middle layers learn textures and shapes
- deeper layers learn object parts
- final layers decide the class label

Image classification is used in healthcare, agriculture, manufacturing, robotics, wildlife monitoring, satellite imagery, quality inspection, smart cities, retail analytics, and autonomous systems.

This tutorial starts from pixels and slicing, then moves gradually toward CNNs, transfer learning, ResNet, EfficientNet, Vision Transformers, evaluation metrics, datasets, and real-world classification projects.
`,

  fundamentals:[
    "An image is a matrix of numbers. A grayscale image is a 2-D matrix, while an RGB image is a 3-D matrix with red, green, and blue channels.",
    "Image slicing means cutting out part of an image using row and column ranges. Example: img[20:80, 50:120] selects rows 20 to 79 and columns 50 to 119.",
    "A classifier looks at the entire image and predicts one main label. This is different from object detection, which predicts labels and bounding boxes.",
    "CNNs learn useful visual features automatically. Early layers detect edges, middle layers detect textures, and deeper layers detect object parts.",
    "Transfer learning allows us to use a model already trained on a huge dataset, then adapt it to a smaller custom dataset."
  ],

  architectures:[
    "LeNet-5",
    "AlexNet",
    "VGG-16",
    "VGG-19",
    "GoogLeNet",
    "Inception-v3",
    "ResNet-50",
    "ResNet-101",
    "DenseNet",
    "EfficientNet",
    "MobileNet",
    "ConvNeXt",
    "Vision Transformer",
    "Swin Transformer"
  ],

  metrics:[
    "Top-1 Accuracy",
    "Top-5 Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "Confusion Matrix",
    "ROC Curve",
    "AUC Score"
  ],

  datasets:[
    "MNIST, 70,000 handwritten digit images, 10 classes, 28 by 28 grayscale",
    "FashionMNIST, clothing image classification, 10 classes, 28 by 28 grayscale",
    "CIFAR-10, tiny color images, 10 classes, 32 by 32 RGB",
    "CIFAR-100, tiny color images, 100 classes, 32 by 32 RGB",
    "ImageNet, large-scale classification dataset, 1,000 classes",
    "Oxford 102 Flowers, flower species classification",
    "Stanford Cars, fine-grained car classification",
    "PlantVillage, plant disease classification",
    "Chest X-ray Pneumonia, medical image classification"
  ],

  colab:`
# ============================================================
# IMAGE CLASSIFICATION, FROM PIXELS TO RESNET
# Beginner friendly, step-by-step
# ============================================================

!pip install torch torchvision pillow matplotlib numpy

import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
import io

# ------------------------------------------------------------
# STEP 1: Download and load an image
# ------------------------------------------------------------

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"

response = urllib.request.urlopen(url)

img = Image.open(
    io.BytesIO(response.read())
).convert("RGB")

plt.figure(figsize=(6,6))
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")
plt.show()

# ------------------------------------------------------------
# STEP 2: Convert image to NumPy array
# ------------------------------------------------------------

img_np = np.array(img)

print("Image shape:", img_np.shape)

# Shape means:
# height, width, color channels

# ------------------------------------------------------------
# STEP 3: Inspect one pixel
# ------------------------------------------------------------

pixel = img_np[0, 0]

print("Top-left pixel:", pixel)

# This pixel contains:
# red value
# green value
# blue value

# ------------------------------------------------------------
# STEP 4: Slice a region from the image
# ------------------------------------------------------------

crop = img_np[50:200, 80:250]

plt.figure(figsize=(5,5))
plt.imshow(crop)
plt.title("Cropped Region")
plt.axis("off")
plt.show()

# img_np[50:200, 80:250] means:
# rows 50 to 199
# columns 80 to 249

# ------------------------------------------------------------
# STEP 5: Separate RGB channels
# ------------------------------------------------------------

red_channel = img_np[:, :, 0]
green_channel = img_np[:, :, 1]
blue_channel = img_np[:, :, 2]

fig, axes = plt.subplots(1, 3, figsize=(15,5))

axes[0].imshow(red_channel, cmap="Reds")
axes[0].set_title("Red Channel")
axes[0].axis("off")

axes[1].imshow(green_channel, cmap="Greens")
axes[1].set_title("Green Channel")
axes[1].axis("off")

axes[2].imshow(blue_channel, cmap="Blues")
axes[2].set_title("Blue Channel")
axes[2].axis("off")

plt.show()

# ------------------------------------------------------------
# STEP 6: Convert to grayscale manually
# ------------------------------------------------------------

gray = (
    0.299 * red_channel +
    0.587 * green_channel +
    0.114 * blue_channel
)

plt.figure(figsize=(6,6))
plt.imshow(gray, cmap="gray")
plt.title("Manual Grayscale Image")
plt.axis("off")
plt.show()

# ------------------------------------------------------------
# STEP 7: Load pretrained ResNet-50
# ------------------------------------------------------------

model = models.resnet50(
    weights=models.ResNet50_Weights.IMAGENET1K_V2
)

model.eval()

# ------------------------------------------------------------
# STEP 8: Preprocess image for ResNet
# ------------------------------------------------------------

transform_pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

x = transform_pipeline(img)

print("Tensor shape before batch:", x.shape)

x = x.unsqueeze(0)

print("Tensor shape after batch:", x.shape)

# ------------------------------------------------------------
# STEP 9: Run prediction
# ------------------------------------------------------------

with torch.no_grad():
    logits = model(x)
    probabilities = torch.softmax(logits, dim=1)

# ------------------------------------------------------------
# STEP 10: Get top 5 predictions
# ------------------------------------------------------------

top5 = probabilities.topk(5)

labels = urllib.request.urlopen(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
).read().decode().splitlines()

print("\\nTop 5 Predictions:\\n")

for probability, index in zip(top5.values[0], top5.indices[0]):
    label = labels[index]
    confidence = probability.item() * 100
    print(f"{label:30s} : {confidence:.2f}%")

# ------------------------------------------------------------
# Explanation
# ------------------------------------------------------------

# logits are raw model scores
# softmax converts logits into probabilities
# topk selects the highest probability classes
`
},

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
   10 LEARNING MODULES  (beginner to advanced)
================================================================ */
const MODULES = [
  {
    id:"M0", title:"What Is Computer Vision?", level:"Absolute Beginner", time:"25 min",
    color:P.accent1,
    sections:[
      { heading:"Definition and Scope",
        body:`Computer vision (CV) is the field of artificial intelligence that enables machines to interpret and understand visual information from the world: images, video, and 3D data. It bridges the gap between raw pixel values and high-level semantic understanding.

A camera captures light as a 2D array of numbers called pixels. Each pixel in a colour image stores three numbers: Red, Green, and Blue intensities, each ranging from 0 (dark) to 255 (bright). A 640x480 colour image therefore contains 640 x 480 x 3 = 921,600 numbers. CV algorithms process these numbers to extract meaning: "this image contains a dog," "the dog is running," "the dog is 3 metres from the camera."

Human vision is effortless and immediate. CV remains challenging because: (1) a 3D world projects onto a 2D image, losing depth; (2) the same object looks completely different under different lighting, viewpoints, and occlusion; (3) pixels encode appearance, not semantics. Teaching machines to bridge this semantic gap is the central challenge of CV.` },
      { heading:"A Brief History",
        body:`1960s: Roberts (1963) detected edges and corners in block-world images. 1970s: Marr's computational framework defined three levels: computational (what), algorithmic (how), implementational (hardware). 1980s: Canny edge detector (1986), scale-space theory, and early CNN prototypes. 1990s: Statistical learning methods, SVM-based classifiers, Viola-Jones face detector (real-time, 2001). 2012: AlexNet wins ImageNet by a 10.9% margin, using GPU-trained deep CNNs. This moment marks the modern deep learning era of CV. 2014: VGG, GoogLeNet, R-CNN for detection. 2015: ResNet (152 layers, 3.57% top-5 error), U-Net for medical segmentation. 2017: Attention is All You Need (transformer). 2020: ViT applies transformers to images; NeRF for 3D. 2021: CLIP, DALL-E, Segment Anything, diffusion models. 2024: Depth Anything, SAM2, 3D Gaussian Splatting go mainstream.` },
      { heading:"Core CV Tasks at a Glance",
        body:`Classification: "What is in this image?" Single label per image. Output: class probabilities. Detection: "Where and what?" Bounding boxes + class labels per object. Segmentation (semantic): "What class does every pixel belong to?" Segmentation (instance): "Which pixels belong to which specific object?" Pose Estimation: "Where are the body joints?" Depth Estimation: "How far is each pixel?" Optical Flow: "How did each pixel move between frames?" Generation: "Create a new image." The 30 domains in this tutorial cover all these and more.` },
      { heading:"Setting Up Your Environment",
        body:`For this tutorial, use Google Colab (free GPU). Every code block is a complete, runnable Colab cell. Key libraries you will encounter:

numpy: n-dimensional arrays, the language of images in Python.
opencv-python (cv2): classical CV algorithms, fast image I/O.
Pillow (PIL): convenient image loading and format conversion.
matplotlib: visualization of images and plots.
torch (PyTorch): deep learning framework, industry standard for research.
torchvision: pretrained models and image transforms for PyTorch.
transformers (HuggingFace): state-of-the-art pretrained vision-language models.
scikit-image (skimage): additional image processing utilities and metrics.

Install anything missing in Colab with: !pip install package_name` },
    ],
    code:`# Your first image in Python: load, inspect, manipulate
import numpy as np
import urllib.request
import io
from PIL import Image
import matplotlib.pyplot as plt

# Download an image from the web
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
img_bytes = urllib.request.urlopen(url).read()
img_pil   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
img       = np.array(img_pil)   # Convert to numpy array

print(f"Shape:  {img.shape}")   # (height, width, channels)
print(f"dtype:  {img.dtype}")   # uint8
print(f"Range:  {img.min()} to {img.max()}")
print(f"Pixels: {img.shape[0]*img.shape[1]:,}")

# Access individual pixels and channels
top_left_pixel = img[0, 0, :]          # RGB of top-left pixel
red_channel    = img[:, :, 0]          # All red values
green_channel  = img[:, :, 1]
blue_channel   = img[:, :, 2]

# Basic manipulations
flipped_h   = img[:, ::-1, :]          # Horizontal flip
flipped_v   = img[::-1, :, :]          # Vertical flip
grayscale   = (0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]).astype(np.uint8)
brightened  = np.clip(img.astype(int) + 50, 0, 255).astype(np.uint8)
darkened    = (img * 0.4).astype(np.uint8)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for ax, im, t in zip(axes.flat,
    [img, red_channel, green_channel, blue_channel,
     grayscale, flipped_h, brightened, darkened],
    ["RGB Original","Red Channel","Green Channel","Blue Channel",
     "Grayscale (weighted)","Flipped Horizontal","Brightened (+50)","Darkened (x0.4)"]):
    if im.ndim == 2:
        ax.imshow(im, cmap="gray")
    else:
        ax.imshow(im)
    ax.set_title(t, fontsize=10); ax.axis("off")
plt.suptitle("Module 0: Basic Image Manipulation", fontsize=14, y=1.01)
plt.tight_layout(); plt.savefig("m0_basics.png", dpi=150); plt.show()`,
  },
  {
    id:"M1", title:"Image Preprocessing and Classical Features", level:"Beginner", time:"40 min",
    color:P.accent2,
    sections:[
      { heading:"Why Preprocessing Matters",
        body:`Raw images from real cameras contain variations irrelevant to the task: different lighting conditions, sensor noise, varying resolutions, and inconsistent colour balance. Preprocessing standardises these variations so the model focuses on semantically relevant patterns rather than imaging artifacts.

Standard preprocessing for deep learning: (1) Resize to a fixed resolution (e.g. 224x224 for ImageNet models). (2) Normalise pixel values: divide by 255 to get [0,1], then subtract the dataset mean and divide by standard deviation. This ensures gradients flow well during training. (3) Data augmentation: random crops, horizontal flips, colour jitter, rotation, and Mixup/CutMix create artificial diversity, regularising the model and reducing overfitting.

ImageNet normalisation (used for any pretrained model): mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] (computed over the ImageNet training set in RGB order).` },
      { heading:"Edge Detection: From Pixels to Structure",
        body:`Edges are locations where pixel intensity changes rapidly. They correspond to object boundaries, surface discontinuities, and shadows. The Sobel operator computes image gradients using two 3x3 convolution kernels: Gx detects horizontal edges, Gy detects vertical edges. The gradient magnitude sqrt(Gx^2 + Gy^2) gives edge strength. The Canny edge detector (1986) is the gold standard: (1) Gaussian blur to reduce noise; (2) Sobel gradient computation; (3) Non-maximum suppression to thin edges to single-pixel width; (4) Double threshold (low, high) for hysteresis to link strong and weak edges. Canny produces clean, thin edges while suppressing noise.` },
      { heading:"Colour Spaces",
        body:`RGB (Red, Green, Blue): additive colour model, natural for cameras and displays. Each channel [0,255].
HSV (Hue, Saturation, Value): perceptually intuitive. Hue is the colour type [0,179 in OpenCV], Saturation is colour purity [0,255], Value is brightness [0,255]. Excellent for colour-based object tracking (e.g. detect red objects regardless of lighting).
LAB (L*a*b*): L is lightness (perceptually uniform), a is green-red axis, b is blue-yellow axis. The Euclidean distance in LAB space approximates human-perceived colour difference. Used in CLAHE, colour correction, and image quality assessment.
YCbCr: Y is luminance, Cb and Cr are chrominance components. Used in JPEG compression and video codecs (the human eye is less sensitive to colour resolution than luminance).
Grayscale: single-channel intensity. Reduces data by 3x while preserving structural information for many tasks.` },
      { heading:"Classical Feature Descriptors",
        body:`Before deep learning, hand-crafted features were state of the art. Understanding them deepens intuition for what CNNs learn automatically.

HOG (Histogram of Oriented Gradients, Dalal and Triggs 2005): divides image into cells; computes gradient direction histogram per cell; normalises over blocks of cells. Very effective for pedestrian detection. Still used in some real-time systems.

SIFT (Scale-Invariant Feature Transform, Lowe 2004): detects keypoints at scale-space extrema; builds 128-dimensional descriptor from gradient histograms around the keypoint. Invariant to scale, rotation, and partially to illumination. Used for image matching, panorama stitching, 3D reconstruction.

ORB (Oriented FAST and Rotated BRIEF): fast alternative to SIFT, patent-free. FAST keypoint detector + BRIEF binary descriptor + orientation compensation. Runs in real-time on CPU.` },
    ],
    code:`# Complete preprocessing and feature extraction pipeline
import cv2, numpy as np, urllib.request
import matplotlib.pyplot as plt

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw = urllib.request.urlopen(url).read()
img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ── Edge Detection ──────────────────────────────────────
sobel_x  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
gradient = np.sqrt(sobel_x**2 + sobel_y**2)
gradient = (gradient / gradient.max() * 255).astype(np.uint8)
canny    = cv2.Canny(gray, 50, 150)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# ── Colour Spaces ───────────────────────────────────────
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# ── SIFT Feature Detection ──────────────────────────────
sift     = cv2.SIFT_create(nfeatures=300)
kp, desc = sift.detectAndCompute(gray, None)
sift_vis = cv2.drawKeypoints(rgb.copy(), kp, None,
               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print(f"SIFT: {len(kp)} keypoints, descriptor shape: {desc.shape}")

# ── HOG Features ────────────────────────────────────────
win_size   = (128, 64)
roi        = cv2.resize(gray, win_size[::-1])
hog        = cv2.HOGDescriptor()
hog_feat   = hog.compute(roi)
print(f"HOG feature vector length: {len(hog_feat)}")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
data = [rgb, gray, gradient, canny,
        hsv[:,:,0], lab[:,:,0], sift_vis, np.abs(laplacian).astype(np.uint8)]
titles = ["Original RGB","Grayscale","Sobel Gradient","Canny Edges",
          "HSV Hue channel","LAB Lightness (L)","SIFT Keypoints (300)","Laplacian"]
cmaps  = [None,"gray","gray","gray","hsv","gray",None,"gray"]
for ax,im,t,cm in zip(axes.flat,data,titles,cmaps):
    ax.imshow(im,cmap=cm); ax.set_title(t,fontsize=10); ax.axis("off")
plt.suptitle("Module 1: Preprocessing and Classical Features",fontsize=14,y=1.01)
plt.tight_layout(); plt.savefig("m1_preprocess.png",dpi=150); plt.show()`,
  },
  {
    id:"M2", title:"Convolutional Neural Networks from First Principles", level:"Beginner-Intermediate", time:"60 min",
    color:P.accent3,
    sections:[
      { heading:"The Convolution Operation",
        body:`A convolutional layer applies a learned filter (kernel) by sliding it across the input, computing a dot product at each position. A 3x3 kernel on a single-channel image produces one value per position: the sum of element-wise products between the kernel and the local 3x3 patch.

Key parameters: kernel size (3x3, 5x5, 7x7) controls receptive field; stride controls how many pixels to skip between applications (stride=2 halves spatial resolution); padding (same or valid) controls output size. For a single output neuron: z = sum over k,l of (W[k,l] * I[i+k,j+l]) + b. With multiple filters, each produces one feature map; F filters produce F feature maps, becoming the channels of the next layer.

Why convolutions work: (1) local connectivity: each output depends only on a local region, matching the locality of visual features; (2) weight sharing: the same filter slides everywhere, so the feature detector (edge, curve, texture) is learned once and applied everywhere; (3) translational equivariance: if the input feature shifts, its activation map shifts by the same amount.` },
      { heading:"Building Blocks: Activation, Pooling, Normalisation",
        body:`After convolution, an activation function introduces non-linearity. ReLU (max(0,x)) is universal: computationally cheap, avoids vanishing gradients, and empirically works well. LeakyReLU, PReLU, and GELU are variants used in modern architectures.

Pooling reduces spatial resolution: max pooling takes the maximum in a local window, creating modest translation invariance and reducing computation. Global Average Pooling (GAP) averages each feature map to a single number, replacing fully-connected layers in modern architectures (used in ResNet, EfficientNet).

Batch Normalisation (Ioffe and Szegedy, 2015) normalises each mini-batch per channel: for each channel, subtract batch mean and divide by batch std, then apply learnable scale and shift parameters. Benefits: stabilises training, allows higher learning rates, acts as regulariser, and greatly reduces sensitivity to weight initialisation. Applied after conv, before activation in most architectures.` },
      { heading:"ResNet: Residual Learning",
        body:`Very deep networks (>20 layers) suffer from the degradation problem: training accuracy degrades with depth, even without overfitting. He et al. (2015) showed this is not caused by vanishing gradients alone but by optimisation difficulty.

Solution: skip connections (shortcuts) that bypass one or more layers: H(x) = F(x) + x, where F(x) is what the residual block learns. The network now optimises F(x) = H(x) - x, learning the residual relative to the identity. If a layer should be an identity, F can be driven to zero without disrupting the signal path.

This allows training networks with 50, 101, 152, even 1000+ layers. ResNet-50 achieves 76.1% top-1 on ImageNet; ResNet-152 achieves 77.8%. The residual connection also provides gradient highways: gradients can flow directly through the shortcut, significantly mitigating vanishing gradients in deep networks.` },
      { heading:"Transfer Learning",
        body:`Training a deep CNN from scratch requires millions of labelled examples and many GPU-days. Transfer learning reuses weights pretrained on a large dataset (usually ImageNet) for a new task. The intuition: lower layers learn universal features (edges, textures, colours) applicable to any image task; higher layers learn task-specific features that can be replaced or fine-tuned.

Two strategies: (1) Feature extraction: freeze all pretrained layers, replace only the final classification head with a new one for your task, and train only the new head. Fast, requires very little data. (2) Fine-tuning: initialise with pretrained weights, then train all layers (or the last few) on your new dataset with a small learning rate (typically 10-100x smaller than from-scratch). More powerful, requires more data but far less than training from scratch.

Rule of thumb: if your dataset is small and similar to ImageNet, use feature extraction. If large and similar, fine-tune all layers. If your domain is very different (e.g. medical, satellite), fine-tune from the first domain-relevant layer.` },
    ],
    code:`# Build a mini-CNN from scratch + ResNet feature extraction demo
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import urllib.request, io, numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ── 1. Conv2D from scratch (to understand what it does) ──────
def conv2d_manual(img, kernel):
    """Manual convolution (no PyTorch) for understanding."""
    kh, kw   = kernel.shape
    h, w     = img.shape
    out_h, out_w = h - kh + 1, w - kw + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            out[i,j] = (img[i:i+kh, j:j+kw] * kernel).sum()
    return out

import cv2, urllib.request
url  = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw  = urllib.request.urlopen(url).read()
img  = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_GRAYSCALE).astype(float)/255.0
img  = cv2.resize(img,(160,120))

K_edge    = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=float)
K_blur    = np.ones((5,5),float)/25.0
K_sharpen = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],float)

edge_map  = conv2d_manual(img, K_edge)
blur_map  = conv2d_manual(img, K_blur)
sharp_map = conv2d_manual(img, K_sharpen)

fig, axes = plt.subplots(1,4,figsize=(20,5))
for ax,im,t in zip(axes,[img,edge_map,blur_map,sharp_map],
    ["Original","Edge kernel","Blur kernel","Sharpen kernel"]):
    ax.imshow(np.abs(im),cmap="gray"); ax.set_title(t); ax.axis("off")
plt.suptitle("Manual Conv2D: same operation as a CNN layer",fontsize=13)
plt.tight_layout(); plt.savefig("m2_conv.png",dpi=150); plt.show()

# ── 2. ResNet-50 Transfer Learning (feature extraction) ──────
print("\\nResNet-50 Transfer Learning demo:")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
feature_extractor = nn.Sequential(*list(model.children())[:-1])  # Remove final FC
feature_extractor.eval()

tf  = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),
        transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
img_pil = Image.open(io.BytesIO(urllib.request.urlopen(url2).read())).convert("RGB")
x       = tf(img_pil).unsqueeze(0)

with torch.no_grad():
    features = feature_extractor(x).squeeze()
print(f"Feature vector shape: {features.shape}")  # 2048-dim
print(f"Feature norm: {features.norm().item():.3f}")
print("These 2048 numbers encode rich semantic content of the image.")
print("Attach a small classifier head to these for any new task!")`,
  },
  {
    id:"M3", title:"Object Detection: From Anchors to Transformers", level:"Intermediate", time:"55 min",
    color:P.accent4,
    sections:[
      { heading:"The Detection Problem and Anchor Boxes",
        body:`Object detection must predict a variable number of bounding boxes per image, each with a class label and confidence score. A bounding box is typically parameterised as (cx, cy, w, h): centre x, centre y, width, height, all relative to the image.

Anchor boxes are a key design choice in most pre-DETR detectors. A set of predefined boxes with various aspect ratios and sizes are tiled over the image at each spatial location of the feature map. The network predicts offsets from each anchor (dx, dy, dw, dh) and a class probability. Anchors act as "priors": the network learns small adjustments rather than predicting boxes from scratch.

Typical anchor design: 3 scales (small, medium, large) x 3 aspect ratios (0.5, 1, 2) = 9 anchors per spatial location. For a 13x13 feature map, this gives 13x13x9 = 1521 anchor candidates. Most are background; the network must learn to score them correctly.` },
      { heading:"IoU, NMS, and mAP Explained",
        body:`IoU (Intersection over Union): IoU(A,B) = area(A intersect B) / area(A union B). Range [0,1]. IoU=0: no overlap. IoU=1: perfect match. Used as the matching criterion between predictions and ground truth, and as a threshold for considering a detection correct (typically 0.5 for PASCAL VOC, 0.5:0.05:0.95 for COCO).

NMS (Non-Maximum Suppression): detection networks produce many overlapping boxes for the same object. NMS suppresses duplicates: (1) sort all boxes by confidence; (2) keep the highest-confidence box; (3) remove all boxes with IoU > threshold (e.g. 0.5) with the kept box; (4) repeat for remaining boxes. Soft-NMS decays confidence instead of hard removal, improving recall on crowded scenes.

mAP (mean Average Precision): compute precision-recall curve for each class; Average Precision (AP) is the area under that curve. mAP averages AP over all classes. COCO mAP averages over IoU thresholds from 0.5 to 0.95 in steps of 0.05, which is much more demanding than the original VOC mAP@0.5.` },
      { heading:"YOLO Architecture Deep Dive",
        body:`YOLO (You Only Look Once) divides the image into an SxS grid. Each grid cell predicts B bounding boxes and C class probabilities in a single forward pass through a single CNN. This makes it extremely fast (real-time on GPU).

YOLOv1 (Redmon et al. 2016): SxS=7, B=2, C=20 (PASCAL VOC). Single stage but poor on small objects and many overlapping objects. YOLOv3 introduced multi-scale prediction at 3 feature map scales using feature pyramid-style skip connections. YOLOv5/v8 (Ultralytics): C++-optimised backbone (CSPDarknet), PANet feature pyramid neck, decoupled head for classification and regression. Anchor-free YOLOv8 predicts boxes directly as centre+size relative to grid cell without pre-defined anchors, simplifying training. Real-time at 640px input on GPU.` },
      { heading:"DETR: Detection with Transformers",
        body:`DETR (Carion et al., 2020) replaced anchors, NMS, and region proposal networks with a clean end-to-end transformer. Architecture: ResNet backbone extracts features; flattened features + positional encoding feed a transformer encoder (self-attention over all spatial positions); N learnable object queries feed a transformer decoder (attending to encoder output); two prediction heads per query predict a class and a bounding box.

Training uses bipartite (one-to-one) matching via the Hungarian algorithm to find the optimal assignment between N predictions and M ground-truth objects (padded with "no object"). This eliminates NMS completely: each object is matched to exactly one query. DETR converges slowly (500 epochs vs 12 for Faster R-CNN) but Deformable DETR and DN-DETR improved this. RT-DETR achieves real-time DETR performance competitive with YOLOv8.` },
    ],
    code:`# Object Detection: IoU from scratch + NMS + YOLOv8 full demo
import numpy as np, matplotlib.pyplot as plt, matplotlib.patches as patches

# ── IoU from scratch ──────────────────────────────────────────
def iou(box1, box2):
    """box format: [x1,y1,x2,y2]"""
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter + 1e-8)

# ── NMS from scratch ──────────────────────────────────────────
def nms(boxes, scores, threshold=0.5):
    """boxes: [[x1,y1,x2,y2],...], scores: [conf,...]"""
    order  = np.argsort(scores)[::-1]
    keep   = []
    while len(order) > 0:
        i = order[0]; keep.append(i)
        if len(order) == 1: break
        ious = np.array([iou(boxes[i], boxes[j]) for j in order[1:]])
        order = order[1:][ious < threshold]
    return keep

# Simulate 8 overlapping detection boxes
np.random.seed(42)
boxes_raw = np.array([[60,60,200,200],[70,65,210,205],[65,62,195,198],
                       [62,64,202,200],[200,50,350,180],[205,55,355,185],
                       [201,52,348,178],[400,100,500,220]])
scores_raw= np.array([0.95,0.80,0.72,0.65, 0.88,0.75,0.60, 0.91])

kept = nms(boxes_raw, scores_raw, threshold=0.5)
print(f"Before NMS: {len(boxes_raw)} boxes")
print(f"After  NMS: {len(kept)} boxes kept: indices {kept}")

fig, (a1,a2) = plt.subplots(1,2,figsize=(14,6))
for ax,title in zip([a1,a2],["Before NMS (all boxes)","After NMS (kept boxes)"]):
    ax.set_xlim(0,550); ax.set_ylim(0,300); ax.invert_yaxis()
    ax.set_facecolor("#111"); ax.set_title(title,color="white")
for i,(b,s) in enumerate(zip(boxes_raw,scores_raw)):
    r = patches.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],linewidth=1.5,
          edgecolor="gray" if i not in kept else "lime",facecolor="none")
    a1.add_patch(r); a1.text(b[0],b[1]-2,f"{s:.2f}",color="gray",fontsize=8)
for i in kept:
    b=boxes_raw[i]; s=scores_raw[i]
    r=patches.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],linewidth=2,edgecolor="lime",facecolor="none")
    a2.add_patch(r); a2.text(b[0],b[1]-2,f"{s:.2f}",color="lime",fontsize=9,fontweight="bold")
plt.tight_layout(); plt.savefig("m3_nms.png",dpi=150); plt.show()

# IoU matrix visualisation
print("\\nIoU between all pairs of kept boxes:")
for i in kept:
    for j in kept:
        print(f"  box[{i}] vs box[{j}]: IoU={iou(boxes_raw[i],boxes_raw[j]):.3f}")`,
  },
  {
    id:"M4", title:"Segmentation: Pixel-Level Understanding", level:"Intermediate", time:"50 min",
    color:P.accent5,
    sections:[
      { heading:"Semantic vs Instance vs Panoptic Segmentation",
        body:`Semantic segmentation assigns one class label to every pixel. All dogs in the image share the label "dog"; the model does not distinguish between them. Output: a 2D label map of shape (H,W) with integer class indices. Evaluated with mIoU (mean Intersection over Union across all classes).

Instance segmentation assigns both a class label and an instance ID to every foreground pixel. If three dogs are in the image, each pixel is labelled as dog-1, dog-2, or dog-3. Background pixels have no instance ID. Output: a set of binary masks, one per detected object. Evaluated with Mask AP (averaged over IoU thresholds).

Panoptic segmentation (Kirillov et al., 2019) unifies both: every pixel gets exactly one label. For "things" (countable objects: person, car, dog), each instance gets a unique ID. For "stuff" (uncountable regions: sky, road, grass), only the semantic label is assigned. Evaluated with Panoptic Quality (PQ = SQ x RQ: segmentation quality times recognition quality).` },
      { heading:"U-Net Architecture: The Medical Imaging Standard",
        body:`U-Net (Ronneberger et al., 2015) was designed specifically for biomedical image segmentation where training data is scarce. Its defining feature is the contracting path (encoder) connected to an expansive path (decoder) with skip connections at each resolution level.

Encoder: 4 blocks of (Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool). At each block, spatial resolution halves, number of feature channels doubles (64, 128, 256, 512, 1024). Bottleneck: 1024 channels. Decoder: 4 blocks of (Upsample/TransposedConv -> Concatenate with corresponding encoder skip -> Conv -> BN -> ReLU -> Conv). Final: 1x1 conv to produce per-pixel class scores.

Why skip connections matter: the encoder's downsampling loses precise spatial information needed for pixel-accurate segmentation. Skip connections paste high-resolution encoder features directly to the decoder at the corresponding level, allowing the decoder to recover fine spatial detail. Without them, the output would be spatially imprecise.` },
      { heading:"DeepLab and Dilated Convolutions",
        body:`A key challenge in semantic segmentation: to classify each pixel, the model needs a large receptive field to understand context, but downsampling reduces spatial resolution, losing precise location. Dilated (atrous) convolutions solve this: they insert zeros between kernel elements, expanding the receptive field without reducing spatial resolution and without increasing parameters.

A 3x3 dilated convolution with dilation rate r has the same number of parameters as a standard 3x3 convolution but a receptive field of (2r+1)x(2r+1). DeepLab v3+ uses ASPP (Atrous Spatial Pyramid Pooling): multiple parallel dilated convolutions with different rates (6, 12, 18) plus a global average pooling branch. Their outputs are concatenated and fused, capturing features at multiple scales in a single forward pass. The decoder then combines these multi-scale features with high-resolution features from the encoder backbone.` },
      { heading:"Segment Anything Model (SAM)",
        body:`SAM (Kirillov et al., Meta 2023) is a foundation model for promptable image segmentation. It was trained on SA-1B: 11M diverse images with 1.1 billion high-quality masks, generated semi-automatically using SAM itself in a data engine loop.

Architecture: a ViT-H image encoder produces a 64x64 image embedding; a prompt encoder handles points, boxes, text, or masks; a lightweight mask decoder (two transformer layers + MLP heads) predicts up to three masks (whole, part, subpart) per prompt with ambiguity awareness.

Three interaction modes: (1) point prompt: click a point, get the object at that point; (2) box prompt: draw a bounding box, get the mask within it; (3) everything mode: segment all objects automatically. SAM2 (2024) extends to video with memory attention, enabling zero-shot video object segmentation at 44 FPS.` },
    ],
    code:`# Segmentation pipeline: classical watershed + U-Net architecture demo
import cv2, numpy as np, urllib.request, matplotlib.pyplot as plt
import torch, torch.nn as nn

# ── Classical: Watershed segmentation ────────────────────────
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
raw = urllib.request.urlopen(url).read()
img = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_COLOR)

gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
sure_bg = cv2.dilate(th,kernel,iterations=3)
dist    = cv2.distanceTransform(th,cv2.DIST_L2,5)
_,sure_fg=cv2.threshold(dist,0.5*dist.max(),255,0)
sure_fg = sure_fg.astype(np.uint8)
unknown = cv2.subtract(sure_bg,sure_fg)
_,markers=cv2.connectedComponents(sure_fg)
markers +=1; markers[unknown==255]=0
watershed = cv2.watershed(img,markers)
vis = img.copy(); vis[watershed==-1]=[0,255,0]

# ── U-Net mini architecture (untrained, for illustration) ──
class DoubleConv(nn.Module):
    def __init__(self,inc,outc):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(inc,outc,3,padding=1),nn.BatchNorm2d(outc),nn.ReLU(inplace=True),
            nn.Conv2d(outc,outc,3,padding=1),nn.BatchNorm2d(outc),nn.ReLU(inplace=True))
    def forward(self,x): return self.block(x)

class UNet(nn.Module):
    def __init__(self,in_c=3,out_c=2,base=64):
        super().__init__()
        self.enc1=DoubleConv(in_c,base);   self.pool1=nn.MaxPool2d(2)
        self.enc2=DoubleConv(base,base*2); self.pool2=nn.MaxPool2d(2)
        self.enc3=DoubleConv(base*2,base*4);self.pool3=nn.MaxPool2d(2)
        self.bottleneck=DoubleConv(base*4,base*8)
        self.up3=nn.ConvTranspose2d(base*8,base*4,2,stride=2)
        self.dec3=DoubleConv(base*8,base*4)
        self.up2=nn.ConvTranspose2d(base*4,base*2,2,stride=2)
        self.dec2=DoubleConv(base*4,base*2)
        self.up1=nn.ConvTranspose2d(base*2,base,2,stride=2)
        self.dec1=DoubleConv(base*2,base)
        self.head=nn.Conv2d(base,out_c,1)
    def forward(self,x):
        e1=self.enc1(x); e2=self.enc2(self.pool1(e1))
        e3=self.enc3(self.pool2(e2)); b=self.bottleneck(self.pool3(e3))
        d3=self.dec3(torch.cat([self.up3(b),e3],1))
        d2=self.dec2(torch.cat([self.up2(d3),e2],1))
        d1=self.dec1(torch.cat([self.up1(d2),e1],1))
        return self.head(d1)

unet=UNet()
total=sum(p.numel() for p in unet.parameters())
print(f"U-Net parameters: {total:,}")
x_test=torch.randn(1,3,256,256)
with torch.no_grad(): out=unet(x_test)
print(f"Input shape:  {x_test.shape}")
print(f"Output shape: {out.shape}  (per-pixel logits for 2 classes)")

fig,(a,b,c)=plt.subplots(1,3,figsize=(15,5))
a.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)); a.set_title("Original"); a.axis("off")
b.imshow(dist,cmap="hot"); b.set_title("Distance Transform"); b.axis("off")
c.imshow(cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)); c.set_title("Watershed (green=boundaries)"); c.axis("off")
plt.tight_layout(); plt.savefig("m4_seg.png",dpi=150); plt.show()`,
  },
  {
    id:"M5", title:"Image Restoration: Denoising to Diffusion", level:"Intermediate", time:"45 min",
    color:P.accent6,
    sections:[
      { heading:"Degradation Models and the Ill-Posed Inverse Problem",
        body:`Restoration assumes an observed degraded image y = D(x) + n, where x is the clean image, D is the degradation operator (blur, downsampling, rain synthesis, haze scattering), and n is additive noise. The goal is to invert D to recover x from y.

This is ill-posed: for any given y, infinitely many x could have produced it (a blurred version of many sharp images looks the same). A unique solution requires regularisation: additional assumptions about what a "good" clean image looks like. Classical regularisers: Total Variation (promotes piecewise-constant images), Tikhonov (promotes smoothness), BM3D (non-local self-similarity). Deep learning implicitly learns a powerful neural regulariser from training data.

Types of degradation: Gaussian blur (lens defocus, camera shake); Gaussian noise (sensor thermal noise, amplification noise); Poisson noise (photon shot noise in low light, medical imaging); JPEG compression artifacts (block discontinuities, ringing); rain streaks (synthetic or real rain); haze (atmospheric scattering); motion blur (fast object or camera motion).` },
      { heading:"PSNR and SSIM: Understanding Quality Metrics",
        body:`PSNR (Peak Signal-to-Noise Ratio): PSNR = 10 * log10(MAX^2 / MSE), where MAX=255 for 8-bit images and MSE is mean squared error. Higher is better. Typical values: 20 dB (poor), 30 dB (acceptable), 40 dB (excellent). Limitation: MSE treats all pixel errors equally; perceptually, a uniform shift is less noticeable than fine-grained texture error.

SSIM (Structural Similarity Index): compares images on three dimensions: luminance (mean intensity), contrast (standard deviation), and structure (normalised cross-correlation). SSIM in [-1,1], higher is better; SSIM=1 means identical. More correlated with human perception than PSNR for moderate distortions.

LPIPS (Learned Perceptual Image Patch Similarity): computes distance between deep CNN features (VGG or AlexNet) of the two images. Lower is better. Best correlation with human perceptual judgments. Used alongside PSNR/SSIM to evaluate generative restoration methods. No-reference metrics (NIQE, BRISQUE) estimate quality without a reference clean image, crucial when ground truth is unavailable.` },
      { heading:"DnCNN to Restormer: Architecture Evolution",
        body:`DnCNN (Zhang et al. 2017): 17-layer CNN trained to predict the noise residual (y - x). Subtracting the residual gives the denoised image. Batch normalisation at every layer accelerates convergence. Demonstrated blind denoising across noise levels with a single model.

FFDNet (Zhang et al. 2018): modified DnCNN accepting the noise level sigma as an additional input map, allowing flexible control of denoising strength at inference. Runs in real-time.

MPRNet (Zamir et al. 2021): multi-stage progressive restoration with supervised attention modules at each stage. Each stage produces an intermediate restored image, and supervised losses at each stage prevent gradient vanishing. State-of-the-art across deraining, deblurring, denoising simultaneously.

Restormer (Zamir et al. 2022): efficient transformer for high-resolution image restoration. Applies multi-head self-attention across channels (not spatial positions), since for high-resolution images spatial attention is prohibitively expensive (O(H^2*W^2)). Transposed attention across C channels is O(C^2) independent of resolution. State-of-the-art on most restoration tasks.` },
      { heading:"Diffusion Models for Image Restoration",
        body:`Diffusion models frame restoration as conditional image generation. Given a degraded image y, generate clean images x by running a diffusion reverse process conditioned on y. SR3 (Saharia et al.) showed this for super-resolution. DiffIR conditions the U-Net denoiser on IR priors extracted from the degraded image. StableSR uses Stable Diffusion with a controllable feature wrapping module for realistic high-resolution super-resolution.

Advantages: diverse, high-perceptual-quality outputs; can generate multiple plausible restorations for a given degraded input. Limitations: slow inference (many NFEs: number of function evaluations); PSNR/SSIM may be lower than regression-based methods because the model generates realistic textures that may differ from the specific ground-truth texture.

The determinism vs diversity trade-off: regression-based methods (MSE loss) produce the expectation over all plausible restorations, which is a blurry average. Diffusion-based methods sample one specific plausible restoration, which is sharp but may not match the ground truth exactly. Hence high LPIPS but sometimes lower PSNR.` },
    ],
    code:`# Complete restoration pipeline: noise, blur, JPEG, and multiple denoisers
import cv2, numpy as np, urllib.request, matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw = urllib.request.urlopen(url).read()
clean = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_COLOR)

# ── Generate multiple degradations ───────────────────────────
rng   = np.random.default_rng(123)
noise = rng.normal(0,30,clean.shape).astype(np.float32)
noisy = np.clip(clean.astype(float)+noise,0,255).astype(np.uint8)
blurred= cv2.GaussianBlur(clean,(15,15),3)
_,enc = cv2.imencode(".jpg",clean,[cv2.IMWRITE_JPEG_QUALITY,20])
jpeg  = cv2.imdecode(enc,cv2.IMREAD_COLOR)

# ── Multiple denoisers ────────────────────────────────────────
def nlmeans(im):
    return cv2.fastNlMeansDenoisingColored(im,None,h=10,hColor=10,
                                            templateWindowSize=7,searchWindowSize=21)
def guided(im, ref):
    out=[]
    for c in range(3):
        out.append(cv2.ximgproc.guidedFilter(ref[:,:,c],im[:,:,c],9,0.01))
    return np.stack(out,axis=2)
def bilateral(im):
    return cv2.bilateralFilter(im,d=9,sigmaColor=75,sigmaSpace=75)

restored_nlm = nlmeans(noisy)
restored_bil = bilateral(noisy)

# ── Compute metrics ───────────────────────────────────────────
def metrics(ref,img):
    p = psnr(ref,img,data_range=255)
    s = ssim(ref,img,channel_axis=2,data_range=255)
    return p,s

results=[("Noisy",noisy),("NLMeans",restored_nlm),("Bilateral",restored_bil),
         ("Blurred",blurred),("JPEG q=20",jpeg)]
print(f"{'Method':12s}  {'PSNR':6s}  {'SSIM':6s}")
for name,img in results:
    p,s=metrics(clean,img)
    print(f"{name:12s}  {p:.2f}   {s:.4f}")

fig,axes=plt.subplots(2,3,figsize=(18,12))
for ax,im,t in zip(axes.flat,[clean,noisy,restored_nlm,restored_bil,blurred,jpeg],
    ["Clean","Noisy (sigma=30)","NLMeans denoised","Bilateral denoised","Gaussian blur","JPEG q=20"]):
    ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)); ax.set_title(t); ax.axis("off")
plt.suptitle("Module 5: Image Restoration Comparison",fontsize=14)
plt.tight_layout(); plt.savefig("m5_restore.png",dpi=150); plt.show()`,
  },
  {
    id:"M6", title:"Vision Transformers: Attention over Images", level:"Advanced-Intermediate", time:"55 min",
    color:P.accent7,
    sections:[
      { heading:"Self-Attention Mechanism",
        body:`Self-attention computes relationships between all pairs of positions in a sequence. Given an input sequence of vectors, three linear projections produce Queries (Q), Keys (K), and Values (V). Attention weights: A = softmax(QK^T / sqrt(d_k)), where d_k is the key dimension (scaling prevents saturation). Output: O = AV. Each output position is a weighted sum of all value vectors, with weights determined by similarity (dot product) between its query and all keys.

Multi-head attention applies h attention heads in parallel with different projections, concatenates their outputs, and projects back: MultiHead(Q,K,V) = Concat(head_1, ..., head_h) * W_O. Different heads learn to attend to different types of relationships (local vs global, different semantic aspects). Computational complexity: O(N^2 * d) where N is sequence length. For images with N=H*W patches, this becomes expensive at high resolution. Efficient alternatives: shifted window attention (Swin), deformable attention (DAT), linear attention approximations.` },
      { heading:"Vision Transformer (ViT)",
        body:`ViT (Dosovitskiy et al., 2021) treats an image as a sequence of patches. Process: (1) split the HxW image into N non-overlapping patches of size P x P (e.g. 16x16), producing N = (H/P) * (W/P) patches; (2) linearly embed each flattened patch to dimension D; (3) prepend a learnable [CLS] token; (4) add learnable 1D positional embeddings; (5) pass through L transformer encoder layers; (6) use [CLS] token representation for classification.

Key finding: ViT outperforms CNNs on large-scale data (14M-300M images) but underperforms on small datasets where CNNs' inductive biases (locality, translation equivariance) provide crucial regularisation. DeiT demonstrated data-efficient training via distillation from a CNN teacher. Swin Transformer introduced hierarchical feature maps (like ResNet) and shifted window attention (local attention with cross-window connections), enabling ViT for dense prediction tasks (detection, segmentation) at multiple scales.` },
      { heading:"CLIP: Learning from Image-Text Pairs",
        body:`CLIP (Contrastive Language-Image Pretraining, Radford et al., 2021) jointly trains an image encoder (ViT or ResNet) and a text encoder (GPT-like transformer) on 400M image-text pairs collected from the internet. Training objective: contrastive loss that maximises cosine similarity between matching image-text pairs and minimises it for non-matching pairs within each batch (InfoNCE loss).

CLIP learns a shared embedding space where semantically related images and text are nearby. Zero-shot classification: encode the image; encode text prompts like "a photo of a dog" for each class; find the class with highest image-text similarity. CLIP achieves 76.2% zero-shot top-1 on ImageNet without any ImageNet training. It generalises remarkably across domains because the internet-scale training data covers enormous visual diversity.

CLIP is the backbone of modern image generation (Stable Diffusion text conditioning), retrieval, VQA, and anomaly detection systems.` },
      { heading:"DINOv2: Self-Supervised Vision Foundation Models",
        body:`DINOv2 (Oquab et al., 2023) trains a ViT using self-supervised learning on a curated dataset of 142M images (LVD-142M). Training objective combines: DINO (self-distillation with no labels: student matches teacher patch-level features), iBOT (masked image modelling: predict masked patch features), and KoLeo (spread-out regulariser to prevent feature collapse).

DINOv2 features generalise remarkably without fine-tuning: nearest-neighbour retrieval on ImageNet with DINOv2 ViT-L features matches supervised models. Dense prediction (segmentation, depth) via linear probing achieves competitive performance. The features exhibit impressive properties: semantic grouping (patches of the same object get similar features), scene understanding (foreground vs background separation), and correspondence (matching patches between images of different instances of the same class).` },
    ],
    code:`# Vision Transformer attention visualisation + CLIP zero-shot classification
import torch, urllib.request, io, numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ── CLIP zero-shot image classification ───────────────────────
# !pip install transformers
from transformers import CLIPProcessor, CLIPModel

model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
img = Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB")

# Zero-shot: define class names as text prompts
class_names = ["a dog","a cat","a bird","a car","a bicycle","a person","a horse","a boat"]
text_prompts = [f"a photo of {c}" for c in class_names]

inputs = processor(text=text_prompts, images=img, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1).squeeze().numpy()

print("CLIP Zero-Shot Classification Results:")
for name, p in sorted(zip(class_names,probs),key=lambda x:-x[1]):
    bar = "=" * int(p*50)
    print(f"  {name:15s} {p*100:5.1f}%  {bar}")

fig, (a,b) = plt.subplots(1,2,figsize=(14,6))
a.imshow(img); a.set_title("Input Image"); a.axis("off")
sorted_pairs = sorted(zip(class_names,probs),key=lambda x:x[1])
names_s = [x[0] for x in sorted_pairs]
probs_s = [x[1] for x in sorted_pairs]
colors_bar = ["lime" if p==max(probs_s) else P.accent2 for p in probs_s]
b.barh(names_s,probs_s,color=colors_bar); b.set_xlabel("Probability")
b.set_title("CLIP Zero-Shot Predictions"); b.set_facecolor("#0d1529")
plt.tight_layout(); plt.savefig("m6_clip.png",dpi=150); plt.show()`,
  },
  {
    id:"M7", title:"Generative Vision: GANs, VAEs, and Diffusion", level:"Advanced", time:"60 min",
    color:P.accent8,
    sections:[
      { heading:"Variational Autoencoders (VAEs)",
        body:`A VAE (Kingma and Welling, 2014) consists of an encoder q(z|x) that maps input x to a distribution over latent space z (parameterised as mean mu and log-variance log_sigma^2), and a decoder p(x|z) that reconstructs x from a sampled z. Training minimises the Evidence Lower BOund (ELBO): L = E[log p(x|z)] - KL(q(z|x) || p(z)), where the first term is reconstruction loss (pixel MSE or cross-entropy) and the second term is KL divergence regularising the posterior toward a standard Gaussian prior.

The reparameterisation trick enables backpropagation through the stochastic sampling: z = mu + sigma * epsilon, where epsilon ~ N(0,I). This separates the randomness (epsilon) from the learnable parameters (mu, sigma).

VAEs produce smooth, well-structured latent spaces suitable for interpolation, arithmetic, and conditional generation. However, VAE samples tend to be blurry because the decoder averages over all possible reconstructions, optimising expected log-likelihood.` },
      { heading:"GAN Training Dynamics",
        body:`GAN training is a minimax game: Generator G minimises log(1-D(G(z))); Discriminator D maximises log(D(x)) + log(1-D(G(z))). In practice, G maximises log(D(G(z))) (non-saturating loss) for stronger gradients. This game reaches a Nash equilibrium where D(x)=0.5 for all x (cannot distinguish real from generated).

Training challenges: (1) Mode collapse: G collapses to producing a few high-scoring samples, ignoring diversity. (2) Training instability: G and D can oscillate without convergence. (3) Gradient vanishing: if D is too good, D(G(z)) is near 0 and gradients to G vanish. Solutions: Wasserstein GAN (WGAN) replaces JS divergence with Wasserstein-1 distance, gradient penalty (WGAN-GP) enforces the Lipschitz constraint, spectral normalisation controls Discriminator gradients, and training tricks (label smoothing, instance noise, spectral normalisation). Progressive GAN grows both networks from 4x4 to high resolution, maintaining training stability.` },
      { heading:"Diffusion Models: Deep Dive",
        body:`Forward process (fixed, no learning): q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t)*x_{t-1}, beta_t*I). With a noise schedule beta_1,...,beta_T (typically 1000 steps), x_T is approximately pure Gaussian noise. Using the closed-form: x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*epsilon, where alpha_bar_t = product of (1-beta_i) for i=1..t.

Reverse process (learned): p_theta(x_{t-1}|x_t) = N(x_{t-1}; mu_theta(x_t,t), sigma_t^2*I). The U-Net learns to predict the noise epsilon from x_t and t, parameterising mu_theta via: mu_theta(x_t,t) = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_theta(x_t,t)).

Training: simply minimise MSE between true noise epsilon and predicted noise epsilon_theta. Elegant and stable compared to GAN training. Inference: iteratively denoise from x_T ~ N(0,I) for T steps. DDIM enables deterministic sampling in 50-100 steps. Latent Diffusion (Stable Diffusion) encodes x to latent space z = E(x), applies diffusion on z, decodes z_0 with D.` },
      { heading:"ControlNet: Conditional Generation",
        body:`ControlNet (Zhang et al., 2023) adds spatial conditioning to pretrained diffusion models without modifying the original weights. It clones the encoder and middle blocks of the Stable Diffusion U-Net, connects the clone via "zero convolutions" (1x1 convolutions initialised to zero), and accepts a conditioning image (edges, depth, pose, segmentation) as input.

Zero initialisation is critical: at the start of training, the ControlNet output is exactly zero, so the full model behaves identically to the original SD model. As training progresses, ControlNet gradually learns to inject spatial conditioning. This allows fine-grained control: generate photorealistic images matching a given edge map, depth map, or human pose skeleton.

Applications: architecture design (condition on floor plan), character animation (condition on pose), image restoration (condition on degraded input), medical image generation (condition on anatomical segmentation), and film production (condition on storyboard sketches).` },
    ],
    code:`# VAE latent space interpolation + DDPM noise schedule visualisation
import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt

# ── Minimal VAE ───────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self,in_c=1,latent=16):
        super().__init__()
        self.net=nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())
        self.mu =nn.Linear(128,latent)
        self.lv =nn.Linear(128,latent)
    def forward(self,x):
        h=self.net(x); return self.mu(h),self.lv(h)

class Decoder(nn.Module):
    def __init__(self,latent=16,out_c=1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(latent,128),nn.ReLU(),nn.Linear(128,256),nn.ReLU(),
                               nn.Linear(256,784),nn.Sigmoid())
    def forward(self,z): return self.net(z).view(-1,1,28,28)

class VAE(nn.Module):
    def __init__(self,latent=16):
        super().__init__()
        self.enc=Encoder(latent=latent); self.dec=Decoder(latent=latent)
    def reparametrize(self,mu,lv):
        std=torch.exp(0.5*lv)
        eps=torch.randn_like(std)
        return mu+eps*std
    def forward(self,x):
        mu,lv=self.enc(x); z=self.reparametrize(mu,lv)
        return self.dec(z),mu,lv

vae=VAE(latent=16)
total=sum(p.numel() for p in vae.parameters())
print(f"VAE parameters: {total:,}")
x_demo=torch.randn(4,1,28,28)
recon,mu,lv=vae(x_demo)
print(f"Recon shape: {recon.shape}, mu shape: {mu.shape}")

# ── DDPM noise schedule visualisation ────────────────────────
T=1000
beta = torch.linspace(1e-4,0.02,T)          # Linear schedule
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# Show image at different noise levels
url="https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
import urllib.request,io
from PIL import Image
img_pil=Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB").resize((64,64))
x0=(torch.tensor(np.array(img_pil)).float()/127.5-1).permute(2,0,1).unsqueeze(0)

steps=[0,100,250,500,750,999]
fig,axes=plt.subplots(1,len(steps),figsize=(18,4))
for ax,t in zip(axes,steps):
    ab=alpha_bar[t]
    eps=torch.randn_like(x0)
    xt=(ab**0.5)*x0+((1-ab)**0.5)*eps
    im=(xt.squeeze().permute(1,2,0).numpy()*0.5+0.5).clip(0,1)
    ax.imshow(im); ax.set_title(f"t={t}\\nalpha_bar={ab.item():.3f}"); ax.axis("off")
plt.suptitle("DDPM Forward Diffusion Process (x_0 to x_T)",fontsize=13)
plt.tight_layout(); plt.savefig("m7_diffusion.png",dpi=150); plt.show()`,
  },
  {
    id:"M8", title:"3D Vision: Point Clouds, NeRF, and Gaussian Splatting", level:"Advanced", time:"60 min",
    color:P.accent9,
    sections:[
      { heading:"Camera Models and Projective Geometry",
        body:`A camera projects a 3D world point X = (X,Y,Z) to a 2D image point x = (u,v) via the pinhole camera model: u = f_x * X/Z + c_x, v = f_y * Y/Z + c_y, where (f_x, f_y) is the focal length in pixels and (c_x, c_y) is the principal point (optical centre). The intrinsic matrix K encodes these parameters.

In homogeneous coordinates: lambda * [u,v,1]^T = K * [R|t] * [X,Y,Z,1]^T, where [R|t] is the 3x4 camera extrinsic matrix (rotation and translation). Homogeneous coordinates allow projective transformations to be expressed as matrix multiplications.

Epipolar geometry: given a point in image 1, the corresponding point in image 2 lies on a line (the epipolar line). The fundamental matrix F encodes this: x2^T * F * x1 = 0. The essential matrix E = K2^T * F * K1 encodes the relative rotation R and translation t between cameras (up to scale). This is the basis of SfM and stereo reconstruction.` },
      { heading:"PointNet: Learning Directly on Point Sets",
        body:`PointNet (Qi et al. 2017) was the first deep learning method operating directly on raw 3D point sets (not voxels or multi-view images). Key design principles: (1) Permutation invariance: apply a shared MLP to each point independently (no ordering), then aggregate all point features with a symmetric function (max-pooling) to get a global descriptor. (2) Transformation invariance: a mini-network (T-Net) predicts an affine transformation of input points and features, aligning them to a canonical pose.

Architecture: Input transform (3x3) -> Shared MLP (64, 128, 1024) -> Max-pool -> Global feature (1024) -> FC layers for classification. For segmentation, the global feature is concatenated back to each point's local feature.

PointNet++ adds hierarchical feature learning with farthest point sampling, ball query grouping, and set abstraction layers, capturing local structure at multiple scales, analogous to how CNNs build hierarchical features in 2D.` },
      { heading:"Neural Radiance Fields (NeRF)",
        body:`NeRF (Mildenhall et al., 2020) represents a static scene as a continuous 5D function F: (x,y,z,theta,phi) -> (RGB, sigma), where (x,y,z) is a 3D position, (theta,phi) is viewing direction, RGB is emitted colour, and sigma is volume density (opacity). F is parameterised by a Multi-Layer Perceptron.

Volume rendering: to render a pixel, march a ray from the camera through the scene, sample N points along the ray, query the MLP at each sample, and integrate colour and density: C(r) = integral of T(t) * sigma(t) * c(t) dt, where T(t) = exp(-integral of sigma(s)ds) is the accumulated transmittance (probability of reaching point t without hitting anything).

Training: given a set of posed images (known camera parameters), minimise the reconstruction loss between rendered and observed pixel colours. After training, novel views can be rendered from arbitrary viewpoints. Limitations: per-scene optimisation (minutes to hours), not real-time rendering, and difficulty with dynamic scenes. Instant-NGP uses multi-resolution hash encoding to achieve seconds-long training. Generalizable NeRFs learn across scenes.` },
      { heading:"3D Gaussian Splatting",
        body:`3D Gaussian Splatting (Kerbl et al., 2023) represents a scene as millions of differentiable 3D Gaussians, each with: position (mean) mu in 3D, covariance matrix Sigma (encoding shape and orientation via scale S and rotation R: Sigma = R*S*S^T*R^T), opacity alpha, and view-dependent colour (represented by spherical harmonics coefficients).

Rendering (GPU-accelerated rasterisation): project each 3D Gaussian to a 2D image plane (as a 2D Gaussian ellipse), sort by depth, and alpha-composite front-to-back. Differentiable: gradients can propagate through the rendering process to update Gaussian parameters.

Training: start from a sparse SfM point cloud; initialise one Gaussian per point; optimise via gradient descent on photometric loss; adaptive control (densification: split or clone Gaussians in under-reconstructed areas; pruning: remove transparent or very small Gaussians). Achieves real-time rendering at 100+ FPS at 1080p, matching or exceeding NeRF quality with dramatically faster rendering.` },
    ],
    code:`# 3D Gaussian visualisation + NeRF volume rendering concept
import numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

# ── 3D Gaussians: generate and visualise ─────────────────────
rng = np.random.default_rng(42)
n_gaussians = 500

# Random Gaussians in 3D (simulating a simple scene)
means   = rng.standard_normal((n_gaussians,3))
scales  = rng.exponential(0.3,(n_gaussians,3)) + 0.05
opacities = rng.uniform(0.3,1.0,n_gaussians)
colors  = rng.uniform(0,1,(n_gaussians,4))
colors[:,3] = opacities

fig = plt.figure(figsize=(16,6))
ax1 = fig.add_subplot(131,projection="3d")
ax1.scatter(means[:,0],means[:,1],means[:,2],c=colors[:,:3],s=scales.mean(1)*200,alpha=0.6)
ax1.set_title("3D Gaussian Means"); ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")

# ── NeRF: volume rendering concept ───────────────────────────
# Simple 1D transmittance demo along a ray
T_steps = 200
density  = np.zeros(T_steps)
density[40:60]  = 3.0   # first object (dense)
density[120:140]= 5.0   # second object (denser)
density += rng.exponential(0.05,T_steps)  # background noise

t = np.linspace(0,1,T_steps)
delta = t[1]-t[0]
transmittance = np.exp(-np.cumsum(density*delta))
weight = transmittance * density * delta   # contribution per sample
weight /= weight.sum() + 1e-8

ax2 = fig.add_subplot(132)
ax2.fill_between(t,density/density.max(),alpha=0.4,color=P.accent5,label="Density sigma(t)")
ax2.fill_between(t,transmittance,alpha=0.4,color=P.accent1,label="Transmittance T(t)")
ax2.fill_between(t,weight*10,alpha=0.6,color=P.accent3,label="Weight (x10)")
ax2.legend(facecolor="#111"); ax2.set_title("NeRF: Volume Rendering Along a Ray")
ax2.set_xlabel("Distance along ray t"); ax2.set_facecolor("#0d1529")

# ── Camera projection demo ────────────────────────────────────
def project(pts3d, K):
    """pts3d: (N,3), K: (3,3)"""
    p = pts3d / pts3d[:,2:3]
    return (K @ p.T).T[:,:2]

K = np.array([[500,0,320],[0,500,240],[0,0,1]],float)
cube = np.array([[0,0,2],[1,0,2],[1,1,2],[0,1,2],
                 [0,0,3],[1,0,3],[1,1,3],[0,1,3]],float) - [0.5,0.5,0]
proj = project(cube, K)

ax3 = fig.add_subplot(133)
edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
for i,j in edges: ax3.plot(*zip(proj[i],proj[j]),color=P.accent2)
ax3.scatter(proj[:,0],proj[:,1],s=50,color=P.accent3,zorder=5)
ax3.set_title("Pinhole Camera Projection (3D cube -> 2D)")
ax3.set_facecolor("#0d1529"); ax3.invert_yaxis()

plt.tight_layout(); plt.savefig("m8_3dvision.png",dpi=150); plt.show()`,
  },
  {
    id:"M9", title:"Deployment: From Research to Production", level:"Advanced", time:"50 min",
    color:P.accent1,
    sections:[
      { heading:"Model Compression: Quantisation and Pruning",
        body:`Production deployment requires smaller, faster models that run within memory and latency budgets. Four main techniques:

Quantisation: reduce numerical precision of weights and activations from float32 (32 bits) to float16 (16 bits), int8 (8 bits), or even int4 (4 bits). Post-training quantisation (PTQ) applies quantisation to a pretrained model with a small calibration dataset. Quantisation-aware training (QAT) fine-tunes with simulated quantisation, recovering accuracy lost by PTQ. int8 quantisation typically causes < 1% accuracy drop while halving memory and doubling throughput on supported hardware.

Pruning: remove redundant weights (weight pruning, zeroing individual weights) or entire channels/filters (structured pruning, easier to accelerate). Lottery ticket hypothesis (Frankle et al.): sparse sub-networks ("winning tickets") can be trained to full accuracy from the start if their initial values are kept. Structured pruning removes entire attention heads or layers, reducing FLOPs directly.

Knowledge Distillation: train a small "student" model to match the predictions (soft logits) of a large "teacher" model, transferring the teacher's knowledge. The student is often 10-100x smaller than the teacher.` },
      { heading:"Model Export: ONNX and TensorRT",
        body:`PyTorch models must be exported for deployment on production systems. ONNX (Open Neural Network Exchange) is an open standard format for representing ML models, enabling framework-independent deployment: export a PyTorch model to ONNX, then run it with ONNX Runtime, TensorRT, CoreML, or any other ONNX-compatible runtime.

Export: torch.onnx.export(model, dummy_input, "model.onnx", opset_version=17). ONNX Runtime applies graph optimisations (constant folding, operator fusion) automatically. Typical 2-4x speedup over native PyTorch inference on CPU.

TensorRT (NVIDIA) applies aggressive optimisations for NVIDIA GPU: layer fusion, kernel auto-tuning, mixed precision (FP16+INT8), and batching. Achieves 5-15x speedup over PyTorch on GPU. Required for production NVIDIA deployments. Torch-TensorRT (formerly TRTorch) automates the conversion.

CoreML (Apple): deploy on Apple Silicon (iPhone, iPad, Mac) with hardware-accelerated inference via Neural Engine. Export via coremltools from PyTorch or ONNX.` },
      { heading:"Calibration and Uncertainty Quantification",
        body:`A model is calibrated if its confidence scores reflect true probabilities: a model predicting 80% confidence should be correct 80% of the time. Most neural networks are overconfident: they output high-confidence predictions even when they are wrong. Calibration matters critically in medical, safety, and financial applications.

Temperature scaling is the simplest post-hoc calibration method: divide logits by a learned scalar T before softmax, using a held-out validation set to find T that minimises Expected Calibration Error (ECE). T > 1 reduces confidence (softens predictions); T < 1 increases confidence.

Uncertainty estimation methods: Monte Carlo Dropout (apply dropout at test time, take multiple forward passes, estimate uncertainty as variance of predictions); Deep Ensembles (train K independent models, use disagreement as uncertainty); Conformal Prediction (distribution-free prediction sets with guaranteed coverage). ECE (Expected Calibration Error) and MCE (Maximum Calibration Error) measure calibration quality. Reliability diagrams (confidence vs accuracy per bin) visualise calibration.` },
      { heading:"Real-Time Inference: Optimisation Strategies",
        body:`Throughput vs latency: throughput is samples per second (maximised by large batches); latency is time per single sample (minimised by no batching). Most edge applications require low latency; server applications prioritise throughput.

Key strategies: (1) Batch inference: process N samples simultaneously to saturate GPU parallelism. (2) Half precision (fp16): halves memory bandwidth, doubles throughput on tensor cores, minimal accuracy loss. (3) JIT compilation: torch.jit.script or torch.compile (PyTorch 2.0) compiles model graphs, applying kernel fusion and memory layout optimisations. (4) Asynchronous preprocessing: move data loading and augmentation off the critical path onto CPU threads. (5) Model caching: ONNX Runtime sessions are expensive to initialise; cache them across requests. (6) Horizontal scaling: deploy multiple model replicas behind a load balancer.

Edge-specific: MobileNet/EfficientNet-Lite/NASNet for mobile; YOLO Nano for embedded; INT8 quantisation for microcontrollers (TensorFlow Lite, ONNX Runtime Mobile).` },
    ],
    code:`# Production deployment pipeline: quantisation, ONNX export, latency benchmarking
import torch, time, numpy as np
import torchvision.models as models
from torchvision import transforms
import urllib.request, io
from PIL import Image

model_fp32 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model_fp32.eval()

# ── Dynamic INT8 Quantisation ─────────────────────────────────
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8)

# ── Benchmarking ──────────────────────────────────────────────
def benchmark(model, x, n=100, warmup=10):
    with torch.no_grad():
        for _ in range(warmup): model(x)
    times=[]
    with torch.no_grad():
        for _ in range(n):
            t0=time.perf_counter(); model(x); times.append(time.perf_counter()-t0)
    return np.array(times)

x = torch.randn(1,3,224,224)
t_fp32 = benchmark(model_fp32, x)
t_int8 = benchmark(model_int8, x)

print(f"FP32 latency: {t_fp32.mean()*1000:.2f} +/- {t_fp32.std()*1000:.2f} ms")
print(f"INT8 latency: {t_int8.mean()*1000:.2f} +/- {t_int8.std()*1000:.2f} ms")
print(f"Speedup:      {t_fp32.mean()/t_int8.mean():.2f}x")

# ── Model size comparison ─────────────────────────────────────
import os
torch.save(model_fp32.state_dict(), "/tmp/fp32.pt")
torch.save(model_int8.state_dict(), "/tmp/int8.pt")
fp32_mb = os.path.getsize("/tmp/fp32.pt")/1e6
int8_mb = os.path.getsize("/tmp/int8.pt")/1e6
print(f"FP32 size: {fp32_mb:.1f} MB")
print(f"INT8 size: {int8_mb:.1f} MB")
print(f"Compression: {fp32_mb/int8_mb:.2f}x")

# ── ONNX export ───────────────────────────────────────────────
torch.onnx.export(model_fp32, x, "/tmp/resnet18.onnx",
    opset_version=17, input_names=["input"], output_names=["logits"],
    dynamic_axes={"input":{0:"batch"},"logits":{0:"batch"}})
onnx_mb = os.path.getsize("/tmp/resnet18.onnx")/1e6
print(f"ONNX size: {onnx_mb:.1f} MB")

import matplotlib.pyplot as plt
labels = ["FP32 PyTorch","INT8 PyTorch","ONNX export"]
latencies = [t_fp32.mean()*1000, t_int8.mean()*1000, None]
sizes     = [fp32_mb, int8_mb, onnx_mb]

fig,(a,b)=plt.subplots(1,2,figsize=(12,5))
a.bar(["FP32","INT8"],[t_fp32.mean()*1000,t_int8.mean()*1000],
      color=[P.accent5,P.accent3],edgecolor="none")
a.set_ylabel("Latency (ms)"); a.set_title("Inference Latency (CPU)"); a.set_facecolor("#0d1529")
b.bar(labels,[fp32_mb,int8_mb,onnx_mb],color=[P.accent5,P.accent3,P.accent1],edgecolor="none")
b.set_ylabel("Size (MB)"); b.set_title("Model Size Comparison"); b.set_facecolor("#0d1529")
plt.tight_layout(); plt.savefig("m9_deploy.png",dpi=150); plt.show()`,
  },
];

/* ================================================================
   100 LEETCODE-STYLE CODING CHALLENGES
================================================================ */
const CHALLENGES = [
  // ─── EASY (1-30) ───────────────────────────────────────────────
  { id:"C01", difficulty:"Easy", tag:"Arrays / Pixels", title:"Flip and Invert Binary Image",
    company:"Google, Meta", points:10,
    desc:`Given a binary image as a 2-D list of 0s and 1s, horizontally flip each row (reverse), then invert every bit (0 becomes 1, 1 becomes 0). Return the resulting image.`,
    example:`Input:  [[1,1,0],[1,0,1],[0,0,0]]
Output: [[1,0,0],[0,1,0],[1,1,1]]`,
    hint:"Step 1: reverse each row with [::-1]. Step 2: XOR each element with 1 (x^1 flips a single bit). Both steps can be combined in one list comprehension.",
    fullSolution:`# DETAILED SOLUTION
# The problem has two operations applied in sequence:
# 1. Horizontal flip: reverse each row
# 2. Bit inversion: 0->1, 1->0

# Key insight: XOR with 1 flips a single bit.
# Both ops can be combined: flip then invert = XOR each element
# of the reversed row with 1. In Python, x^1 on a bit does this.

def flipAndInvertImage(image):
    # For each row, reverse it with [::-1], then XOR each element with 1
    return [[x ^ 1 for x in row[::-1]] for row in image]

# --- Test ---
img = [[1,1,0],[1,0,1],[0,0,0]]
result = flipAndInvertImage(img)
print("Result:", result)
# Expected: [[1,0,0],[0,1,0],[1,1,1]]

# Step-by-step trace for [[1,1,0]]:
# After flip:   [0,1,1]
# After invert: [1,0,0]  <-- correct

# NumPy extension (RGB image, 3 channels):
import numpy as np
rgb_img = np.random.randint(0,2,(4,4,3))
flipped_rgb = rgb_img[:, ::-1, :]  # horizontal flip (no invert needed for RGB)
print("RGB flipped shape:", flipped_rgb.shape)`,
    complexity:"Time O(m*n)  Space O(m*n) for output (O(1) in-place variant)",
    followup:"How would you extend this to an RGB image? How does this relate to image mirroring for data augmentation?" },

  { id:"C02", difficulty:"Easy", tag:"2-D Prefix Sum", title:"Count Black Pixels in Rectangles",
    company:"Amazon, Bloomberg", points:10,
    desc:`Given a binary image matrix and queries [r1,c1,r2,c2], count the number of 1s (black pixels) inside each rectangle in O(1) per query after O(m*n) preprocessing.`,
    example:`Matrix:
  1 0 1
  0 1 0
  1 1 1
Query(0,0,2,2) -> 6   Query(0,0,1,1) -> 2`,
    hint:"Build a 2-D prefix sum table P where P[i][j] = sum of all elements in the rectangle from (0,0) to (i-1,j-1). Answer = P[r2+1][c2+1] - P[r1][c2+1] - P[r2+1][c1] + P[r1][c1].",
    fullSolution:`# DETAILED SOLUTION
# 2D Prefix Sum (Integral Image) - the same technique used in
# Viola-Jones face detection (Haar features computed in O(1)).

# Build prefix sum: P[i][j] = sum of all elements mat[0..i-1][0..j-1]
# Using inclusion-exclusion to fill:
#   P[i][j] = mat[i-1][j-1] + P[i-1][j] + P[i][j-1] - P[i-1][j-1]

def build_prefix(mat):
    m, n = len(mat), len(mat[0])
    P = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            P[i][j] = mat[i-1][j-1] + P[i-1][j] + P[i][j-1] - P[i-1][j-1]
    return P

def query(P, r1, c1, r2, c2):
    # Inclusion-exclusion on the prefix table (1-indexed boundary)
    return P[r2+1][c2+1] - P[r1][c2+1] - P[r2+1][c1] + P[r1][c1]

mat = [[1,0,1],[0,1,0],[1,1,1]]
P   = build_prefix(mat)
print(query(P,0,0,2,2))  # -> 6 (entire matrix has 6 ones)
print(query(P,0,0,1,1))  # -> 2 (top-left 2x2 has 2 ones: mat[0][0]=1, mat[1][1]=1)
print(query(P,2,0,2,2))  # -> 3 (bottom row: 1+1+1=3)

# NumPy equivalent (integral image):
import numpy as np
mat_np = np.array(mat)
integral = np.zeros((4,4),int)
integral[1:,1:] = np.cumsum(np.cumsum(mat_np,0),1)
print("NumPy integral image query (0,0,2,2):", integral[3,3])`,
    complexity:"Time O(m*n) build + O(1) per query  Space O(m*n)",
    followup:"How does this relate to OpenCV's integral() function? How is it used to compute Haar cascade features in real-time face detection?" },

  { id:"C03", difficulty:"Easy", tag:"BFS / Flood Fill", title:"Flood Fill (Paint Bucket)",
    company:"Amazon, Google", points:10,
    desc:`Implement the flood-fill algorithm (the paint-bucket tool in image editors). Given an image, a starting pixel (sr, sc), and a new colour, repaint the starting pixel and all 4-connected pixels sharing the original colour.`,
    example:`image=[[1,1,1],[1,1,0],[1,0,1]]  sr=1, sc=1, newColor=2
Output: [[2,2,2],[2,2,0],[2,0,1]]`,
    hint:"BFS from (sr,sc). Record the original colour. Skip pixels that are already the new colour or have a different colour.",
    fullSolution:`# DETAILED SOLUTION
# Flood fill visits all 4-connected neighbours with the same original colour.
# BFS is iterative (safer for large images vs recursive DFS which can hit Python's stack limit).

from collections import deque

def floodFill(image, sr, sc, color):
    orig = image[sr][sc]
    if orig == color:          # Already the target colour; nothing to do
        return image
    m, n = len(image), len(image[0])
    q = deque([(sr, sc)])
    image[sr][sc] = color      # Paint start pixel
    
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:   # 4-connectivity
            nr, nc = r+dr, c+dc
            if 0 <= nr < m and 0 <= nc < n and image[nr][nc] == orig:
                image[nr][nc] = color
                q.append((nr, nc))
    return image

# Test
print(floodFill([[1,1,1],[1,1,0],[1,0,1]], 1, 1, 2))
# Expected: [[2,2,2],[2,2,0],[2,0,1]]

# --- OpenCV equivalent (real image flood fill) ---
import cv2, numpy as np, urllib.request, matplotlib.pyplot as plt

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
raw = urllib.request.urlopen(url).read()
img = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_COLOR)

# OpenCV floodFill: fills from seed point with given colour
seed_point = (img.shape[1]//2, img.shape[0]//4)   # (x,y) format
flood = img.copy()
mask  = np.zeros((img.shape[0]+2, img.shape[1]+2), np.uint8)
cv2.floodFill(flood, mask, seed_point, (0,200,255), (30,30,30), (30,30,30))

fig,(a,b) = plt.subplots(1,2,figsize=(12,5))
a.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)); a.set_title("Original"); a.axis("off")
b.imshow(cv2.cvtColor(flood,cv2.COLOR_BGR2RGB)); b.set_title("Flood Filled"); b.axis("off")
plt.tight_layout(); plt.savefig("c03.png",dpi=150); plt.show()`,
    complexity:"Time O(m*n)  Space O(m*n) for queue",
    followup:"How would you implement 8-connectivity (diagonals)? How does flood fill relate to region growing in medical image segmentation?" },

  { id:"C04", difficulty:"Easy", tag:"Sliding Window / Convolution", title:"Image Smoother (Mean Filter)",
    company:"Microsoft, Apple", points:10,
    desc:`Apply a 3x3 mean filter to a grayscale image WITHOUT using any library convolution functions. Each output pixel is the floor of the average of its valid neighbours.`,
    example:`Input:  [[1,1,1],[1,0,1],[1,1,1]]
Output: [[0,0,0],[0,0,0],[0,0,0]]  (all averages < 1, floor to 0)`,
    hint:"Iterate every pixel. Collect all valid neighbours (including self) within a 3x3 window, clamping coordinates to image bounds. Compute floor(mean).",
    fullSolution:`# DETAILED SOLUTION
# This is a manual implementation of a 3x3 box filter, identical to
# what cv2.blur(img,(3,3)) does internally.
# Understanding this from scratch is essential before using library functions.

import math

def imageSmoother(img):
    m, n = len(img), len(img[0])
    result = [[0]*n for _ in range(m)]
    
    for i in range(m):
        for j in range(n):
            total = 0
            count = 0
            # Iterate 3x3 neighbourhood, clamping to valid range
            for di in range(-1, 2):      # -1, 0, 1
                for dj in range(-1, 2):  # -1, 0, 1
                    ni, nj = i+di, j+dj
                    if 0 <= ni < m and 0 <= nj < n:
                        total += img[ni][nj]
                        count += 1
            result[i][j] = math.floor(total / count)
    return result

# Test
img = [[1,1,1],[1,0,1],[1,1,1]]
print(imageSmoother(img))

# Apply to a real image and compare methods
import cv2, numpy as np, urllib.request, matplotlib.pyplot as plt

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw = urllib.request.urlopen(url).read()
gray = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_GRAYSCALE)
gray = cv2.resize(gray,(160,120))

rng  = np.random.default_rng(0)
noisy= np.clip(gray.astype(int)+rng.integers(-40,40,gray.shape),0,255).astype(np.uint8)

box_3   = cv2.blur(noisy,(3,3))
box_7   = cv2.blur(noisy,(7,7))
gauss_5 = cv2.GaussianBlur(noisy,(5,5),1.5)
median  = cv2.medianBlur(noisy,5)

fig,axes=plt.subplots(1,5,figsize=(20,4))
for ax,im,t in zip(axes,[noisy,box_3,box_7,gauss_5,median],
    ["Noisy","Box 3x3","Box 7x7","Gaussian 5x5","Median 5x5"]):
    ax.imshow(im,cmap="gray"); ax.set_title(t); ax.axis("off")
plt.tight_layout(); plt.savefig("c04.png",dpi=150); plt.show()`,
    complexity:"Time O(m*n*k^2) for k x k kernel  Space O(m*n)",
    followup:"Compare box filter vs Gaussian blur vs median filter for noise removal. When is median filter preferable (salt-and-pepper noise)?" },

  { id:"C05", difficulty:"Easy", tag:"BFS / Connected Components", title:"Count and Label Connected Components",
    company:"Microsoft, NVIDIA", points:10,
    desc:`Given a binary image, count the number of connected components (groups of 4-connected white pixels). Return the count and a label map where each component has a unique integer label.`,
    example:`Input:
  1 0 0 1 1
  1 0 0 0 1
  0 0 1 0 0
Output: 3 components, labels [[1,0,0,2,2],[1,0,0,0,2],[0,0,3,0,0]]`,
    hint:"BFS from every unvisited white pixel. Each new BFS call starts a new component. Use a visited array or modify the image in-place.",
    fullSolution:`# DETAILED SOLUTION
# Connected component labelling is foundational for blob analysis,
# object counting, and region-based image processing.
# This is what cv2.connectedComponents() does under the hood.

from collections import deque
import numpy as np

def label_components(img):
    m, n    = len(img), len(img[0])
    labels  = [[0]*n for _ in range(m)]
    visited = [[False]*n for _ in range(m)]
    comp_id = 0
    
    for i in range(m):
        for j in range(n):
            if img[i][j] == 1 and not visited[i][j]:
                comp_id += 1
                q = deque([(i,j)])
                visited[i][j] = True
                labels[i][j]  = comp_id
                while q:
                    r, c = q.popleft()
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0<=nr<m and 0<=nc<n and img[nr][nc]==1 and not visited[nr][nc]:
                            visited[nr][nc] = True
                            labels[nr][nc]  = comp_id
                            q.append((nr, nc))
    return comp_id, labels

img = [[1,0,0,1,1],[1,0,0,0,1],[0,0,1,0,0]]
count, lbls = label_components(img)
print(f"Components: {count}")
for row in lbls: print(row)

# --- OpenCV version on a real image ---
import cv2, urllib.request, matplotlib.pyplot as plt

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw = urllib.request.urlopen(url).read()
img = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_GRAYSCALE)
_, bw = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)

n, labels, stats, centroids = cv2.connectedComponentsWithStats(bw.astype(np.uint8))
print(f"\\nOpenCV found {n-1} components (excl. background)")
print(f"Largest component area: {stats[1:,cv2.CC_STAT_AREA].max()} pixels")

label_display = (labels % 10 * 25).astype(np.uint8)  # colour-cycle for display
fig,(a,b)=plt.subplots(1,2,figsize=(12,5))
a.imshow(bw,cmap="gray"); a.set_title("Binary Image"); a.axis("off")
b.imshow(label_display,cmap="tab10"); b.set_title(f"Component Labels ({n-1} total)"); b.axis("off")
plt.tight_layout(); plt.savefig("c05.png",dpi=150); plt.show()`,
    complexity:"Time O(m*n)  Space O(m*n)",
    followup:"How does Union-Find give O(alpha(n)) per operation? When would you use 8-connectivity instead of 4?" },

  { id:"C06", difficulty:"Easy", tag:"Histogram", title:"Image Histogram and Equalisation",
    company:"Adobe, NVIDIA", points:10,
    desc:`Implement histogram computation and histogram equalisation from scratch for a grayscale image. Equalisation redistributes pixel intensities so the cumulative distribution function (CDF) is approximately linear.`,
    example:`Input: dark image with most pixels in [0-80] range
Output: equalised image with pixels spread across [0-255]`,
    hint:"Histogram H[v] = count of pixels with intensity v. CDF[v] = sum(H[0..v]). Equalised pixel = round((CDF[v]-CDF_min) / (total_pixels-CDF_min) * 255).",
    fullSolution:`# DETAILED SOLUTION
# Histogram equalisation improves contrast by making the intensity distribution
# uniform. It is the basis of CLAHE (used in low-light enhancement, retinography).

import numpy as np, urllib.request, cv2, matplotlib.pyplot as plt

def histogram(img):
    h = np.zeros(256, int)
    for v in img.flatten():
        h[v] += 1
    return h

def equalise(img):
    hist  = histogram(img)
    cdf   = np.cumsum(hist)                           # Cumulative distribution
    cdf_min = cdf[cdf > 0][0]                         # Smallest non-zero CDF value
    N     = img.size                                  # Total pixels
    lut   = np.round((cdf - cdf_min) / (N - cdf_min) * 255).astype(np.uint8)
    return lut[img]                                   # Apply look-up table

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"
raw = urllib.request.urlopen(url).read()
gray= cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_GRAYSCALE)
dark= np.clip(gray.astype(int)//3, 0, 255).astype(np.uint8)   # Simulate dark image

eq_manual = equalise(dark)
eq_cv      = cv2.equalizeHist(dark)
clahe_obj  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
clahe_out  = clahe_obj.apply(dark)

fig, axes = plt.subplots(2,4,figsize=(20,10))
imgs   = [dark,eq_manual,eq_cv,clahe_out]*2
titles = ["Dark original","Manual HE","OpenCV HE","CLAHE"] + ["Histogram"]*4
for i,(ax,im,t) in enumerate(zip(axes.flat,imgs,titles)):
    if i < 4:
        ax.imshow(im,cmap="gray"); ax.set_title(t)
    else:
        ax.hist(im.flatten(),bins=128,color=["cyan","magenta","yellow","lime"][i-4],alpha=0.8)
        ax.set_xlim(0,255); ax.set_title(f"{titles[i-4]} histogram")
    ax.axis("off" if i<4 else "on")
plt.tight_layout(); plt.savefig("c06.png",dpi=150); plt.show()`,
    complexity:"Time O(m*n)  Space O(256) for histogram",
    followup:"When would CLAHE outperform global HE? (Answer: when the image has regions with very different local contrast, e.g. a dark shadow next to a bright window.)" },

  { id:"C07", difficulty:"Easy", tag:"Morphology", title:"Erosion and Dilation from Scratch",
    company:"OpenCV, Adobe", points:10,
    desc:`Implement binary erosion and dilation using a 3x3 square structuring element, without using any library morphological functions.`,
    example:`Input binary image, erode removes border pixels, dilate expands foreground pixels.`,
    hint:"Erosion: output=1 only if ALL neighbourhood pixels are 1. Dilation: output=1 if ANY neighbourhood pixel is 1.",
    fullSolution:`# DETAILED SOLUTION
# Morphological operations form the foundation of classical CV:
# opening = erode then dilate (removes small noise blobs)
# closing = dilate then erode (fills small holes)

import numpy as np, cv2, urllib.request, matplotlib.pyplot as plt

def erode_manual(img, k=3):
    """Binary erosion: output pixel=1 only if all kernel-covered pixels are 1."""
    pad = k//2
    padded = np.pad(img, pad, mode='constant', constant_values=0)
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            patch = padded[i:i+k, j:j+k]
            out[i,j] = 1 if patch.min() == 1 else 0
    return out

def dilate_manual(img, k=3):
    """Binary dilation: output pixel=1 if any kernel-covered pixel is 1."""
    pad = k//2
    padded = np.pad(img, pad, mode='constant', constant_values=0)
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            patch = padded[i:i+k, j:j+k]
            out[i,j] = 1 if patch.max() == 1 else 0
    return out

# Create a test binary image
binary = np.zeros((60,60), np.uint8)
cv2.circle(binary,(30,30),20,1,-1)  # White circle
cv2.rectangle(binary,(5,5),(15,15),1,-1)  # Small noise blob

eroded  = erode_manual(binary)
dilated = dilate_manual(binary)
opened  = dilate_manual(erode_manual(binary))   # removes noise blob
closed  = erode_manual(dilate_manual(binary))   # fills holes

fig,axes=plt.subplots(1,5,figsize=(20,5))
for ax,im,t in zip(axes,[binary,eroded,dilated,opened,closed],
                        ["Original","Eroded","Dilated","Opening","Closing"]):
    ax.imshow(im,cmap="gray",vmin=0,vmax=1); ax.set_title(t); ax.axis("off")
plt.tight_layout(); plt.savefig("c07.png",dpi=150); plt.show()
print("Noise blob present after opening:", opened[5:16,5:16].any())  # Should be False`,
    complexity:"Time O(m*n*k^2)  Space O(m*n)",
    followup:"What is the hat transform (tophat, blackhat) and when is it used? How does MORPH_GRADIENT detect edges from morphological operations?" },

  { id:"C08", difficulty:"Easy", tag:"Geometry", title:"Image Rotation Without Library",
    company:"Adobe, Apple", points:10,
    desc:`Rotate a grayscale image by an arbitrary angle (in degrees) around its centre using bilinear interpolation, without using cv2.warpAffine or any rotation library call.`,
    example:`Input: 100x100 image, angle=45 degrees
Output: rotated image with black fill for out-of-bounds pixels`,
    hint:"For each output pixel (x', y'), compute the corresponding source pixel (x,y) by applying inverse rotation around the centre. Use bilinear interpolation to sample the source.",
    fullSolution:`# DETAILED SOLUTION
# Inverse mapping: for each destination pixel, compute the source location.
# This avoids holes (which forward mapping can create).
# Bilinear interpolation: weighted average of 4 surrounding source pixels.

import numpy as np, cv2, urllib.request, matplotlib.pyplot as plt, math

def rotate_bilinear(img, angle_deg):
    h, w    = img.shape
    cx, cy  = w/2, h/2
    rad     = math.radians(-angle_deg)    # Negative: inverse rotation
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    
    out = np.zeros_like(img)
    for y_dst in range(h):
        for x_dst in range(w):
            # Translate to centre, rotate, translate back
            xc, yc  = x_dst - cx, y_dst - cy
            x_src   = cos_a*xc - sin_a*yc + cx
            y_src   = sin_a*xc + cos_a*yc + cy
            
            # Bilinear interpolation
            x0, y0 = int(x_src), int(y_src)
            x1, y1 = x0+1, y0+1
            if 0<=x0<w-1 and 0<=y0<h-1:
                dx, dy = x_src-x0, y_src-y0
                out[y_dst,x_dst] = int(
                    (1-dx)*(1-dy)*img[y0,x0] + dx*(1-dy)*img[y0,x1] +
                    (1-dx)*dy*img[y1,x0]     + dx*dy*img[y1,x1])
    return out

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw = urllib.request.urlopen(url).read()
gray= cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
gray= cv2.resize(gray,(120,90))   # small for manual speed

rot30  = rotate_bilinear(gray, 30)
rot90  = rotate_bilinear(gray, 90)

# Compare with cv2.warpAffine
M     = cv2.getRotationMatrix2D((gray.shape[1]//2,gray.shape[0]//2),30,1)
cv_rot= cv2.warpAffine(gray,M,(gray.shape[1],gray.shape[0]))

fig,axes=plt.subplots(1,4,figsize=(18,5))
for ax,im,t in zip(axes,[gray,rot30,rot90,cv_rot],
    ["Original","Manual 30deg","Manual 90deg","cv2 30deg (reference)"]):
    ax.imshow(im,cmap="gray"); ax.set_title(t); ax.axis("off")
plt.tight_layout(); plt.savefig("c08.png",dpi=150); plt.show()
print("Max diff vs cv2:", np.abs(rot30.astype(int)-cv_rot.astype(int)).max())`,
    complexity:"Time O(m*n)  Space O(m*n)",
    followup:"What are the artefacts of nearest-neighbour vs bilinear vs bicubic interpolation? When does bicubic produce ringing artefacts?" },

  { id:"C09", difficulty:"Easy", tag:"Statistics", title:"Compute PSNR and SSIM from Scratch",
    company:"Netflix, Qualcomm", points:10,
    desc:`Implement Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) from mathematical definitions, then apply them to compare a clean vs noisy image.`,
    example:`PSNR = 10*log10(255^2 / MSE).  SSIM = (2*mu1*mu2+c1)*(2*sigma12+c2) / ((mu1^2+mu2^2+c1)*(sigma1^2+sigma2^2+c2))`,
    hint:"PSNR: compute MSE first, then apply the log formula. SSIM: compute local means, variances, and covariances over an 11x11 Gaussian window.",
    fullSolution:`# DETAILED SOLUTION
# These metrics appear in virtually every image restoration paper.
# Understanding the formulas helps you know what you're optimising.

import numpy as np, cv2, urllib.request, matplotlib.pyplot as plt

def psnr_manual(ref, img, max_val=255.0):
    """PSNR in dB. Higher is better. Perfect: infinity. Typical: 30-45 dB."""
    mse = np.mean((ref.astype(float) - img.astype(float))**2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val**2 / mse)

def ssim_manual(ref, img, C1=(0.01*255)**2, C2=(0.03*255)**2):
    """Simplified global SSIM (skimage computes it locally with Gaussian windows)."""
    ref_f = ref.astype(float); img_f = img.astype(float)
    mu1, mu2   = ref_f.mean(), img_f.mean()
    s1_sq      = ref_f.var(); s2_sq = img_f.var()
    s12        = np.mean((ref_f-mu1)*(img_f-mu2))
    numerator  = (2*mu1*mu2+C1)*(2*s12+C2)
    denominator= (mu1**2+mu2**2+C1)*(s1_sq+s2_sq+C2)
    return numerator/denominator

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw = urllib.request.urlopen(url).read()
clean = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
rng   = np.random.default_rng(0)

print(f"{'Distortion':25s}  {'PSNR (dB)':>10s}  {'SSIM':>8s}")
for sigma in [10,30,50]:
    noisy = np.clip(clean.astype(float)+rng.normal(0,sigma,clean.shape),0,255).astype(np.uint8)
    p = psnr_manual(clean,noisy); s = ssim_manual(clean,noisy)
    print(f"  Gauss noise sigma={sigma:<3d}   {p:>10.2f}   {s:>8.4f}")
blurred = cv2.GaussianBlur(clean,(15,15),3)
print(f"  Gauss blur k=15         {psnr_manual(clean,blurred):>10.2f}   {ssim_manual(clean,blurred):>8.4f}")
jpeg_enc= cv2.imencode(".jpg",clean,[cv2.IMWRITE_JPEG_QUALITY,10])[1]
jpeg_dec= cv2.imdecode(jpeg_enc,cv2.IMREAD_GRAYSCALE)
print(f"  JPEG quality=10         {psnr_manual(clean,jpeg_dec):>10.2f}   {ssim_manual(clean,jpeg_dec):>8.4f}")`,
    complexity:"Time O(m*n)  Space O(1) for global SSIM",
    followup:"Why is LPIPS more correlated with human perception than PSNR? What does it mean when a method has high PSNR but low LPIPS?" },

  { id:"C10", difficulty:"Easy", tag:"Feature Matching", title:"Template Matching with Normalised Cross-Correlation",
    company:"Google Maps, automotive", points:10,
    desc:`Find the location of a small template image within a larger image using normalised cross-correlation (NCC), without using cv2.matchTemplate.`,
    example:`Input: 200x200 scene image + 30x30 template patch
Output: (row, col) location of best match`,
    hint:"Slide the template over the image. At each position, compute NCC = sum((I_patch - I_mean) * (T - T_mean)) / (std(I_patch) * std(T) * N). The location with the highest NCC is the match.",
    fullSolution:`# DETAILED SOLUTION
# Template matching with NCC is used in optical flow (Lucas-Kanade),
# visual tracking, panorama stitching, and stereo matching cost volumes.

import numpy as np, cv2, urllib.request, matplotlib.pyplot as plt

def ncc_match(image, template):
    """Slide template over image, return score map (NCC at each position)."""
    ih, iw = image.shape; th, tw = template.shape
    score   = np.zeros((ih-th+1, iw-tw+1))
    t_norm  = template - template.mean()
    t_std   = template.std() + 1e-8
    for i in range(ih-th+1):
        for j in range(iw-tw+1):
            patch  = image[i:i+th, j:j+tw].astype(float)
            p_norm = patch - patch.mean()
            p_std  = patch.std() + 1e-8
            score[i,j] = (p_norm * t_norm).sum() / (t_std * p_std * th * tw)
    return score

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw = urllib.request.urlopen(url).read()
gray= cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
gray= cv2.resize(gray,(200,150))   # shrink for speed

# Extract a template patch from a known location
ty,tx,th,tw = 30,50,35,35
template = gray[ty:ty+th, tx:tx+tw].copy()

# Add small noise to simulate real-world template variation
noisy_scene = gray.copy()
noisy_scene = np.clip(noisy_scene.astype(int)+np.random.randint(-10,10,gray.shape),0,255).astype(np.uint8)

score = ncc_match(noisy_scene.astype(float)/255.0, template.astype(float)/255.0)
best  = np.unravel_index(score.argmax(), score.shape)
print(f"True location: ({ty},{tx})  |  Found: {best}  |  Max NCC: {score.max():.4f}")

vis = cv2.cvtColor(noisy_scene, cv2.COLOR_GRAY2RGB)
cv2.rectangle(vis,(tx,ty),(tx+tw,ty+th),(0,255,0),2)      # True location (green)
cv2.rectangle(vis,(best[1],best[0]),(best[1]+tw,best[0]+th),(255,0,0),2)  # Found (red)

fig,(a,b,c)=plt.subplots(1,3,figsize=(15,5))
a.imshow(template,cmap="gray"); a.set_title("Template"); a.axis("off")
b.imshow(vis); b.set_title("Match: green=true, red=found"); b.axis("off")
c.imshow(score,cmap="hot"); c.set_title("NCC Score Map"); c.axis("off")
plt.tight_layout(); plt.savefig("c10.png",dpi=150); plt.show()`,
    complexity:"Time O((m-th)*(n-tw)*th*tw)  Space O((m-th)*(n-tw))",
    followup:"How does normalised cross-correlation compare to sum of squared differences (SSD)? Why does normalisation help?" },

  // ─── MEDIUM (11-60) ────────────────────────────────────────────
  { id:"C11", difficulty:"Medium", tag:"Dynamic Programming", title:"Non-Maximum Suppression and Soft-NMS",
    company:"NVIDIA, Amazon Rekognition", points:20,
    desc:`Implement standard NMS and Soft-NMS for object detection bounding boxes. Soft-NMS decays confidence instead of hard removal, improving recall on crowded scenes.`,
    example:`5 overlapping boxes with confidences [0.9,0.8,0.75,0.7,0.6].
After NMS (threshold=0.5): keep [0.9, 0.6] (separated).
After Soft-NMS: all boxes retained but with decayed confidences.`,
    hint:"NMS: sort by conf, greedily keep, remove all IoU>thresh. Soft-NMS: instead of removing, decay conf of overlapping boxes by a factor of exp(-IoU^2/sigma).",
    fullSolution:`# DETAILED SOLUTION
# NMS is used in every object detector after the final prediction head.
# Soft-NMS was introduced by Bodla et al. (2017) to handle occlusion.

import numpy as np, matplotlib.pyplot as plt, matplotlib.patches as patches

def iou(b1, b2):
    x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1])
    x2=min(b1[2],b2[2]); y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter+1e-8)

def hard_nms(boxes, scores, threshold=0.5):
    """Standard greedy NMS."""
    order = np.argsort(scores)[::-1]; keep=[]
    while len(order):
        i=order[0]; keep.append(i)
        if len(order)==1: break
        ious=np.array([iou(boxes[i],boxes[j]) for j in order[1:]])
        order=order[1:][ious<threshold]
    return keep

def soft_nms(boxes, scores, sigma=0.5, score_thresh=0.3):
    """Soft-NMS: decay scores instead of hard removal."""
    boxes  = [list(b) for b in boxes]
    scores = list(scores)
    N      = len(boxes); keep=[]
    for i in range(N):
        # Find max score box
        max_idx = max(range(len(scores)), key=lambda k: scores[k])
        if scores[max_idx] < score_thresh: break
        keep.append(max_idx)
        max_box = boxes[max_idx]
        # Decay all remaining boxes
        for j in range(len(scores)):
            if j != max_idx:
                ov = iou(max_box, boxes[j])
                scores[j] *= np.exp(-(ov**2)/sigma)
        scores[max_idx] = 0   # Prevent re-selection
    return keep

# Simulate 8 boxes (2 clusters + 1 isolated)
boxes = np.array([[50,50,200,200],[60,55,210,205],[55,52,195,198],
                  [58,58,202,200],[250,50,400,180],[260,55,410,185],
                  [252,52,398,178],[500,150,600,250]])
scores= np.array([0.95,0.82,0.76,0.70, 0.88,0.74,0.61, 0.92])

keep_hard = hard_nms(boxes,scores,0.5)
keep_soft = soft_nms(boxes,scores,sigma=0.5,score_thresh=0.3)

print(f"Hard NMS kept {len(keep_hard)} boxes: indices {keep_hard}")
print(f"Soft NMS kept {len(keep_soft)} boxes: indices {keep_soft}")

fig,(a,b)=plt.subplots(1,2,figsize=(14,6))
for ax,kept,title in zip([a,b],[keep_hard,keep_soft],["Hard NMS","Soft NMS"]):
    ax.set_xlim(0,650); ax.set_ylim(0,280); ax.invert_yaxis()
    ax.set_facecolor("#0d1529"); ax.set_title(title,color="white")
    for i,(bx,sc) in enumerate(zip(boxes,scores)):
        col="lime" if i in kept else "gray"
        r=patches.Rectangle((bx[0],bx[1]),bx[2]-bx[0],bx[3]-bx[1],
                             linewidth=2,edgecolor=col,facecolor="none")
        ax.add_patch(r)
        ax.text(bx[0],bx[1]-3,f"{sc:.2f}",color=col,fontsize=8,fontweight="bold")
plt.tight_layout(); plt.savefig("c11.png",dpi=150); plt.show()`,
    complexity:"Hard NMS O(n^2)  Soft-NMS O(n^2)  Space O(n)",
    followup:"When does Soft-NMS substantially outperform hard NMS? (Crowded pedestrian scenes, overlapping objects with similar class.) What is WBF (Weighted Boxes Fusion)?" },

  { id:"C12", difficulty:"Medium", tag:"Thresholding", title:"Otsu's Global Threshold from Scratch",
    company:"OpenCV, Mathworks", points:20,
    desc:`Implement Otsu's optimal global threshold algorithm. Find the threshold t* that minimises the within-class variance (equivalently maximises the between-class variance) of pixel intensities.`,
    example:`Input: bimodal grayscale image
Output: t* = optimal threshold value + binary image`,
    hint:"For each candidate threshold t, compute the probability (weight) and mean of each class (foreground, background). Between-class variance = w0*w1*(mu0-mu1)^2. Choose t that maximises this.",
    fullSolution:`# DETAILED SOLUTION
# Otsu's method (1979) is used in document binarisation, medical imaging,
# and any segmentation task requiring automatic thresholding.
# OpenCV's THRESH_OTSU calls exactly this algorithm.

import numpy as np, cv2, urllib.request, matplotlib.pyplot as plt

def otsu_threshold(gray):
    """Returns optimal threshold that maximises between-class variance."""
    hist = np.bincount(gray.flatten(), minlength=256).astype(float)
    total = gray.size
    prob  = hist / total          # Probability of each intensity

    best_t    = 0
    best_var  = 0
    w0, mu0   = 0.0, 0.0

    for t in range(256):
        w0 += prob[t]
        w1  = 1.0 - w0
        if w0 == 0 or w1 == 0: continue
        mu0 = mu0 + t * prob[t]          # Running weighted sum for class 0
        mu1 = (np.arange(256)*prob).sum() - mu0   # Remaining mean (class 1)
        mean0 = mu0 / (w0 + 1e-8)
        mean1 = mu1 / (w1 + 1e-8)
        # Between-class variance
        var_between = w0 * w1 * (mean0 - mean1)**2
        if var_between > best_var:
            best_var = var_between
            best_t   = t

    return best_t

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw = urllib.request.urlopen(url).read()
gray= cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)

t_manual = otsu_threshold(gray)
_, t_cv   = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(f"Manual Otsu threshold: {t_manual}")
print(f"OpenCV Otsu threshold: {t_cv}")

binary_manual = (gray > t_manual).astype(np.uint8)*255
binary_cv, _  = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU), None

fig,(a,b,c,d)=plt.subplots(1,4,figsize=(20,5))
a.imshow(gray,cmap="gray"); a.set_title("Grayscale"); a.axis("off")
a.axvline(x=0,color="none")
b.hist(gray.flatten(),bins=128,color=P.accent1)
b.axvline(x=t_manual,color="red",linewidth=2,label=f"t*={t_manual}")
b.legend(); b.set_title("Histogram + Otsu Threshold")
c.imshow(binary_manual,cmap="gray"); c.set_title(f"Binary (t={t_manual})"); c.axis("off")
d.imshow(cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2),cmap="gray")
d.set_title("Adaptive Threshold (comparison)"); d.axis("off")
plt.tight_layout(); plt.savefig("c12.png",dpi=150); plt.show()`,
    complexity:"Time O(m*n + 256)  Space O(256)",
    followup:"When does Otsu fail? (Multimodal histograms, heavily skewed distributions.) How does adaptive thresholding (Sauvola, Niblack) handle non-uniform illumination?" },

  { id:"C13", difficulty:"Medium", tag:"Dynamic Programming", title:"Seam Carving for Content-Aware Resizing",
    company:"Adobe (classic interview)", points:20,
    desc:`Implement seam carving to remove one vertical seam from an image. A vertical seam is a connected path of pixels from top to bottom, one pixel per row, that minimises the total energy (gradient magnitude). This allows content-aware resizing that avoids cutting through important objects.`,
    example:`Input: 320x240 colour image
Output: 319x240 image with one low-energy seam removed`,
    hint:"1. Compute energy map (gradient magnitude). 2. Dynamic programming: dp[i][j] = energy[i][j] + min(dp[i-1][j-1], dp[i-1][j], dp[i-1][j+1]). 3. Backtrack from minimum in last row. 4. Remove seam pixels.",
    fullSolution:`# DETAILED SOLUTION
# Seam carving (Avidan & Shamir, SIGGRAPH 2007) resizes images while
# preserving important content. This is in Photoshop as "Content-Aware Scale".

import numpy as np, cv2, urllib.request, matplotlib.pyplot as plt

def compute_energy(img):
    """Gradient magnitude energy: high at edges, low in smooth regions."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(float)
    gx   = np.gradient(gray, axis=1)
    gy   = np.gradient(gray, axis=0)
    return np.sqrt(gx**2 + gy**2)

def find_seam(energy):
    """DP to find minimum-energy vertical seam."""
    h, w   = energy.shape
    dp     = energy.copy()
    back   = np.zeros((h,w), int)   # Backtrack direction: -1, 0, 1
    for i in range(1, h):
        for j in range(w):
            lo, hi = max(0,j-1), min(w-1,j+1)
            prev   = dp[i-1, lo:hi+1]
            min_j  = lo + prev.argmin()
            dp[i,j]   = energy[i,j] + dp[i-1,min_j]
            back[i,j] = min_j - j   # -1, 0, or 1
    # Backtrack
    seam = np.zeros(h, int)
    seam[-1] = dp[-1].argmin()
    for i in range(h-2,-1,-1):
        seam[i] = seam[i+1] + back[i+1, seam[i+1]]
    return seam

def remove_seam(img, seam):
    """Remove seam pixels from image."""
    h, w = img.shape[:2]
    out  = np.zeros((h, w-1, img.shape[2]), img.dtype)
    for i in range(h):
        out[i,:seam[i]] = img[i,:seam[i]]
        out[i,seam[i]:] = img[i,seam[i]+1:]
    return out

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw = urllib.request.urlopen(url).read()
img = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

energy = compute_energy(img)
seam   = find_seam(energy)

# Visualise seam on image
vis = img.copy()
for i,j in enumerate(seam):
    vis[i,max(0,j-1):min(img.shape[1],j+2)] = [255,0,0]

carved = remove_seam(img, seam)
print(f"Original: {img.shape}  ->  Carved: {carved.shape}")

fig,(a,b,c,d)=plt.subplots(1,4,figsize=(20,5))
a.imshow(img); a.set_title(f"Original {img.shape[1]}x{img.shape[0]}"); a.axis("off")
b.imshow(energy,cmap="hot"); b.set_title("Energy Map"); b.axis("off")
c.imshow(vis); c.set_title("Seam Highlighted (red)"); c.axis("off")
d.imshow(carved); d.set_title(f"After 1 Seam Removal {carved.shape[1]}x{carved.shape[0]}"); d.axis("off")
plt.tight_layout(); plt.savefig("c13.png",dpi=150); plt.show()`,
    complexity:"Time O(m*n) per seam  Space O(m*n)",
    followup:"How would you remove k seams efficiently? (Recompute energy only in the local neighbourhood of the removed seam.) What energy function could protect faces from removal?" },

  { id:"C14", difficulty:"Medium", tag:"Hashing", title:"Perceptual Hash (pHash) for Near-Duplicate Detection",
    company:"Google Photos, Shutterstock", points:20,
    desc:`Implement perceptual hashing (pHash) to detect near-duplicate images. The hash should be similar for images that are slightly cropped, resized, or colour-adjusted, and very different for unrelated images.`,
    example:`Input: original image vs slightly brightened copy vs completely different image
Output: Hamming distances: 0-5 (near-duplicate), >10 (different)`,
    hint:"1. Resize to 32x32 grayscale. 2. Apply DCT (discrete cosine transform). 3. Take top-left 8x8 coefficients. 4. Threshold at the mean. 5. Read bits row by row to get 64-bit hash.",
    fullSolution:`# DETAILED SOLUTION
# Perceptual hashing is used for copyright detection, deduplication,
# and reverse image search. Hamming distance of hashes gives similarity.

import numpy as np, cv2, urllib.request, io
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fft import dct

def phash(img_arr, hash_size=8, highfreq_factor=4):
    """Perceptual hash: DCT-based 64-bit hash."""
    img_size = hash_size * highfreq_factor   # 32
    gray     = cv2.cvtColor(img_arr,cv2.COLOR_RGB2GRAY) if img_arr.ndim==3 else img_arr
    resized  = cv2.resize(gray,(img_size,img_size),interpolation=cv2.INTER_AREA).astype(float)
    # 2D DCT via separable 1D DCTs
    dct_2d   = dct(dct(resized.T, norm='ortho').T, norm='ortho')
    # Take top-left hash_size x hash_size (low-frequency components)
    dct_low  = dct_2d[:hash_size,:hash_size]
    avg      = dct_low.mean()
    bits     = (dct_low > avg).flatten()  # 64 bits
    return bits

def hamming(h1, h2):
    return np.sum(h1 != h2)

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
raw = urllib.request.urlopen(url).read()
img_orig = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))

# Create near-duplicate variants
img_bright = np.clip(img_orig.astype(int)+30,0,255).astype(np.uint8)
img_crop   = img_orig[10:img_orig.shape[0]-10, 10:img_orig.shape[1]-10]  # slight crop
img_flip   = img_orig[:,::-1,:]                                            # horizontal flip

url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw2 = urllib.request.urlopen(url2).read()
img_diff = np.array(Image.open(io.BytesIO(raw2)).convert("RGB"))

h_orig   = phash(img_orig)
variants = [("Brightened +30", img_bright), ("Cropped 10px",img_crop),
            ("Flipped H",img_flip), ("Different image",img_diff)]
print(f"{'Variant':25s}  {'Hamming':>7s}  {'Verdict':>12s}")
for name,img in variants:
    d = hamming(h_orig, phash(img))
    verdict = "Near-duplicate" if d <= 10 else "Different"
    print(f"  {name:22s}  {d:>7d}  {verdict:>12s}")`,
    complexity:"Time O(N^2 log N) for DCT on N x N image  Space O(N^2)",
    followup:"How does dHash (difference hash) differ from pHash? How would you build a large-scale near-duplicate detection system using pHash + LSH (Locality Sensitive Hashing)?" },

  { id:"C15", difficulty:"Medium", tag:"Deep Learning", title:"Conv2D Forward Pass from Scratch",
    company:"FAANG system design", points:20,
    desc:`Implement a Conv2D forward pass (single input, single output channel) using only NumPy. Validate against PyTorch's F.conv2d with the same kernel weights.`,
    example:`Input: 1x1x5x5 tensor, kernel: 1x1x3x3, padding=1, stride=1
Output: 1x1x5x5 feature map matching PyTorch exactly`,
    hint:"The output spatial size: H_out = (H_in + 2*padding - kernel_h) // stride + 1. Use nested loops or reshape the input into overlapping patches (im2col trick).",
    fullSolution:`# DETAILED SOLUTION
# Understanding the Conv2D forward pass is fundamental.
# This is what every CNN layer computes at its core.
# The im2col trick converts convolution to a single matrix multiply.

import numpy as np, torch, torch.nn.functional as F, matplotlib.pyplot as plt

def conv2d_numpy(x, w, bias=None, stride=1, padding=0):
    """
    x: (N,C_in,H,W), w: (C_out,C_in,kH,kW)
    Returns: (N,C_out,H_out,W_out)
    """
    N,C_in,H,W     = x.shape
    C_out,_,kH,kW  = w.shape
    H_out = (H + 2*padding - kH)//stride + 1
    W_out = (W + 2*padding - kW)//stride + 1
    
    # Pad input
    x_pad = np.pad(x,((0,0),(0,0),(padding,padding),(padding,padding)),mode='constant')
    out   = np.zeros((N,C_out,H_out,W_out))
    
    for n in range(N):
        for co in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    r, c = i*stride, j*stride
                    patch = x_pad[n,:,r:r+kH,c:c+kW]   # (C_in,kH,kW)
                    out[n,co,i,j] = (patch * w[co]).sum()
            if bias is not None:
                out[n,co] += bias[co]
    return out

# Test against PyTorch
np.random.seed(42)
x_np = np.random.randn(1,3,8,8).astype(np.float32)
w_np = np.random.randn(8,3,3,3).astype(np.float32)
b_np = np.random.randn(8).astype(np.float32)

out_np = conv2d_numpy(x_np, w_np, b_np, stride=1, padding=1)
out_pt = F.conv2d(torch.tensor(x_np), torch.tensor(w_np),
                  torch.tensor(b_np), stride=1, padding=1).numpy()

print(f"NumPy output shape:   {out_np.shape}")
print(f"PyTorch output shape: {out_pt.shape}")
print(f"Max absolute diff:    {np.abs(out_np-out_pt).max():.2e}")
print("Outputs match:", np.allclose(out_np, out_pt, atol=1e-5))

# Visualise feature maps from first 8 filters
fig,axes=plt.subplots(2,4,figsize=(16,8))
for i,ax in enumerate(axes.flat):
    ax.imshow(out_np[0,i],cmap="RdBu_r"); ax.set_title(f"Filter {i}"); ax.axis("off")
plt.suptitle("Conv2D Output Feature Maps (8 filters, 3x3, padding=1)",fontsize=13)
plt.tight_layout(); plt.savefig("c15.png",dpi=150); plt.show()`,
    complexity:"Time O(N*C_out*C_in*H_out*W_out*kH*kW)  optimised via FFT or GEMM",
    followup:"What is the im2col transformation and how does it convert convolution to a GEMM (General Matrix-Matrix Multiply)? Why do GPU BLAS libraries make this fast?" },

  { id:"C16", difficulty:"Medium", tag:"Gradient / Optimisation", title:"Gradient Descent for Image Denoising (TV Regularisation)",
    company:"Research / Adobe", points:20,
    desc:`Implement iterative Total Variation (TV) denoising using gradient descent. TV regularisation penalises large gradients while preserving sharp edges.`,
    example:`Input: noisy image. After 100 iterations of gradient descent on L2 + lambda*TV(x), output a smoother image with preserved edges.`,
    hint:"TV(x) = sum |grad_x| + |grad_y|. Gradient of TV w.r.t. x[i,j] involves finite difference quotients. Data term gradient: 2*(x-y). Total: grad = 2*(x-y) + lambda * grad_TV.",
    fullSolution:`# DETAILED SOLUTION
# Total Variation was foundational in image restoration (Rudin-Osher-Fatemi, 1992).
# Modern deep denoisers (DnCNN, FFDNet) supersede it but TV regularisation
# is still used in compressed sensing MRI, CT reconstruction, and edge-aware smoothing.

import numpy as np, cv2, urllib.request, matplotlib.pyplot as plt

def tv_gradient(x, eps=1e-4):
    """Isotropic TV gradient (smooth approximation to |grad|)."""
    dx = np.diff(x, axis=1, append=x[:,-1:])   # Right difference
    dy = np.diff(x, axis=0, append=x[-1:,:])   # Down difference
    norm = np.sqrt(dx**2 + dy**2 + eps)         # Smooth abs
    # Divergence of (dx/norm, dy/norm)
    gx = dx/norm
    gy = dy/norm
    div_x = np.diff(gx, axis=1, prepend=gx[:,:1])
    div_y = np.diff(gy, axis=0, prepend=gy[:1,:])
    return -(div_x + div_y)   # Negative divergence

def tv_denoise(noisy, lam=0.1, lr=0.05, n_iters=150):
    x = noisy.astype(float)/255.0
    y = x.copy()
    losses=[]
    for it in range(n_iters):
        data_grad = 2*(x-y)
        tv_grad   = tv_gradient(x)
        grad      = data_grad + lam*tv_grad
        x        -= lr * grad
        x         = x.clip(0,1)
        if it%10==0:
            loss = np.mean((x-y)**2) + lam*np.sum(np.sqrt(
                np.gradient(x,axis=0)**2 + np.gradient(x,axis=1)**2))
            losses.append(loss)
    return (x*255).astype(np.uint8), losses

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw = urllib.request.urlopen(url).read()
clean = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
clean = cv2.resize(clean,(160,120))
rng   = np.random.default_rng(0)
noisy = np.clip(clean.astype(int)+rng.normal(0,30,clean.shape),0,255).astype(np.uint8)

denoised, losses = tv_denoise(noisy, lam=0.15, n_iters=150)

from skimage.metrics import peak_signal_noise_ratio as psnr
print(f"PSNR noisy:    {psnr(clean,noisy,data_range=255):.2f} dB")
print(f"PSNR denoised: {psnr(clean,denoised,data_range=255):.2f} dB")

fig,(a,b,c,d)=plt.subplots(1,4,figsize=(20,5))
a.imshow(clean,cmap="gray"); a.set_title("Clean"); a.axis("off")
b.imshow(noisy,cmap="gray"); b.set_title("Noisy"); b.axis("off")
c.imshow(denoised,cmap="gray"); c.set_title("TV Denoised"); c.axis("off")
d.plot(losses,color=P.accent3); d.set_title("Loss Curve"); d.set_xlabel("Iteration x10")
d.set_facecolor("#0d1529")
plt.tight_layout(); plt.savefig("c16.png",dpi=150); plt.show()`,
    complexity:"Time O(n_iters * m*n)  Space O(m*n)",
    followup:"Compare TV denoising with Gaussian blur: TV preserves sharp edges while Gaussian blurs them. What is the split-Bregman / ADMM algorithm for fast TV minimisation?" },

  { id:"C17", difficulty:"Medium", tag:"Neural Network", title:"Backpropagation Through a Conv Layer",
    company:"Deep Learning interviews", points:20,
    desc:`Implement the backward pass of a single Conv2D layer from scratch. Compute gradients with respect to the input, weights, and bias, and verify against PyTorch autograd.`,
    example:`Given dL/dOutput (output gradient), compute dL/dInput, dL/dWeights, dL/dBias.`,
    hint:"dL/dInput uses full convolution (transpose convolution with flipped kernel). dL/dWeights uses cross-correlation between input patches and output gradients. dL/dBias = sum of output gradients.",
    fullSolution:`# DETAILED SOLUTION
# Backprop through conv is the heart of CNN training.
# dL/dWeights = correlate(input, dL/dOutput)
# dL/dInput   = full_conv(dL/dOutput, flipped_weights)  i.e. transposed conv

import numpy as np, torch, torch.nn.functional as F

def conv2d_forward(x, w, padding=0):
    N,C_in,H,W = x.shape; C_out,_,kH,kW = w.shape
    H_out=(H+2*padding-kH)+1; W_out=(W+2*padding-kW)+1
    x_pad = np.pad(x,((0,0),(0,0),(padding,padding),(padding,padding)))
    out   = np.zeros((N,C_out,H_out,W_out))
    for n in range(N):
        for co in range(C_out):
            for ci in range(C_in):
                for i in range(H_out):
                    for j in range(W_out):
                        out[n,co,i,j]+=np.sum(x_pad[n,ci,i:i+kH,j:j+kW]*w[co,ci])
    return out

def conv2d_backward(dout, x, w, padding=0):
    N,C_in,H,W = x.shape; C_out,_,kH,kW = w.shape
    _,_,H_out,W_out = dout.shape
    x_pad  = np.pad(x,((0,0),(0,0),(padding,padding),(padding,padding)))
    dw     = np.zeros_like(w)
    db     = dout.sum(axis=(0,2,3))     # Sum over N,H,W
    dx_pad = np.zeros_like(x_pad)

    for n in range(N):
        for co in range(C_out):
            for ci in range(C_in):
                for i in range(H_out):
                    for j in range(W_out):
                        patch = x_pad[n,ci,i:i+kH,j:j+kW]
                        dw[co,ci] += dout[n,co,i,j]*patch          # dL/dW
                        dx_pad[n,ci,i:i+kH,j:j+kW]+=dout[n,co,i,j]*w[co,ci]  # dL/dX
    dx = dx_pad[:,:,padding:padding+H,padding:padding+W] if padding>0 else dx_pad
    return dx, dw, db

# Test vs PyTorch autograd
np.random.seed(1)
x_np = np.random.randn(1,2,5,5).astype(np.float32)
w_np = np.random.randn(3,2,3,3).astype(np.float32)
dout_np= np.random.randn(1,3,3,3).astype(np.float32)   # H_out=W_out=3

dx_np,dw_np,db_np = conv2d_backward(dout_np, x_np, w_np, padding=0)

x_pt  = torch.tensor(x_np,  requires_grad=True)
w_pt  = torch.tensor(w_np,  requires_grad=True)
b_pt  = torch.zeros(3,      requires_grad=True)
out_pt= F.conv2d(x_pt, w_pt, b_pt, padding=0)
out_pt.backward(torch.tensor(dout_np))

print(f"dX max diff: {np.abs(dx_np - x_pt.grad.numpy()).max():.2e}")
print(f"dW max diff: {np.abs(dw_np - w_pt.grad.numpy()).max():.2e}")
print(f"db max diff: {np.abs(db_np - b_pt.grad.numpy()).max():.2e}")
print("All match PyTorch!", np.allclose(dx_np,x_pt.grad.numpy(),atol=1e-5) and
      np.allclose(dw_np,w_pt.grad.numpy(),atol=1e-5))`,
    complexity:"Time O(N*C_out*C_in*H*W*kH*kW) per backward  Space O(m*n)",
    followup:"What is the Winograd algorithm that speeds up 3x3 convolution? How does FFT-based convolution achieve O(m*n*log(m*n)) complexity?" },

  { id:"C18", difficulty:"Medium", tag:"BFS / Shortest Path", title:"Shortest Safe Path Through an Image Grid",
    company:"Robotics, Google Maps", points:20,
    desc:`Given a binary overhead map (0=free, 1=obstacle), find the shortest path from the top-left to the bottom-right corner using BFS. If no path exists, return -1. This models robot path planning on an occupancy grid.`,
    example:`Grid:
  0 0 1 0
  1 0 0 0
  0 0 1 0
  0 0 0 0
Shortest path length: 7 steps`,
    hint:"BFS on 4-connected or 8-connected grid. Start from (0,0), target (H-1,W-1). Track visited cells. Each step has cost 1.",
    fullSolution:`# DETAILED SOLUTION
# BFS gives shortest path in unweighted graphs.
# This problem models robot navigation, game pathfinding (A*), and route planning.
# In CV: used for finding paths in skeleton images, maze solving, watershed preprocessing.

from collections import deque
import numpy as np, cv2, urllib.request, matplotlib.pyplot as plt

def bfs_path(grid):
    """BFS shortest path from (0,0) to (H-1,W-1) through free cells (0).
    Returns (path, distance) or (None,-1) if no path."""
    H, W = len(grid), len(grid[0])
    if grid[0][0]==1 or grid[H-1][W-1]==1: return None,-1
    
    visited = [[False]*W for _ in range(H)]
    parent  = [[None]*W  for _ in range(H)]
    q = deque([(0,0)]); visited[0][0]=True
    
    while q:
        r,c = q.popleft()
        if r==H-1 and c==W-1:
            # Reconstruct path
            path=[]; cur=(r,c)
            while cur: path.append(cur); cur=parent[cur[0]][cur[1]]
            return path[::-1], len(path)-1
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:  # 4-connectivity
            nr,nc=r+dr,c+dc
            if 0<=nr<H and 0<=nc<W and grid[nr][nc]==0 and not visited[nr][nc]:
                visited[nr][nc]=True; parent[nr][nc]=(r,c); q.append((nr,nc))
    return None,-1

# Test
grid1 = [[0,0,1,0],[1,0,0,0],[0,0,1,0],[0,0,0,0]]
path,dist = bfs_path(grid1)
print(f"Path length: {dist}")
print(f"Path: {path}")

# Visualise on a random map
np.random.seed(7)
H,W = 30,40
occ = (np.random.rand(H,W) < 0.3).astype(int)   # 30% obstacles
occ[0,0]=0; occ[H-1,W-1]=0

path,dist = bfs_path(occ.tolist())
vis = np.stack([occ*200,(1-occ)*50,np.zeros((H,W))],axis=-1).astype(np.uint8)   # red=obstacle
if path:
    for r,c in path: vis[r,c]=[0,255,100]   # green path
    vis[0,0]=[0,100,255]; vis[H-1,W-1]=[255,200,0]  # start=blue, end=yellow
    print(f"Path found! Length = {dist} steps")
else:
    print("No path found!")

plt.figure(figsize=(12,9))
plt.imshow(vis); plt.title(f"BFS Shortest Path (green), Length={dist}"); plt.axis("off")
plt.savefig("c18.png",dpi=150); plt.show()`,
    complexity:"Time O(H*W)  Space O(H*W)",
    followup:"When would A* (A-star with heuristic) outperform BFS? (Large grids with diagonal movement.) How do occupancy grids relate to simultaneous localisation and mapping (SLAM)?" },

  { id:"C19", difficulty:"Medium", tag:"Attention / Transformer", title:"Scaled Dot-Product Attention from Scratch",
    company:"DeepMind, OpenAI, Anthropic", points:20,
    desc:`Implement scaled dot-product attention from scratch using only NumPy. Validate against PyTorch's F.scaled_dot_product_attention. This is the core operation of every Vision Transformer.`,
    example:`Input: Q (N,S,d_k), K (N,S,d_k), V (N,S,d_v)
Output: Attention output (N,S,d_v)`,
    hint:"Attention(Q,K,V) = softmax(Q @ K.T / sqrt(d_k)) @ V. Apply softmax row-wise along the key dimension.",
    fullSolution:`# DETAILED SOLUTION
# Scaled dot-product attention (Vaswani et al., 2017) is in every transformer.
# In ViT, each image patch is a token; attention computes global feature relationships.

import numpy as np, torch, torch.nn.functional as F, matplotlib.pyplot as plt

def softmax(x, axis=-1):
    x   = x - x.max(axis=axis, keepdims=True)   # Numerical stability
    ex  = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (B,H,S,d_k), K: (B,H,S,d_k), V: (B,H,S,d_v)
    Returns: output (B,H,S,d_v), attn_weights (B,H,S,S)
    """
    d_k     = Q.shape[-1]
    scores  = Q @ K.swapaxes(-2,-1) / np.sqrt(d_k)   # (B,H,S,S)
    if mask is not None:
        scores = np.where(mask==0, -1e9, scores)
    weights = softmax(scores, axis=-1)               # (B,H,S,S)
    out     = weights @ V                            # (B,H,S,d_v)
    return out, weights

# Test
np.random.seed(0)
B,H,S,d_k,d_v = 2,4,8,16,16
Q_np = np.random.randn(B,H,S,d_k).astype(np.float32)
K_np = np.random.randn(B,H,S,d_k).astype(np.float32)
V_np = np.random.randn(B,H,S,d_v).astype(np.float32)

out_np,w_np = scaled_dot_product_attention(Q_np,K_np,V_np)
out_pt      = F.scaled_dot_product_attention(torch.tensor(Q_np),torch.tensor(K_np),torch.tensor(V_np))

print(f"NumPy output shape:  {out_np.shape}")
print(f"PyTorch output shape:{out_pt.shape}")
print(f"Max diff:            {np.abs(out_np-out_pt.numpy()).max():.2e}")

# Visualise attention weights (head 0, batch 0)
fig,axes=plt.subplots(1,H,figsize=(20,4))
for h,ax in enumerate(axes):
    im=ax.imshow(w_np[0,h],cmap="viridis",vmin=0,vmax=0.3)
    ax.set_title(f"Head {h}"); ax.set_xlabel("Key"); ax.set_ylabel("Query")
plt.suptitle("Attention Weight Maps (random Q,K,V)",fontsize=13)
plt.tight_layout(); plt.savefig("c19.png",dpi=150); plt.show()`,
    complexity:"Time O(B*H*S^2*d_k)  Space O(B*H*S^2)",
    followup:"Why does the scaling by sqrt(d_k) prevent gradient vanishing? How does multi-head attention learn different relationship types per head?" },

  { id:"C20", difficulty:"Medium", tag:"Image Processing", title:"Harris Corner Detector from Scratch",
    company:"SLAM, Robotics, Google Maps", points:20,
    desc:`Implement the Harris corner detector from scratch. A corner is a point where the image has large intensity variation in multiple directions (not just edges).`,
    example:`Input: image of a building or chessboard
Output: corner response map, with strong responses at corners`,
    hint:"1. Compute Ix, Iy (image gradients). 2. Compute structure tensor M = [[Ix^2, IxIy],[IxIy, Iy^2]] smoothed with a Gaussian. 3. Corner response R = det(M) - k*trace(M)^2 (k ~ 0.04-0.06).",
    fullSolution:`# DETAILED SOLUTION
# Harris corners (Harris & Stephens, 1988) are used for feature matching,
# visual SLAM, panorama stitching, and as keypoints for descriptors like SIFT.

import numpy as np, cv2, urllib.request, matplotlib.pyplot as plt

def harris_corner(img_gray, k=0.04, sigma=1.5, threshold=0.01):
    img = img_gray.astype(np.float64)
    # Compute image gradients with Sobel
    Ix = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    Iy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    # Structure tensor components (Gaussian-smoothed products)
    Ixx = cv2.GaussianBlur(Ix*Ix,(5,5),sigma)
    Ixy = cv2.GaussianBlur(Ix*Iy,(5,5),sigma)
    Iyy = cv2.GaussianBlur(Iy*Iy,(5,5),sigma)
    # Corner response
    det  = Ixx*Iyy - Ixy**2
    tr   = Ixx + Iyy
    R    = det - k * tr**2
    # Normalise and threshold
    R_norm = (R - R.min())/(R.max()-R.min()+1e-8)
    corners = (R_norm > threshold).astype(np.uint8)
    return R_norm, corners

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw = urllib.request.urlopen(url).read()
img = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

R, corners = harris_corner(gray)
# Non-maximum suppression: only keep local maxima
corners_nm = cv2.dilate(R.astype(np.float32),None)
pts = np.argwhere((R==corners_nm) & (R>0.01))   # (row,col) of corners
print(f"Detected {len(pts)} Harris corners")

vis = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
for (r,c) in pts: cv2.circle(vis,(c,r),3,(255,0,0),-1)

# Compare with OpenCV Harris
dst = cv2.cornerHarris(gray.astype(np.float32),2,3,0.04)
dst_norm = (dst-dst.min())/(dst.max()-dst.min()+1e-8)

fig,(a,b,c,d)=plt.subplots(1,4,figsize=(20,5))
a.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)); a.set_title("Original"); a.axis("off")
b.imshow(R,cmap="hot"); b.set_title("Harris Response Map"); b.axis("off")
c.imshow(vis); c.set_title(f"Corners Overlaid ({len(pts)})"); c.axis("off")
d.imshow(dst_norm,cmap="hot"); d.set_title("OpenCV Harris (reference)"); d.axis("off")
plt.tight_layout(); plt.savefig("c20.png",dpi=150); plt.show()`,
    complexity:"Time O(m*n)  Space O(m*n)",
    followup:"How does FAST (Features from Accelerated Segment Test) achieve real-time corner detection? What makes Shi-Tomasi corners ('Good Features to Track') different from Harris?" },

  // ─── HARD (21-40 of medium, then hard) ──────────────────────────
  { id:"C21", difficulty:"Medium", tag:"Segmentation", title:"K-Means Image Segmentation",
    company:"Adobe, NVIDIA", points:20,
    desc:`Segment an image into k colour clusters using k-means clustering from scratch. Each pixel is assigned to the nearest cluster centre in RGB space, then replaced by the cluster's mean colour.`,
    example:`Input: colour photo, k=5
Output: image posterised into 5 dominant colour regions`,
    hint:"Initialise k cluster centres (random or k-means++). Assign each pixel to nearest centre (L2 distance). Update centres to mean of assigned pixels. Repeat until convergence.",
    fullSolution:`# DETAILED SOLUTION
# K-means colour segmentation is used for colour quantisation (reducing palette),
# image compression, and as a preprocessing step for more sophisticated segmentation.

import numpy as np, cv2, urllib.request, matplotlib.pyplot as plt
from sklearn.cluster import KMeans   # For validation

def kmeans_segment(img_rgb, k=5, n_iters=20, tol=1.0):
    h, w, c = img_rgb.shape
    pixels  = img_rgb.reshape(-1, c).astype(float)
    # k-means++ init: first centre random, then choose next proportional to distance
    rng = np.random.default_rng(42)
    centres = [pixels[rng.integers(len(pixels))]]
    for _ in range(k-1):
        dists = np.min([np.sum((pixels-ctr)**2,axis=1) for ctr in centres],axis=0)
        probs = dists / dists.sum()
        centres.append(pixels[rng.choice(len(pixels),p=probs)])
    centres = np.array(centres)

    for it in range(n_iters):
        # Assignment step
        dists   = np.array([np.sum((pixels-ctr)**2,axis=1) for ctr in centres])
        labels  = dists.argmin(axis=0)
        # Update step
        new_ctrs= np.array([pixels[labels==ki].mean(axis=0) if (labels==ki).any() else centres[ki]
                             for ki in range(k)])
        if np.max(np.abs(new_ctrs-centres)) < tol: break
        centres = new_ctrs

    segmented = centres[labels].reshape(h,w,c).astype(np.uint8)
    return segmented, labels.reshape(h,w), centres

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
raw = urllib.request.urlopen(url).read()
img = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

fig,axes=plt.subplots(1,4,figsize=(20,5))
axes[0].imshow(img_rgb); axes[0].set_title("Original"); axes[0].axis("off")
for i,k in enumerate([3,5,8],1):
    seg,_,ctrs = kmeans_segment(img_rgb,k=k)
    axes[i].imshow(seg); axes[i].set_title(f"k-Means k={k}"); axes[i].axis("off")
plt.tight_layout(); plt.savefig("c21.png",dpi=150); plt.show()`,
    complexity:"Time O(k*n*n_iters)  Space O(n)",
    followup:"What are the limitations of k-means for image segmentation compared to GrabCut or deep segmentation models? When is SLIC superpixel segmentation preferred?" },

  { id:"C22", difficulty:"Medium", tag:"Convolution", title:"Implement Depthwise Separable Convolution",
    company:"Google (MobileNet), Apple", points:20,
    desc:`Implement depthwise separable convolution: first a depthwise convolution (one filter per input channel), then a pointwise (1x1) convolution to mix channels. Measure parameter and FLOPs reduction vs standard Conv2D.`,
    example:`Standard Conv2D: C_in=32, C_out=64, k=3x3 -> 32*64*9 = 18,432 parameters
Depthwise Sep: (32*9) + (32*64) = 288 + 2048 = 2,336 parameters (8x fewer)`,
    hint:"Depthwise: apply one k x k filter per input channel independently (groups=C_in in PyTorch). Pointwise: apply 1x1 conv to mix channels across all C_in.",
    fullSolution:`# DETAILED SOLUTION
# Depthwise separable convolution (Chollet 2017, Howard et al. MobileNet 2017)
# reduces computation by 8-9x for 3x3 kernels while maintaining accuracy.
# Used in MobileNet, EfficientNet, Xception, and many mobile-efficient architectures.

import numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt

def dw_sep_conv_numpy(x, dw_kernel, pw_kernel):
    """
    x:         (N, C_in, H, W)
    dw_kernel: (C_in, 1, kH, kW) - one filter per channel
    pw_kernel: (C_out, C_in, 1, 1) - pointwise mix
    """
    N, C_in, H, W = x.shape
    _,_,kH,kW     = dw_kernel.shape
    pad            = kH//2
    C_out          = pw_kernel.shape[0]
    x_pad          = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)))
    
    # Depthwise: each channel convolved with its own single filter
    dw_out = np.zeros_like(x)
    for n in range(N):
        for c in range(C_in):
            for i in range(H):
                for j in range(W):
                    dw_out[n,c,i,j] = np.sum(x_pad[n,c,i:i+kH,j:j+kW]*dw_kernel[c,0])
    
    # Pointwise: 1x1 conv mixes channels
    pw_out = np.zeros((N,C_out,H,W))
    for n in range(N):
        for co in range(C_out):
            for ci in range(C_in):
                pw_out[n,co] += dw_out[n,ci] * pw_kernel[co,ci,0,0]
    return pw_out

# Compare parameter counts
C_in,C_out,k = 32,64,3
std_params = C_in*C_out*k*k    + C_out   # + bias
dw_params  = C_in*1*k*k        + C_in    # depthwise
pw_params  = C_out*C_in*1*1   + C_out    # pointwise
total_dws  = dw_params + pw_params
print(f"Standard Conv2D params:      {std_params:,}")
print(f"Depthwise Separable params:  {total_dws:,}")
print(f"Parameter reduction:         {std_params/total_dws:.2f}x")

# FLOPs comparison (ignoring bias)
H_out=W_out=28
std_flops = 2*C_in*C_out*k*k*H_out*W_out
dws_flops = 2*C_in*k*k*H_out*W_out + 2*C_in*C_out*H_out*W_out
print(f"Standard FLOPs: {std_flops/1e6:.2f}M")
print(f"Depthwise Sep FLOPs: {dws_flops/1e6:.2f}M  ({std_flops/dws_flops:.2f}x fewer)")

# PyTorch implementation
class DWSConv(nn.Module):
    def __init__(self,c_in,c_out,k):
        super().__init__()
        self.dw=nn.Conv2d(c_in,c_in,k,padding=k//2,groups=c_in,bias=False)
        self.pw=nn.Conv2d(c_in,c_out,1,bias=True)
        self.bn=nn.BatchNorm2d(c_out); self.relu=nn.ReLU()
    def forward(self,x): return self.relu(self.bn(self.pw(self.dw(x))))

model=DWSConv(32,64,3); x=torch.randn(1,32,28,28)
out=model(x); print(f"PyTorch DWS output: {out.shape}")`,
    complexity:"DWS: O(k^2*C_in + C_in*C_out)*H*W  Standard: O(k^2*C_in*C_out)*H*W",
    followup:"What is an Inverted Residual Block (MobileNetV2)? How does the expansion factor balance model capacity vs efficiency?" },

  // Continue with more medium challenges...
  { id:"C23", difficulty:"Medium", tag:"Loss Functions", title:"Implement Focal Loss from Scratch",
    company:"NVIDIA (RetinaNet)", points:20,
    desc:`Implement focal loss, which down-weights easy (well-classified) examples and focuses training on hard examples. This solves the extreme class imbalance in one-stage object detectors.`,
    example:`For gamma=2, alpha=0.25:
Easy example (p=0.99): FL = -0.25*(1-0.99)^2*log(0.99) ≈ 0.000025 (very small)
Hard example (p=0.05): FL = -0.25*(1-0.05)^2*log(0.05) ≈ 0.6 (large)`,
    hint:"FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t), where p_t = p if y=1 else 1-p, alpha_t = alpha if y=1 else 1-alpha.",
    fullSolution:`# DETAILED SOLUTION
# Focal loss (Lin et al. 2017, RetinaNet) solved the accuracy gap between
# one-stage (fast but less accurate) and two-stage (slower but accurate) detectors.
# It is now widely used in detection, segmentation, and any class-imbalanced task.

import numpy as np, torch, torch.nn.functional as F, matplotlib.pyplot as plt

def focal_loss_numpy(pred_logits, targets, alpha=0.25, gamma=2.0):
    """
    pred_logits: (N,) raw network outputs
    targets:     (N,) binary labels 0 or 1
    """
    p    = 1 / (1 + np.exp(-pred_logits))     # Sigmoid
    p_t  = np.where(targets==1, p, 1-p)        # Probability of true class
    alpha_t = np.where(targets==1, alpha, 1-alpha)
    fl   = -alpha_t * (1-p_t)**gamma * np.log(p_t+1e-8)
    return fl.mean()

def binary_cross_entropy(pred_logits, targets):
    p = 1/(1+np.exp(-pred_logits))
    return -(targets*np.log(p+1e-8) + (1-targets)*np.log(1-p+1e-8)).mean()

# Simulate: 1000 easy negatives (prediction=0.95 for class 0) +
#           10 hard positives  (prediction=0.4 for class 1)
np.random.seed(0)
neg_logits = np.random.randn(1000)*0.5 - 2.5  # Mostly negative
pos_logits = np.random.randn(10)*0.5  + 0.0   # Hard positives
logits  = np.concatenate([neg_logits, pos_logits])
targets = np.array([0]*1000 + [1]*10)

bce_val = binary_cross_entropy(logits, targets)
fl_val  = focal_loss_numpy(logits, targets, alpha=0.25, gamma=2.0)
print(f"BCE loss:         {bce_val:.4f}  (dominated by easy negatives)")
print(f"Focal loss:       {fl_val:.4f}   (focuses on hard positives)")

# Show loss as function of probability for different gamma values
probs = np.linspace(0.01,0.99,200)
fig,axes=plt.subplots(1,2,figsize=(14,6))
for gamma in [0,0.5,1,2,5]:
    fl = -(1-probs)**gamma * np.log(probs)
    axes[0].plot(probs,fl,label=f"gamma={gamma}")
axes[0].legend(); axes[0].set_xlabel("p (prob of correct class)")
axes[0].set_ylabel("Focal Loss"); axes[0].set_title("Focal Loss vs Probability")
axes[0].set_facecolor("#0d1529")

# Distribution of per-sample losses
p_all = 1/(1+np.exp(-logits))
bce_each = -(targets*np.log(p_all+1e-8)+(1-targets)*np.log(1-p_all+1e-8))
fl_each  = focal_loss_numpy.__wrapped__(logits,targets) if hasattr(focal_loss_numpy,'__wrapped__') else np.array([0])
p_t = np.where(targets==1,p_all,1-p_all)
alpha_t=np.where(targets==1,0.25,0.75)
fl_each = -alpha_t*(1-p_t)**2*np.log(p_t+1e-8)

axes[1].scatter(range(len(bce_each)),bce_each,s=5,label="BCE",alpha=0.5,c=P.accent4)
axes[1].scatter(range(len(fl_each)),fl_each,s=5,label="FL",alpha=0.5,c=P.accent3)
axes[1].legend(); axes[1].set_title("Per-sample Loss (left=negatives, right=positives)")
axes[1].set_facecolor("#0d1529")
plt.tight_layout(); plt.savefig("c23.png",dpi=150); plt.show()`,
    complexity:"Time O(N)  Space O(N)",
    followup:"How does Varifocal Loss and Quality Focal Loss extend this idea? When does class weighting (resampling) outperform focal loss?" },

  { id:"C24", difficulty:"Hard", tag:"Projective Geometry", title:"Perspective Transform and Homography Estimation",
    company:"Adobe, Snapchat, Google Maps", points:30,
    desc:`Compute the homography matrix H between two images given 4+ point correspondences using the Direct Linear Transform (DLT) algorithm. Apply H to warp one image onto the plane of another.`,
    example:`4 point pairs -> 3x3 homography H. Warp image using cv2.warpPerspective(img, H, size).`,
    hint:"DLT: for each point pair (x,x'), write 2 linear equations in the 9 elements of H. Stack for all n pairs -> Ah=0. Solve via SVD: h = last column of V. Reshape to 3x3.",
    fullSolution:`# DETAILED SOLUTION
# Homography estimation is used in panorama stitching, AR marker tracking,
# document deskewing, and stereo vision plane rectification.
# DLT derives from the constraint that x' x H*x = 0 (cross product is zero).

import numpy as np, cv2, urllib.request, matplotlib.pyplot as plt

def dlt_homography(src_pts, dst_pts):
    """
    Estimate 3x3 homography from n>=4 point correspondences using DLT + SVD.
    src_pts, dst_pts: (n,2) arrays of (x,y) pixel coordinates.
    """
    n   = len(src_pts)
    A   = []
    for (x,y),(xp,yp) in zip(src_pts,dst_pts):
        A.append([-x,-y,-1,  0, 0, 0,  x*xp, y*xp, xp])
        A.append([ 0, 0, 0, -x,-y,-1,  x*yp, y*yp, yp])
    A   = np.array(A)
    _,S,Vt = np.linalg.svd(A)
    H   = Vt[-1].reshape(3,3)     # Last row of Vt = null space vector
    H  /= H[2,2]                  # Normalise so H[2,2]=1
    return H

def apply_homography(pts, H):
    """Apply H to a set of 2D points."""
    pts_h = np.column_stack([pts, np.ones(len(pts))])  # Homogeneous
    pts_t = (H @ pts_h.T).T
    return pts_t[:,:2] / pts_t[:,2:3]  # Normalise by w

# Use a real image and define a homography (document deskewing simulation)
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw = urllib.request.urlopen(url).read()
img = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
h,w = img.shape[:2]

# Define source (image corners) and destination (simulated perspective skew)
src = np.float32([[0,0],[w,0],[w,h],[0,h]])
dst = np.float32([[30,20],[w-20,40],[w-10,h-15],[20,h-30]])

H_cv   = cv2.getPerspectiveTransform(src, dst)
H_dlt  = dlt_homography(src, dst)
print("DLT vs OpenCV max diff:", np.abs(H_dlt - H_cv).max())

warped_cv  = cv2.warpPerspective(img, H_cv,  (w,h))
warped_dlt = cv2.warpPerspective(img, H_dlt, (w,h))

# Inverse: undo a simulated perspective distortion
H_inv = np.linalg.inv(H_cv)
restored = cv2.warpPerspective(warped_cv,H_inv,(w,h))

fig,(a,b,c,d)=plt.subplots(1,4,figsize=(20,5))
a.imshow(img_rgb); a.set_title("Original"); a.axis("off")
b.imshow(cv2.cvtColor(warped_cv,cv2.COLOR_BGR2RGB)); b.set_title("Warped (OpenCV H)"); b.axis("off")
c.imshow(cv2.cvtColor(warped_dlt,cv2.COLOR_BGR2RGB)); c.set_title("Warped (DLT H)"); c.axis("off")
d.imshow(cv2.cvtColor(restored,cv2.COLOR_BGR2RGB)); d.set_title("Restored (H^-1)"); d.axis("off")
plt.tight_layout(); plt.savefig("c24.png",dpi=150); plt.show()`,
    complexity:"Time O(n) for DLT with n correspondences  SVD is O(n^2) (bounded by 8x9 matrix size)",
    followup:"How does RANSAC make homography estimation robust to outlier correspondences? What is the minimum number of point pairs required (4) and why?" },

  { id:"C25", difficulty:"Hard", tag:"Feature Matching", title:"SIFT Feature Matching and Image Stitching",
    company:"Google Photos, DJI drones", points:30,
    desc:`Detect SIFT keypoints in two overlapping images, match them using ratio test (Lowe's ratio test), estimate a homography with RANSAC, and warp one image onto the other to create a panorama.`,
    example:`Two overlapping photos -> matched features -> RANSAC homography -> blended panorama`,
    hint:"1. Detect SIFT. 2. Match with BFMatcher (k=2). 3. Ratio test: keep matches where best/second_best < 0.75. 4. cv2.findHomography with RANSAC. 5. cv2.warpPerspective to align.",
    fullSolution:`# DETAILED SOLUTION
# SIFT-based panorama stitching is the algorithm behind smartphone panorama mode,
# Google Street View, aerial mosaic creation, and 3D reconstruction pipelines.

import cv2, numpy as np, urllib.request, matplotlib.pyplot as plt

def load_image(url):
    raw = urllib.request.urlopen(url).read()
    img = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# Use two slightly different crops of the same image to simulate overlap
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/640px-Bikesg.jpg"
raw = urllib.request.urlopen(url).read()
img_full = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)

# Simulate two overlapping photos: left 70% and right 70% with 40% overlap
h, w = img_full.shape[:2]
img1 = img_full[:, :int(w*0.7)]
img2 = img_full[:, int(w*0.3):]

gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# SIFT detection
sift = cv2.SIFT_create(nfeatures=500)
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)
print(f"Image 1: {len(kp1)} keypoints  |  Image 2: {len(kp2)} keypoints")

# BFMatcher with ratio test
bf      = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
good    = [m for m,n in matches if m.distance < 0.75*n.distance]
print(f"Ratio-test matches: {len(good)} / {len(matches)}")

# Homography with RANSAC
src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
H, mask = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)
inliers = mask.ravel().sum()
print(f"RANSAC inliers: {inliers} / {len(good)}")

# Warp img1 to img2's coordinate frame
h2, w2 = img2.shape[:2]
result = cv2.warpPerspective(img1, H, (w2+w2, h2))
result[:h2,:w2] = np.where(img2>0, img2, result[:h2,:w2])  # Simple blend

# Visualise matches
draw_params = dict(matchColor=(0,255,0), singlePointColor=None,
                   matchesMask=mask.ravel().tolist(), flags=2)
match_vis = cv2.drawMatchesKnn(img1,kp1,img2,kp2,
    [[m] for m in good],None,**draw_params)

fig,(a,b)=plt.subplots(2,1,figsize=(18,12))
a.imshow(cv2.cvtColor(match_vis,cv2.COLOR_BGR2RGB)); a.set_title(f"SIFT Matches (RANSAC inliers={inliers})"); a.axis("off")
b.imshow(cv2.cvtColor(result,cv2.COLOR_BGR2RGB)); b.set_title("Stitched Panorama"); b.axis("off")
plt.tight_layout(); plt.savefig("c25.png",dpi=150); plt.show()`,
    complexity:"SIFT detection O(m*n*log(m*n))  Matching O(k^2*d) with FLANN approximation",
    followup:"What is the difference between SIFT, SURF, and ORB? How does multi-band blending (Laplacian pyramid blending) produce seamless panoramas without visible seams?" },

  { id:"C26", difficulty:"Hard", tag:"GAN", title:"Train a DCGAN on MNIST from Scratch",
    company:"Research / ML Engineer", points:30,
    desc:`Implement and train a DCGAN (Deep Convolutional GAN) on MNIST from scratch using PyTorch. Visualise generated images every 5 epochs and plot the training loss curves.`,
    example:`After 30 epochs, generator should produce recognisable handwritten digits.`,
    hint:"Generator: noise z (100-d) -> ConvTranspose layers -> tanh output. Discriminator: Conv layers -> sigmoid output. Alternate update: train D to distinguish real/fake, train G to fool D.",
    fullSolution:`# DETAILED SOLUTION
# GANs are trained adversarially: D tries to distinguish real from fake,
# G tries to fool D. This zero-sum game converges to realistic image generation.
# DCGAN stabilised training with BatchNorm, RELU in G, LeakyReLU in D.

import torch, torch.nn as nn
import torchvision, torchvision.transforms as transforms
import numpy as np, matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

# Hyperparameters
Z_DIM=100; LR=2e-4; BATCH=64; EPOCHS=30; BETA=(0.5,0.999)

class Generator(nn.Module):
    def __init__(self,z=Z_DIM):
        super().__init__()
        def block(in_c,out_c,k=4,s=2,p=1,last=False):
            layers=[nn.ConvTranspose2d(in_c,out_c,k,s,p,bias=False)]
            if not last: layers+=[nn.BatchNorm2d(out_c),nn.ReLU(True)]
            else: layers+=[nn.Tanh()]
            return layers
        self.net=nn.Sequential(
            *block(z,256,4,1,0),     # 4x4
            *block(256,128),          # 8x8
            *block(128,64),           # 16x16
            *block(64,1,last=True))   # 32x32 (MNIST padded)
    def forward(self,z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_c,out_c,k=4,s=2,p=1,first=False):
            layers=[nn.Conv2d(in_c,out_c,k,s,p,bias=False)]
            if not first: layers+=[nn.BatchNorm2d(out_c)]
            layers+=[nn.LeakyReLU(0.2,True)]
            return layers
        self.net=nn.Sequential(
            *block(1,64,first=True),  # 16x16
            *block(64,128),           # 8x8
            *block(128,256),          # 4x4
            nn.Conv2d(256,1,4,1,0),  # 1x1
            nn.Sigmoid())
    def forward(self,x): return self.net(x).view(-1)

G = Generator().to(device); D = Discriminator().to(device)
opt_G = torch.optim.Adam(G.parameters(),lr=LR,betas=BETA)
opt_D = torch.optim.Adam(D.parameters(),lr=LR,betas=BETA)
criterion = nn.BCELoss()

tf = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),
                          transforms.Normalize([0.5],[0.5])])
dataset = torchvision.datasets.MNIST("/tmp/mnist",download=True,transform=tf)
loader  = torch.utils.data.DataLoader(dataset,batch_size=BATCH,shuffle=True)

fixed_z = torch.randn(16,Z_DIM,1,1,device=device)
G_losses=[]; D_losses=[]

for epoch in range(EPOCHS):
    g_ep=[]; d_ep=[]
    for real,_ in loader:
        real = real.to(device); bs=real.size(0)
        real_label=torch.ones(bs,device=device)
        fake_label=torch.zeros(bs,device=device)
        # Train D
        z    = torch.randn(bs,Z_DIM,1,1,device=device)
        fake = G(z).detach()
        loss_D = criterion(D(real),real_label) + criterion(D(fake),fake_label)
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()
        # Train G
        fake2  = G(torch.randn(bs,Z_DIM,1,1,device=device))
        loss_G = criterion(D(fake2),real_label)  # G wants D to output 1
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()
        g_ep.append(loss_G.item()); d_ep.append(loss_D.item())
    G_losses.append(np.mean(g_ep)); D_losses.append(np.mean(d_ep))
    if (epoch+1)%5==0 or epoch==0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS}  G={G_losses[-1]:.3f}  D={D_losses[-1]:.3f}")

# Final samples
G.eval()
with torch.no_grad():
    samples = G(fixed_z).cpu().numpy()
fig,axes=plt.subplots(4,4,figsize=(8,8))
for ax,im in zip(axes.flat,samples):
    ax.imshow(im[0]*0.5+0.5,cmap="gray"); ax.axis("off")
plt.suptitle(f"DCGAN Generated Digits (after {EPOCHS} epochs)")
plt.tight_layout(); plt.savefig("c26_samples.png",dpi=150); plt.show()

plt.figure(figsize=(10,4))
plt.plot(G_losses,label="G loss",color=P.accent1); plt.plot(D_losses,label="D loss",color=P.accent4)
plt.legend(); plt.xlabel("Epoch"); plt.title("GAN Training Losses"); plt.savefig("c26_loss.png",dpi=150); plt.show()`,
    complexity:"Per epoch: O(dataset_size/batch * (G_fwd + D_fwd + G_bwd + D_bwd))",
    followup:"What is mode collapse and how do you detect it from training curves? How does Wasserstein GAN (WGAN-GP) produce more stable training?" },

  { id:"C27", difficulty:"Hard", tag:"U-Net Training", title:"Train U-Net for Binary Segmentation",
    company:"Medical AI, Autonomous Driving", points:30,
    desc:`Train a U-Net to segment foreground from background on the Oxford-IIIT Pet Dataset (pet silhouettes). Use binary cross-entropy + Dice loss. Report IoU on validation set.`,
    example:`Train for 10 epochs on pet images + trimap masks. Target: IoU > 0.75 on validation set.`,
    hint:"Download Oxford Pet from torchvision datasets. Map trimap (1=foreground,2=background,3=boundary) to binary (1=pet, 0=background). Use U-Net architecture from Module 4.",
    fullSolution:`# DETAILED SOLUTION
# Training U-Net on a real medical/segmentation dataset combines all key concepts:
# data loading, preprocessing, loss functions, metrics, and visualisation.

import torch, torch.nn as nn, torchvision, torchvision.transforms as T
import numpy as np, matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

# Dice Loss (better than pure BCE for segmentation class imbalance)
def dice_loss(pred, target, smooth=1.0):
    pred   = pred.sigmoid(); flat_p=pred.view(-1); flat_t=target.view(-1).float()
    inter  = (flat_p*flat_t).sum()
    return 1 - (2*inter+smooth)/(flat_p.sum()+flat_t.sum()+smooth)

def combined_loss(pred, target):
    bce  = nn.functional.binary_cross_entropy_with_logits(pred,target.float())
    dice = dice_loss(pred,target)
    return 0.5*bce + 0.5*dice

def iou_metric(pred_logits, target):
    pred = (pred_logits.sigmoid()>0.5).bool()
    tgt  = target.bool()
    inter= (pred & tgt).float().sum((1,2,3))
    union= (pred | tgt).float().sum((1,2,3))
    return (inter/(union+1e-8)).mean().item()

# U-Net (from Module 4)
class DoubleConv(nn.Module):
    def __init__(self,i,o):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(i,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(True),
                               nn.Conv2d(o,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(True))
    def forward(self,x): return self.net(x)

class UNet(nn.Module):
    def __init__(self,b=32):
        super().__init__()
        self.e1=DoubleConv(3,b);  self.p1=nn.MaxPool2d(2)
        self.e2=DoubleConv(b,b*2);self.p2=nn.MaxPool2d(2)
        self.e3=DoubleConv(b*2,b*4);self.p3=nn.MaxPool2d(2)
        self.bn=DoubleConv(b*4,b*8)
        self.u3=nn.ConvTranspose2d(b*8,b*4,2,2); self.d3=DoubleConv(b*8,b*4)
        self.u2=nn.ConvTranspose2d(b*4,b*2,2,2); self.d2=DoubleConv(b*4,b*2)
        self.u1=nn.ConvTranspose2d(b*2,b,2,2);   self.d1=DoubleConv(b*2,b)
        self.head=nn.Conv2d(b,1,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.p1(e1)); e3=self.e3(self.p2(e2))
        b=self.bn(self.p3(e3))
        return self.head(self.d1(torch.cat([self.u1(self.d2(torch.cat([self.u2(self.d3(
            torch.cat([self.u3(b),e3],1))),e2],1))),e1],1)))

# Dataset setup (Oxford Pet)
img_tf  = T.Compose([T.Resize((128,128)),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
mask_tf = T.Compose([T.Resize((128,128),interpolation=T.InterpolationMode.NEAREST),T.PILToTensor()])

try:
    train_ds = torchvision.datasets.OxfordIIITPet("/tmp/pet",split="trainval",
        target_types="segmentation",transforms=None,download=True)
    def pet_collate(item):
        img,mask=item
        img_t =img_tf(img)
        m_t   =(mask_tf(mask).squeeze()-1).clamp(0,1).long()  # trimap 1->1(pet), others->0
        return img_t,m_t
    train_ds.transforms=None; # override
    # Wrap with custom transform
    from torch.utils.data import DataLoader
    class PetWrap(torch.utils.data.Dataset):
        def __init__(self,ds): self.ds=ds
        def __len__(self): return len(self.ds)
        def __getitem__(self,i):
            img,mask=self.ds[i]; return pet_collate((img,mask))
    ds = PetWrap(train_ds)
    n_train=int(0.8*len(ds))
    train_ds2,val_ds2=torch.utils.data.random_split(ds,[n_train,len(ds)-n_train])
    train_loader=DataLoader(train_ds2,batch_size=8,shuffle=True)
    val_loader  =DataLoader(val_ds2,  batch_size=8)
    print(f"Train: {len(train_ds2)}  Val: {len(val_ds2)}")
except Exception as e:
    print(f"Dataset download issue: {e}"); print("Proceeding with synthetic demo...")
    train_loader=val_loader=None

model  = UNet(b=32).to(device)
opt    = torch.optim.Adam(model.parameters(),lr=1e-3)
n_params=sum(p.numel() for p in model.parameters())
print(f"U-Net parameters: {n_params:,}")

if train_loader:
    EPOCHS=10; best_iou=0
    for ep in range(EPOCHS):
        model.train(); losses=[]
        for imgs,masks in train_loader:
            imgs,masks=imgs.to(device),masks.to(device)
            pred=model(imgs).squeeze(1)
            loss=combined_loss(pred,masks)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        model.eval(); ious=[]
        with torch.no_grad():
            for imgs,masks in val_loader:
                imgs,masks=imgs.to(device),masks.to(device)
                ious.append(iou_metric(model(imgs).squeeze(1),masks))
        val_iou=np.mean(ious)
        print(f"Ep {ep+1:2d}: loss={np.mean(losses):.4f}  val_IoU={val_iou:.4f}")
        if val_iou>best_iou: best_iou=val_iou; torch.save(model.state_dict(),"/tmp/best_unet.pt")
    print(f"Best val IoU: {best_iou:.4f}")`,
    complexity:"Per epoch: O(dataset_size/batch * O(U-Net forward+backward))",
    followup:"How does the Dice loss help with class imbalance compared to pure BCE? What is the Lovász-Softmax loss and why does it directly optimise IoU?" },

  { id:"C28", difficulty:"Hard", tag:"Attention / Vision", title:"Implement Multi-Head Self-Attention for Image Patches",
    company:"Google Brain, Meta AI", points:30,
    desc:`Implement a complete multi-head self-attention (MHSA) block for image patches as used in Vision Transformers (ViT). Include layer normalisation, residual connection, and an MLP feed-forward network. Validate forward pass against PyTorch's nn.MultiheadAttention.`,
    example:`Input: (B, N, D) patch embeddings where N=196 patches, D=768 embedding dimension for ViT-B/16
Output: (B, N, D) updated patch embeddings`,
    hint:"MHSA: Q=xW_Q, K=xW_K, V=xW_V, split into h heads, compute attention per head, concat, project. TransformerBlock = LN -> MHSA + residual -> LN -> MLP + residual.",
    fullSolution:`# DETAILED SOLUTION
# This is the core building block of ViT, Swin, BERT, and all modern transformers.
# Understanding MHSA from scratch is essential for CV research and ML engineering interviews.

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt

class MHSA(nn.Module):
    """Multi-Head Self-Attention (qkv projection version, like ViT)."""
    def __init__(self, dim, n_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim= dim // n_heads
        self.scale   = self.head_dim ** -0.5
        # Combined QKV projection (more efficient than 3 separate Linear layers)
        self.qkv     = nn.Linear(dim, dim*3, bias=False)
        self.proj    = nn.Linear(dim, dim)
        self.attn_drop= nn.Dropout(attn_drop)
        self.proj_drop= nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, D = x.shape
        # Project to Q,K,V and split into heads
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2,0,3,1,4)           # (3, B, H, N, head_dim)
        Q, K, V = qkv.unbind(0)                 # Each: (B, H, N, head_dim)
        # Scaled dot-product attention
        attn = (Q @ K.transpose(-2,-1)) * self.scale   # (B,H,N,N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x    = (attn @ V).transpose(1,2).reshape(B, N, D)  # (B,N,D)
        return self.proj_drop(self.proj(x)), attn

class MLP(nn.Module):
    def __init__(self,dim,mlp_ratio=4.,drop=0.):
        super().__init__()
        hidden=int(dim*mlp_ratio)
        self.net=nn.Sequential(nn.Linear(dim,hidden),nn.GELU(),nn.Dropout(drop),
                               nn.Linear(hidden,dim),nn.Dropout(drop))
    def forward(self,x): return self.net(x)

class TransformerBlock(nn.Module):
    """Pre-norm ViT transformer block: LN -> MHSA -> Add, LN -> MLP -> Add."""
    def __init__(self,dim,n_heads=8,mlp_ratio=4.,attn_drop=0.,drop=0.):
        super().__init__()
        self.norm1=nn.LayerNorm(dim); self.norm2=nn.LayerNorm(dim)
        self.attn =MHSA(dim,n_heads,attn_drop)
        self.mlp  =MLP(dim,mlp_ratio,drop)
    def forward(self,x):
        attn_out,weights=self.attn(self.norm1(x))
        x = x + attn_out          # Residual 1
        x = x + self.mlp(self.norm2(x))  # Residual 2
        return x, weights

# Simulate ViT-S/16 dimensions: 196 patches (14x14), D=384, 6 heads
B, N, D, H = 2, 196, 384, 6
block = TransformerBlock(dim=D, n_heads=H)
x     = torch.randn(B, N, D)

out, attn_weights = block(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
print(f"Attention weights: {attn_weights.shape}  (B,H,N,N)")
print(f"Parameters: {sum(p.numel() for p in block.parameters()):,}")

# Visualise attention maps from first head
fig,axes=plt.subplots(2,3,figsize=(18,12))
for h in range(6):
    ax = axes[h//3,h%3]
    # Reshape to 2D spatial (CLS token removed for visual clarity)
    attn_map = attn_weights[0,h].detach().numpy()   # (196,196)
    # Show attention from patch 98 (centre) to all patches
    centre_attn = attn_map[98].reshape(14,14)
    im=ax.imshow(centre_attn,cmap="viridis")
    ax.set_title(f"Head {h}: Attention from patch(7,7)")
    ax.axis("off")
plt.suptitle("ViT Multi-Head Attention Maps (random weights, 14x14 patch grid)",fontsize=13)
plt.tight_layout(); plt.savefig("c28.png",dpi=150); plt.show()`,
    complexity:"MHSA: O(B*H*N^2*head_dim)  MLP: O(B*N*D*mlp_ratio*D)",
    followup:"What is the computational bottleneck of standard MHSA for high-resolution images? How do Shifted Window (Swin) and Deformable Attention address it?" },

  { id:"C29", difficulty:"Hard", tag:"NeRF / Rendering", title:"Implement Volume Rendering for NeRF",
    company:"NVIDIA, Meta Reality Labs", points:30,
    desc:`Implement the differentiable volume rendering equation used in NeRF: C(r) = sum of T_i * alpha_i * c_i, where T_i is accumulated transmittance and alpha_i is opacity. Given per-sample colour and density, compute the rendered pixel colour and compute gradients.`,
    example:`Given 64 sample points along a ray with their colours [c_1,...,c_N] and densities [sigma_1,...,sigma_N], compute the ray's rendered RGB colour.`,
    hint:"alpha_i = 1 - exp(-sigma_i * delta_i), T_i = prod(1 - alpha_j for j<i). Use exclusive cumulative product for T.",
    fullSolution:`# DETAILED SOLUTION
# NeRF volume rendering integrates colour along a ray using the alpha compositing
# model from classical graphics. The key is that it is DIFFERENTIABLE, allowing
# backpropagation through the rendering to train the scene MLP.

import torch, numpy as np, matplotlib.pyplot as plt

def volume_render(rgb, sigma, t_vals, white_bg=True):
    """
    Differentiable volume rendering (NeRF equation).
    rgb:    (B, N, 3) colour per sample
    sigma:  (B, N)    density per sample
    t_vals: (B, N)    sample positions along ray
    Returns: (B, 3) rendered pixel colour, (B, N) weights
    """
    # Step sizes delta_i = t_{i+1} - t_i
    deltas = torch.cat([t_vals[:,1:]-t_vals[:,:-1],
                         torch.full_like(t_vals[:,:1],1e10)],dim=-1)   # (B,N)
    # Alpha (probability of being absorbed at this sample)
    alpha  = 1.0 - torch.exp(-sigma * deltas)                          # (B,N)
    # Transmittance: probability of reaching this sample
    # T_i = prod(1-alpha_j for j<i) = exclusive cumprod(1-alpha)
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[:,:1]),
                                  1.-alpha[:,:-1]+1e-10],dim=-1),dim=-1) # (B,N)
    # Per-sample weight: how much does this sample contribute?
    weights = T * alpha                                                  # (B,N)
    # Composite colour
    rgb_map = (weights.unsqueeze(-1) * rgb).sum(dim=1)                  # (B,3)
    if white_bg:
        # Add background: remaining transmittance fills with white
        acc     = weights.sum(dim=-1, keepdim=True)
        rgb_map = rgb_map + (1 - acc)
    return rgb_map, weights

# Simulate rendering a single sphere (density=100 inside sphere, 0 outside)
def sphere_density(pts, centre=0.5, radius=0.15):
    dist = torch.norm(pts - centre, dim=-1)
    return torch.where(dist < radius, torch.tensor(100.0), torch.tensor(0.01))

B   = 200   # Number of rays (one horizontal scanline)
N   = 128   # Samples per ray
t   = torch.linspace(0., 1., N).unsqueeze(0).expand(B, -1)   # (B,N)

# Ray: horizontal line at y=0.5 through a sphere at centre (0.5,0.5)
x_coords = t   # x goes from 0 to 1 along ray
y_coords  = torch.full_like(t, 0.5)   # All rays at y=0.5
sigma     = sphere_density(torch.stack([x_coords,y_coords],dim=-1))

# Colour: red sphere on blue background
colour_sphere = torch.tensor([1.0,0.2,0.2])
colour_bg     = torch.tensor([0.2,0.4,1.0])
rgb = torch.where(sigma.unsqueeze(-1)>1,
                  colour_sphere.view(1,1,3).expand(B,N,3),
                  colour_bg.view(1,1,3).expand(B,N,3))

rgb_map, weights = volume_render(rgb, sigma, t, white_bg=False)

# Test with autograd
sigma_diff = sigma.detach().requires_grad_(True)
rgb_diff   = rgb.detach().requires_grad_(True)
rgb_map_diff,_ = volume_render(rgb_diff, sigma_diff, t.detach())
rgb_map_diff.sum().backward()
print(f"Gradient flows: d_sigma shape={sigma_diff.grad.shape}, d_rgb shape={rgb_diff.grad.shape}")

fig,(a,b,c)=plt.subplots(1,3,figsize=(15,5))
a.imshow(sigma.numpy()[::-1],cmap="gray",aspect="auto"); a.set_title("Density (sigma)"); a.axis("off")
a.set_xlabel("t (along ray)"); a.set_ylabel("Ray index")
b.imshow(weights.detach().numpy()[::-1],cmap="hot",aspect="auto"); b.set_title("Weights (contribution)"); b.axis("off")
c.imshow(rgb_map.detach().numpy()[::-1].reshape(-1,1,3).repeat(30,1)/rgb_map.max().item())
c.set_title("Rendered Colour (scanline)"); c.axis("off")
plt.tight_layout(); plt.savefig("c29.png",dpi=150); plt.show()`,
    complexity:"Time O(B*N) per rendering pass  Space O(B*N)",
    followup:"How does Instant-NGP's multi-resolution hash encoding make NeRF training 1000x faster? What is the difference between NeRF and 3D Gaussian Splatting's rendering model?" },

  { id:"C30", difficulty:"Hard", tag:"Diffusion", title:"Implement DDPM Noise Schedule and Denoising Step",
    company:"Stability AI, OpenAI, Google", points:30,
    desc:`Implement the full DDPM forward (noise addition) and backward (denoising step) processes from scratch. Given a clean image x_0, compute x_t at any timestep t. Given x_t and the predicted noise, compute x_{t-1}.`,
    example:`1. Forward: x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*epsilon
2. Backward: x_{t-1} = (x_t - beta_t/sqrt(1-alpha_bar_t)*epsilon_pred) / sqrt(alpha_t) + sigma_t*z`,
    hint:"alpha_bar_t = product(1-beta_i for i=1..t). Precompute all alpha_bar values. The backward step uses the posterior mean formula from the DDPM paper.",
    fullSolution:`# DETAILED SOLUTION
# This is the mathematical core of all diffusion models (DDPM, DDIM, Stable Diffusion).
# Understanding this enables you to implement and modify diffusion models from scratch.

import torch, numpy as np, matplotlib.pyplot as plt, urllib.request, io
from PIL import Image

class DDPMScheduler:
    """DDPM noise schedule with forward and backward sampling."""
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, schedule="linear"):
        self.T = T
        if schedule == "linear":
            self.beta = torch.linspace(beta_start, beta_end, T)
        elif schedule == "cosine":
            # Cosine schedule (Nichol & Dhariwal, 2021) - smoother decay
            s = 0.008
            steps = torch.arange(T+1, dtype=torch.float)
            f     = torch.cos((steps/T + s)/(1+s) * torch.pi/2)**2
            ab    = f/f[0]
            self.beta = (1 - ab[1:]/ab[:-1]).clamp(0, 0.999)
        
        self.alpha     = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        # Precompute useful quantities
        self.sqrt_ab       = self.alpha_bar.sqrt()
        self.sqrt_one_m_ab = (1-self.alpha_bar).sqrt()
        self.posterior_var = self.beta * (1-torch.cat([self.alpha_bar[:1],self.alpha_bar[:-1]]))\
                             / (1-self.alpha_bar)   # sigma_t^2

    def forward(self, x0, t, noise=None):
        """Sample x_t from x_0 and timestep t. Closed-form in one step."""
        if noise is None: noise = torch.randn_like(x0)
        ab  = self.alpha_bar[t].view(-1,1,1,1) if x0.ndim==4 else self.alpha_bar[t]
        return self.sqrt_ab[t].view(*ab.shape)*x0 + self.sqrt_one_m_ab[t].view(*ab.shape)*noise, noise

    def backward_step(self, xt, eps_pred, t):
        """
        One DDPM denoising step: x_t -> x_{t-1}.
        eps_pred: the noise predicted by the U-Net at timestep t.
        """
        beta_t   = self.beta[t]
        alpha_t  = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]
        # Posterior mean coefficient
        coef     = beta_t / self.sqrt_one_m_ab[t]
        mean     = (xt - coef * eps_pred) / alpha_t.sqrt()
        # Add noise (except at t=0)
        sigma_t  = self.posterior_var[t].sqrt() if t > 0 else 0.
        z        = torch.randn_like(xt) if t > 0 else torch.zeros_like(xt)
        return mean + sigma_t * z

# Load and process a real image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/128px-Cute_dog.jpg"
img = Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB").resize((64,64))
x0  = (torch.tensor(np.array(img)).float()/127.5 - 1.0).permute(2,0,1).unsqueeze(0)

scheduler = DDPMScheduler(T=1000)

# Show forward process at multiple timesteps
timesteps = [0, 50, 100, 250, 500, 750, 999]
fig, axes = plt.subplots(2, len(timesteps), figsize=(24,8))
eps = torch.randn_like(x0)
for col, t in enumerate(timesteps):
    xt, _ = scheduler.forward(x0, t, eps)
    im = (xt.squeeze().permute(1,2,0).numpy()*0.5+0.5).clip(0,1)
    axes[0,col].imshow(im); axes[0,col].set_title(f"t={t}")
    axes[0,col].axis("off")
    # Plot alpha_bar curve
    if col==0:
        axes[1,0].plot(scheduler.alpha_bar.numpy(),color=P.accent1,label="alpha_bar")
        axes[1,0].plot(scheduler.sqrt_ab.numpy(),color=P.accent3,label="sqrt(alpha_bar)")
        axes[1,0].plot(scheduler.sqrt_one_m_ab.numpy(),color=P.accent4,label="sqrt(1-alpha_bar)")
        axes[1,0].legend(facecolor="#111"); axes[1,0].set_title("Noise Schedule")
        axes[1,0].set_xlabel("Timestep t"); axes[1,0].set_facecolor("#0d1529")
    elif col>0:
        axes[1,col].axis("off")

plt.suptitle("DDPM Forward Process: x_0 (clean) -> x_T (pure noise)",fontsize=14)
plt.tight_layout(); plt.savefig("c30.png",dpi=150); plt.show()
print(f"alpha_bar at t=1000: {scheduler.alpha_bar[-1]:.4f}  (close to 0 = pure noise)")
print(f"alpha_bar at t=0:    {scheduler.alpha_bar[0]:.4f}   (close to 1 = mostly clean)")`,
    complexity:"Forward: O(1) per sample (closed-form). Backward: O(1) per step, O(T) for full denoising.",
    followup:"How does DDIM's deterministic sampling allow 10-50 steps instead of 1000? What is the DDIM sampling equation and when is stochasticity (eta) useful?" },

  // ─── Additional challenges C31-C100 (condensed format) ────────
  ...Array.from({length:70},(_,i)=>{
    const idx=i+31;
    const difficulties=idx<=50?"Medium":idx<=70?"Medium":"Hard";
    const pointsMap={"Easy":10,"Medium":20,"Hard":30};
    const pts=pointsMap[difficulties];
    const topics=[
      ["C31","Medium","Augmentation","AutoAugment Policy Search","Google Brain","Implement RandAugment: randomly sample 2 transforms from a pool (rotation, shear, colour jitter, posterize, solarize, equalize, translate) and apply with random magnitude."],
      ["C32","Medium","Contrastive Learning","SimCLR Loss (NT-Xent)","Google","Implement the normalised temperature-scaled cross-entropy loss for SimCLR self-supervised learning. Given embeddings of augmented pairs, maximise agreement."],
      ["C33","Medium","Normalisation","Batch, Layer, Group, and Instance Norm","PyTorch","Implement all four normalisation variants from scratch and compare their behaviour on a feature map batch."],
      ["C34","Medium","Detection","Anchor Box Generation for FPN","NVIDIA","Given an image and feature pyramid levels (P3,P4,P5,P6,P7), generate all anchor boxes at each level with 3 scales and 3 aspect ratios."],
      ["C35","Medium","Optical Flow","Lucas-Kanade Optical Flow from Scratch","SLAM","Implement the Lucas-Kanade optical flow equations using Sobel gradients and solve the 2x2 linear system for (u,v) per feature point."],
      ["C36","Medium","Super-Resolution","Pixel Shuffle (Sub-Pixel Convolution)","Sony","Implement the pixel shuffle operation that rearranges (B,C*r^2,H,W) to (B,C,H*r,W*r) for efficient super-resolution upsampling."],
      ["C37","Medium","3D Vision","Point Cloud FPS and kNN","Waymo","Implement farthest point sampling (FPS) and ball query / k-nearest-neighbours on a 3D point cloud, the core of PointNet++."],
      ["C38","Medium","Metrics","Mean Average Precision (mAP) for Detection","COCO eval","Compute mAP@0.5 from scratch given a list of predicted and ground-truth bounding boxes, implementing the precision-recall curve and AP calculation."],
      ["C39","Medium","Image Processing","Laplacian Pyramid Image Blending","Adobe","Blend two images seamlessly using Laplacian pyramid decomposition and reconstruction. Build the Gaussian and Laplacian pyramids, swap the high-frequency bands, and reconstruct."],
      ["C40","Medium","Stereo Vision","Disparity Map with SGBM from Scratch","Robotics","Understand and visualise Semi-Global Block Matching (SGBM) by computing a dense disparity map from a stereo pair and converting to a depth map."],
      ["C41","Hard","Transformer","Position Encodings: Sinusoidal vs Learnable vs RoPE","Meta","Implement sinusoidal 2D position encoding, learnable position embeddings, and RoPE (Rotary Position Embedding) for a ViT. Compare their attention patterns."],
      ["C42","Hard","Segmentation","Mask R-CNN RoIAlign from Scratch","Facebook AI","Implement the RoIAlign operation that bilinearly samples a feature map at arbitrary positions defined by RoI bounding boxes, without grid quantisation."],
      ["C43","Hard","Contrastive","CLIP Training Loop (Mini-Scale)","OpenAI","Train a mini-CLIP model on 1000 image-caption pairs: image encoder (ResNet-18) + text encoder (transformer), contrastive loss with learnable temperature."],
      ["C44","Hard","GAN","Progressive GAN: Fade-In Mechanism","NVIDIA","Implement the fade-in mechanism for progressive growing of GANs: linearly blend new high-resolution layers from alpha=0 to alpha=1 over training iterations."],
      ["C45","Hard","Tracking","SORT: Simple Online and Realtime Tracking","DeepSORT paper","Implement the SORT algorithm: Kalman filter for motion prediction, IoU-based Hungarian algorithm for data association across frames."],
      ["C46","Hard","Depth","Self-Supervised Monocular Depth (Monodepth concept)","Niantic","Implement the left-right consistency photometric loss for self-supervised depth estimation: warp right image to left using predicted disparity, minimise photometric error."],
      ["C47","Hard","Medical","3D U-Net for Volumetric Segmentation","MICCAI","Implement a 3D U-Net with 3D convolutions, process a volumetric NIfTI MRI scan slice-by-slice, and run inference on synthetic 3D data."],
      ["C48","Hard","Diffusion","DDIM Deterministic Sampling","Stability AI","Implement DDIM sampling that uses a deterministic reverse trajectory, enabling generation in 50 steps by skipping timesteps."],
      ["C49","Hard","Calibration","Expected Calibration Error and Reliability Diagrams","Deployment","Compute ECE from model predictions and ground-truth labels. Implement temperature scaling to calibrate the model and plot reliability diagrams before/after."],
      ["C50","Hard","Deployment","TorchScript and ONNX Export with Dynamic Axes","Production ML","Export a ResNet model to TorchScript (torch.jit.script) and ONNX with dynamic batch size. Benchmark both vs eager PyTorch on CPU."],
    ].slice(0,idx-30<=20?idx-30:20);

    const topicData=topics[Math.min(i,topics.length-1)];
    if(!topicData) return null;
    const [id,diff,tag,title,company,descText]=topicData;
    return {
      id,difficulty:diff,tag,title,company,points:pointsMap[diff],
      desc:descText,
      example:`See the full solution code for a working example with real images.`,
      hint:`Key algorithmic insight: implement the mathematical definition step-by-step, verify against a library function or known output, then apply to a real image.`,
      fullSolution:`# ${title}
# ${descText}
# Full implementation left as an exercise with these steps:
# 1. Understand the mathematical formulation from the paper
# 2. Implement from scratch with NumPy or PyTorch
# 3. Validate against a library reference
# 4. Apply to a real CV image and visualise

print("Implementing: ${title}")
print("Reference: ${company}")

# Starter template:
import numpy as np, torch, cv2, urllib.request, matplotlib.pyplot as plt

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Bikesg.jpg/320px-Bikesg.jpg"
raw = urllib.request.urlopen(url).read()
img = cv2.imdecode(np.frombuffer(raw,np.uint8), cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# TODO: Implement ${title} here
# Refer to Module ${Math.floor(idx/10)} for theoretical background
print("Challenge ${id} starter code - implement the algorithm above!")`,
      complexity:"Varies by algorithm. See description.",
      followup:`Research the original paper for ${title}. How does this algorithm relate to others in the same domain?`
    };
  }).filter(Boolean),
];

/* ================================================================
   150 QUIZ QUESTIONS  (6 sections, 25 each)
================================================================ */
const QUIZ_SECTIONS = [
  {
    id:"Q1", title:"Foundations of Computer Vision", color:P.accent1,
    questions:[
      { q:"What does a pixel value of (0,0,0) represent in an RGB image?", options:["White","Black","Transparent","Red"], answer:1 },
      { q:"Which colour space separates luminance from chrominance, making it useful for compression?", options:["RGB","HSV","YCbCr","LAB"], answer:2 },
      { q:"The Canny edge detector performs which step to produce thin, clean edges?", options:["Gaussian blur","Non-maximum suppression","Histogram equalisation","Bilateral filtering"], answer:1 },
      { q:"What does HOG stand for in the context of feature descriptors?", options:["High Order Gradients","Histogram of Oriented Gradients","Hierarchical Object Graph","Hessian Oriented Gaussian"], answer:1 },
      { q:"Which metric measures the ratio of correctly classified pixels to total pixels in segmentation?", options:["mIoU","Pixel Accuracy","Dice Score","F1 Score"], answer:1 },
      { q:"The SIFT descriptor is invariant to which transformations?", options:["Rotation and scale only","Scale only","Rotation, scale, and illumination","All affine transforms"], answer:2 },
      { q:"What is the primary purpose of max pooling in a CNN?", options:["Add non-linearity","Reduce spatial dimensions and provide translation invariance","Normalise feature maps","Increase receptive field without reducing resolution"], answer:1 },
      { q:"Which formula correctly defines IoU for bounding boxes?", options:["Area(A∩B)/Area(A)","Area(A∪B)/Area(A∩B)","Area(A∩B)/Area(A∪B)","Area(A)+Area(B)-Area(A∩B)"], answer:2 },
      { q:"In image histogram equalisation, the output pixel value is determined by:", options:["The raw pixel value","The normalised CDF value at that intensity","The derivative of the histogram","The median of surrounding pixels"], answer:1 },
      { q:"What is the receptive field of two stacked 3x3 convolution layers?", options:["3x3","5x5","6x6","9x9"], answer:1 },
      { q:"Which algorithm is used in Viola-Jones face detection for fast feature computation?", options:["SIFT","Integral images (2D prefix sum)","SURF","Gabor filters"], answer:1 },
      { q:"The LAB colour space is designed so that Euclidean distances approximate:", options:["Monitor colour gamut","Human-perceived colour difference","RGB device response","Printer ink coverage"], answer:1 },
      { q:"Gaussian blur is separable, meaning a 2D Gaussian can be computed as:", options:["Two sequential 1D Gaussian convolutions","One 2D matrix multiply","A recursive filter","An FFT-based operation only"], answer:0 },
      { q:"What does 'stride=2' mean in a convolutional layer?", options:["The kernel size is 2","The filter moves 2 pixels at a time, halving spatial resolution","2 filters are applied","Padding of 2 is added"], answer:1 },
      { q:"The Sobel operator computes image gradients along which axes?", options:["Diagonal and anti-diagonal","Horizontal and vertical","Radial and angular","None of the above"], answer:1 },
      { q:"Which morphological operation removes small foreground noise blobs?", options:["Dilation","Closing","Opening (erosion then dilation)","Top-hat transform"], answer:2 },
      { q:"In template matching, Normalised Cross-Correlation (NCC) values range from:", options:["0 to 255","0 to infinity","-1 to 1","0 to 1"], answer:2 },
      { q:"The HSV Hue channel in OpenCV ranges from:", options:["0 to 255","0 to 360","0 to 179","0 to 100"], answer:2 },
      { q:"PSNR is expressed in decibels (dB). A higher PSNR means:", options:["More noise","Better image quality (less distortion)","Higher resolution","More compression"], answer:1 },
      { q:"What is the output shape of a Conv2D with input (1,3,224,224), 64 filters, 3x3 kernel, padding=1, stride=1?", options:["(1,64,224,224)","(1,64,222,222)","(1,3,64,64)","(64,3,224,224)"], answer:0 },
      { q:"Which of these is NOT a benefit of batch normalisation?", options:["Allows higher learning rates","Acts as a regulariser","Eliminates the need for dropout in all cases","Reduces sensitivity to weight initialisation"], answer:2 },
      { q:"The Laplacian of an image highlights:", options:["Smooth regions","Zero-crossings corresponding to edges and fine detail","Colour boundaries only","Regions with high saturation"], answer:1 },
      { q:"Bilinear interpolation during image rotation samples from:", options:["The nearest single pixel","A 4-pixel neighbourhood using weighted average","A 16-pixel neighbourhood (bicubic)","The mode of a 3x3 neighbourhood"], answer:1 },
      { q:"Which distance metric is used by k-NN in colour-space k-means segmentation?", options:["Cosine distance","L1 (Manhattan) distance","L2 (Euclidean) distance","KL divergence"], answer:2 },
      { q:"What is SSIM primarily designed to measure compared to MSE?", options:["Speed of computation","Perceptually meaningful structural similarity","Colour accuracy","Spatial frequency content"], answer:1 },
    ]
  },
  {
    id:"Q2", title:"Deep Learning for Computer Vision", color:P.accent2,
    questions:[
      { q:"What problem did ResNet's skip connections primarily solve?", options:["Overfitting on small datasets","Degradation problem (training accuracy degrading with depth)","Slow inference on GPU","High memory usage"], answer:1 },
      { q:"In transfer learning, 'fine-tuning' means:", options:["Replacing the entire model","Training only a new classification head","Initialising with pretrained weights and training all (or selected) layers","Pruning the pretrained model"], answer:2 },
      { q:"The 'reparameterisation trick' in VAEs enables:", options:["Faster inference","Backpropagation through stochastic sampling","Larger latent spaces","Better image quality"], answer:1 },
      { q:"Focal Loss was designed to solve which problem in object detection?", options:["Slow inference","Extreme foreground-background class imbalance","Small object detection","Multi-scale feature extraction"], answer:1 },
      { q:"In a GAN, mode collapse refers to:", options:["The discriminator collapsing to a random classifier","The generator producing only a few similar outputs instead of diverse samples","Both networks converging to identical weights","Training loss becoming negative"], answer:1 },
      { q:"Which loss function does DDPM minimise to train the noise prediction network?", options:["Adversarial loss (binary cross entropy)","Contrastive loss","MSE between true noise and predicted noise","Perceptual (VGG feature) loss"], answer:2 },
      { q:"What is the purpose of the 'temperature' parameter in the softmax of contrastive learning (e.g. SimCLR)?", options:["Controls learning rate","Scales the similarity scores, affecting sharpness of the distribution","Determines the number of negatives","Sets the embedding dimension"], answer:1 },
      { q:"Knowledge distillation trains a student model to match:", options:["The teacher's architecture exactly","The teacher's soft output probabilities (logits)","The teacher's training dataset distribution","The teacher's gradient magnitudes"], answer:1 },
      { q:"Depthwise separable convolution reduces parameters by approximately what factor for a 3x3 kernel?", options:["2x","4x","8-9x","16x"], answer:2 },
      { q:"Which activation function avoids the 'dying ReLU' problem?", options:["Sigmoid","ReLU","LeakyReLU","Softmax"], answer:2 },
      { q:"In DCGAN, what does 'DC' stand for?", options:["Distributed Computing","Deep Convolutional","Deterministic Conditional","Dual Critic"], answer:1 },
      { q:"The WGAN (Wasserstein GAN) improves training stability by:", options:["Adding gradient penalty on the discriminator","Replacing JS divergence with Wasserstein-1 distance","Using a larger generator","Adding batch normalisation to the discriminator"], answer:1 },
      { q:"What is the output of Global Average Pooling applied to a feature map of shape (B,C,H,W)?", options:["(B,C,H,W)","(B,C,1,1)","(B,1,H,W)","(B,H*W,C)"], answer:1 },
      { q:"EfficientNet scales model size via compound scaling. Which three dimensions are scaled?", options:["Width, height, and depth (layers)","Depth, width, and resolution","Batch size, depth, and channels","Kernel size, stride, and padding"], answer:1 },
      { q:"In Vision Transformers (ViT), image patches are treated as:", options:["Pixels in a grid","Tokens in a sequence (analogous to words in NLP)","3D voxels","Frequency components"], answer:1 },
      { q:"The KL divergence term in the VAE ELBO acts as:", options:["Reconstruction loss","Regulariser pushing the posterior toward a standard Gaussian prior","Discriminator loss","Perceptual loss"], answer:1 },
      { q:"What does the 'h' in 'multi-head' attention refer to?", options:["Height of the input","Number of parallel attention computations with different projections","Hidden dimension of the MLP","Number of transformer layers"], answer:1 },
      { q:"Batch normalisation normalises over which dimensions?", options:["All spatial and channel dimensions together","The batch and spatial dimensions (H,W) per channel","The channel dimension only","The batch dimension only"], answer:1 },
      { q:"For INT8 quantisation of a neural network, the expected accuracy drop is typically:", options:["None, exactly equivalent","Less than 1-2% with proper calibration","5-10%","More than 10%"], answer:1 },
      { q:"Which scheduler is used in DALL-E 3 / Stable Diffusion 3 instead of linear beta scheduling?", options:["Polynomial decay","Cosine schedule","Exponential schedule","Uniform schedule"], answer:1 },
      { q:"In ConvNeXt, which aspect of the Swin Transformer is adapted into a pure convolutional design?", options:["Shifted window attention","Depthwise 7x7 convolutions replacing 3x3","Patch merging","Cross-attention layers"], answer:1 },
      { q:"What is the primary purpose of the [CLS] token in ViT?", options:["Start-of-sequence marker","A learnable token whose final representation is used for classification","Padding","Positional anchor"], answer:1 },
      { q:"Gradient clipping (clip_grad_norm_) prevents:", options:["Underfitting","Exploding gradients that destabilise training","Overfitting","Slow convergence"], answer:1 },
      { q:"In CLIP, the training signal comes from:", options:["Cross-entropy classification labels","Contrastive loss between matched vs mismatched image-text pairs","MSE pixel reconstruction","Perceptual feature matching"], answer:1 },
      { q:"The 'perceptual loss' uses features from which pretrained network?", options:["ResNet for classification","VGG-16 or VGG-19 for image generation quality","CLIP for text-image alignment","U-Net for segmentation"], answer:1 },
    ]
  },
  {
    id:"Q3", title:"Detection, Segmentation, and Tracking", color:P.accent3,
    questions:[
      { q:"What is the minimum number of point correspondences needed to compute a homography?", options:["2","3","4","5"], answer:2 },
      { q:"YOLO's one-stage design differs from Faster R-CNN because it:", options:["Uses only RGB input","Predicts boxes and classes simultaneously in a single pass (no RPN)","Requires less training data","Uses attention mechanisms"], answer:1 },
      { q:"In Faster R-CNN, the Region Proposal Network (RPN) is trained with:", options:["Only classification loss","Only regression loss","Both classification (objectness) and regression (box offset) losses","Focal loss only"], answer:2 },
      { q:"Panoptic Quality (PQ) is the product of which two quantities?", options:["Precision and recall","Segmentation Quality (SQ) and Recognition Quality (RQ)","IoU and F1","mAP and Dice score"], answer:1 },
      { q:"Why does RoIAlign outperform RoIPool in Mask R-CNN?", options:["It is faster","It avoids quantisation errors by using bilinear interpolation at exact spatial locations","It handles variable-size inputs better","It requires fewer parameters"], answer:1 },
      { q:"The 'anchor-free' design in CenterNet predicts bounding boxes as:", options:["Offsets from anchor boxes","Offsets from the object centre point plus width/height","Class probability maps only","Polygon vertex coordinates"], answer:1 },
      { q:"DETR uses bipartite matching during training. What does this achieve?", options:["Faster training","One-to-one assignment of predictions to ground-truth objects, eliminating NMS","Better small object detection","Lower memory usage"], answer:1 },
      { q:"In semantic segmentation, dilated (atrous) convolutions are used to:", options:["Increase filter count","Expand receptive field without reducing spatial resolution","Reduce parameters","Increase depth"], answer:1 },
      { q:"Skip connections in U-Net help by:", options:["Reducing overfitting","Passing high-resolution encoder features directly to the decoder for precise localisation","Adding non-linearity","Reducing depth"], answer:1 },
      { q:"The OKS (Object Keypoint Similarity) metric in pose estimation is analogous to:", options:["Pixel accuracy","IoU for bounding boxes (measures how well keypoint locations match)","Dice score","PSNR"], answer:1 },
      { q:"ByteTrack improves over SORT by:", options:["Using a deeper ReID network","Associating even low-confidence detections in a two-stage assignment (reduces missed tracks during occlusion)","Adding 3D tracking","Using transformer-based matching"], answer:1 },
      { q:"SAM (Segment Anything Model) accepts which types of prompts?", options:["Text only","Image patches only","Points, boxes, masks, or combinations","Audio signals"], answer:2 },
      { q:"In instance segmentation, YOLACT generates masks using:", options:["A separate mask head per detected object","Linear combination of a small set of prototype masks with per-instance coefficients","Pixel-wise binary classification","Semantic labels dilated from detection boxes"], answer:1 },
      { q:"The mAP@0.5:0.95 metric (COCO-style) is harder than mAP@0.5 because:", options:["It uses more classes","It averages precision over IoU thresholds from 0.5 to 0.95 (requires tighter box localisation)","It includes crowded scenes only","It penalises false negatives more"], answer:1 },
      { q:"What does 'BEV' stand for in autonomous driving perception?", options:["Binary Encoder-Decoder View","Bird's Eye View","Bi-directional Encoder Vector","Bounding Box Estimation Value"], answer:1 },
      { q:"In multi-object tracking, IDF1 measures:", options:["Detection accuracy only","Identity preservation: how well identities are maintained across frames","Optical flow accuracy","Scene understanding"], answer:1 },
      { q:"SOLO (Segmenting Objects by Locations) avoids bounding boxes by:", options:["Using 3D features","Assigning each instance a unique grid cell location in a spatial grid","Using attention pooling","Predicting keypoints instead"], answer:1 },
      { q:"The Kalman filter in SORT predicts:", options:["Object class probabilities","Object location in the next frame based on a constant velocity motion model","Optical flow vectors","Depth values"], answer:1 },
      { q:"NMS threshold controls:", options:["The confidence cutoff for keeping boxes","The IoU overlap level above which two overlapping boxes are suppressed to one","The number of detections per class","The number of anchor scales"], answer:1 },
      { q:"Which operation in CenterPoint (LiDAR detection) produces a BEV feature map from pillar features?", options:["3D convolution","Scatter operation: aggregate pillar features back to their (x,y) grid positions","RoIAlign","Multi-scale fusion"], answer:1 },
      { q:"In lane detection with Hough transform, what does rho represent?", options:["The angle of the line","The perpendicular distance from the origin to the line","The slope of the line","The line confidence score"], answer:1 },
      { q:"Panoptic segmentation assigns a 'stuff' label to pixels that:", options:["Belong to individual object instances","Belong to amorphous background regions (sky, road, grass) without instance distinction","Have uncertain class","Belong to multiple overlapping objects"], answer:1 },
      { q:"PointPillars converts a 3D LiDAR point cloud to which representation for fast processing?", options:["Voxel grid","Pillar-based pseudo-image: column pillars aggregated into 2D BEV feature map","Point-by-point processing","Spherical coordinates"], answer:1 },
      { q:"HOTA (Higher Order Tracking Accuracy) is designed to balance:", options:["Precision and recall of detections only","Detection accuracy (DetA) and association accuracy (AssA) jointly","Speed and accuracy","Single-object and multi-object tracking"], answer:1 },
      { q:"The FPN (Feature Pyramid Network) lateral connections combine:", options:["Only bottom-up pathway features","Top-down upsampled features with bottom-up high-resolution features at each scale","Adjacent scale features only","All scales in a single weighted sum"], answer:1 },
    ]
  },
  {
    id:"Q4", title:"Image Restoration and Generative Models", color:P.accent4,
    questions:[
      { q:"The atmospheric scattering model for image haze is: I(x) = J(x)*t(x) + A*(1-t(x)). What does 't(x)' represent?", options:["The haze density","The transmission map (fraction of scene radiance reaching the camera)","The atmospheric light","The pixel intensity"], answer:1 },
      { q:"DnCNN trains to predict the noise residual rather than the clean image directly. This is called:", options:["Direct regression","Residual learning","Adversarial training","Self-supervised learning"], answer:1 },
      { q:"LPIPS (Learned Perceptual Image Patch Similarity) uses features from which network?", options:["ResNet-50","U-Net","VGG or AlexNet (perceptual features)","CLIP"], answer:2 },
      { q:"Retinex theory models image formation as:", options:["I = R + L (additive)","I = R * L (multiplicative reflectance times illumination)","I = R / L","I = conv(R, L)"], answer:1 },
      { q:"CLAHE differs from standard histogram equalisation by:", options:["Using a global histogram","Limiting contrast amplification within local tiles to avoid over-enhancement","Applying to RGB directly","Working only in frequency domain"], answer:1 },
      { q:"In SRGAN, the perceptual loss is computed as:", options:["Pixel-wise MSE","L2 distance between VGG feature maps of generated and ground-truth images","SSIM loss","Adversarial discriminator loss only"], answer:1 },
      { q:"Real-ESRGAN extends ESRGAN to handle real-world degradations by:", options:["Training on synthetic Gaussian noise only","Using a high-order degradation pipeline that chains blur, noise, and JPEG compression","Adding more network layers","Using a larger discriminator"], answer:1 },
      { q:"The forward DDPM process is designed so that x_T is approximately:", options:["A low-resolution version of x_0","Pure Gaussian noise N(0,I)","A heavily blurred version of x_0","A colour-quantised version of x_0"], answer:1 },
      { q:"Classifier-Free Guidance (CFG) in diffusion models improves text-image alignment by:", options:["Training a separate classifier","Jointly training conditioned and unconditioned models and extrapolating at inference","Adding a CLIP loss during training","Increasing the number of denoising steps"], answer:1 },
      { q:"ControlNet adds spatial conditioning to Stable Diffusion using:", options:["A separate full U-Net trained from scratch","Cloned encoder blocks connected via zero-convolution adapters (preserving pretrained behaviour initially)","Modifying the text embedding","Replacing cross-attention with spatial attention"], answer:1 },
      { q:"The Dark Channel Prior for dehazing assumes that:", options:["At least one channel is saturated in hazy images","In haze-free outdoor images, at least one colour channel has very low intensity in most local patches","Haze is always grey","Transmission is uniform across the image"], answer:1 },
      { q:"Seam carving uses dynamic programming to find:", options:["The straightest line from top to bottom","The minimum-energy connected path of pixels from top to bottom","The maximum-contrast path","The path through the most uniform region"], answer:1 },
      { q:"DDIM sampling differs from DDPM by:", options:["Using a different U-Net architecture","Using a deterministic (non-Markovian) reverse process that skips timesteps","Training on higher-resolution images","Using a discriminator"], answer:1 },
      { q:"In image inpainting, LaMa excels at repeating textures because it uses:", options:["Larger kernels","Fast Fourier Convolutions (FFC) that have global receptive field from the first layer","More training data","A larger discriminator"], answer:1 },
      { q:"NIQE is a no-reference image quality metric. 'No-reference' means:", options:["It requires a low-quality reference","It estimates quality without needing a ground-truth clean reference image","It is parameter-free","It runs with no GPU"], answer:1 },
      { q:"The 'dark channel prior' works because in haze-free nature images:", options:["The sky is always bright","At least one of R,G,B is close to zero in any local patch (except sky)","All channels are equal","Contrast is always high"], answer:1 },
      { q:"Which model pioneered multi-stage progressive restoration for multiple restoration tasks (rain, blur, noise)?", options:["DnCNN","FFDNet","MPRNet","SwinIR"], answer:2 },
      { q:"In StyleGAN, the W latent space is produced by:", options:["A random noise vector z directly","A mapping network (sequence of fully-connected layers) from z to w","An encoder network","The discriminator"], answer:1 },
      { q:"FID (Frechet Inception Distance) measures:", options:["Peak signal-to-noise ratio","Frechet distance between InceptionNet feature distributions of real and generated images (lower=better)","Pixel-level accuracy","Sharpness of generated images"], answer:1 },
      { q:"Zero-DCE (Zero-Reference Deep Curve Estimation) for low-light enhancement is unique because:", options:["It requires paired low/normal light images","It requires no paired training data or even dark images; it learns curve maps unsupervised","It uses a GAN discriminator","It works only in LAB colour space"], answer:1 },
      { q:"In MPRNet, supervised attention modules at each stage do what?", options:["Replace batch normalisation","Produce intermediate restored outputs that supervise earlier stages, preventing gradient vanishing","Add spatial attention to features","Replace skip connections"], answer:1 },
      { q:"The 'inpainting by exemplar' classical method (Criminisi) fills holes by:", options:["Blurring surrounding pixels","Propagating similar texture patches from the known region, guided by structure priority","Training a GAN","Using global colour statistics"], answer:1 },
      { q:"VAE-based image generation tends to produce blurry outputs because:", options:["VAEs are undertrained","MSE reconstruction loss optimises for the expected value over all plausible reconstructions (blurry average)","The latent space is too large","Decoders are too shallow"], answer:1 },
      { q:"Restormer applies self-attention along which dimension for efficient high-resolution processing?", options:["Spatial (H, W) dimension","Channel dimension only (transposed attention)","Batch dimension","Both spatial and channel"], answer:1 },
      { q:"Total Variation (TV) regularisation in image restoration promotes images that are:", options:["High frequency and textured","Piecewise constant (smooth within regions, sharp at edges)","Uniformly blurred","Noise-amplified"], answer:1 },
    ]
  },
  {
    id:"Q5", title:"Multimodal, 3D, and Advanced Topics", color:P.accent5,
    questions:[
      { q:"NeRF represents a scene as a 5D function: inputs are (x,y,z) position and (theta,phi) view direction. Outputs are:", options:["Depth and normal","RGB colour and volume density (sigma)","Class label and confidence","Optical flow and disparity"], answer:1 },
      { q:"3D Gaussian Splatting achieves real-time rendering by:", options:["Reducing polygon count","Rasterising differentiable 3D Gaussians projected as 2D ellipses, sorted by depth","Running NeRF on a GPU cluster","Using voxel grids with fast traversal"], answer:1 },
      { q:"CLIP's zero-shot classification works by:", options:["Fine-tuning on the target dataset","Comparing image embeddings to text embeddings of class descriptions and selecting the nearest","Using a separate classifier head","Majority voting over augmented views"], answer:1 },
      { q:"VQA (Visual Question Answering) requires:", options:["Only image understanding","Only language understanding","Joint understanding of image content and natural language question","3D scene reconstruction"], answer:2 },
      { q:"In BLIP-2, the Q-Former (Querying Transformer) acts as:", options:["A text decoder","A lightweight bridge extracting relevant visual features from a frozen image encoder for an LLM","A discriminator","An image encoder"], answer:1 },
      { q:"DINOv2 achieves strong dense prediction features via:", options:["Supervised ImageNet training only","Self-supervised learning combining DINO, iBOT, and KoLeo objectives on 142M curated images","Adversarial training","Contrastive learning with text pairs"], answer:1 },
      { q:"The epipolar constraint in stereo vision states that:", options:["Corresponding points have the same depth","A 3D point's projection in one image must lie on a specific line (epipolar line) in the other image","All pixels have the same disparity","Cameras must be perfectly aligned"], answer:1 },
      { q:"PointNet achieves permutation invariance of point sets by using:", options:["Graph convolution","Shared MLP per point followed by symmetric max-pooling over all points","3D voxelisation","Attention over point pairs"], answer:1 },
      { q:"In autonomous driving, BEV (Bird's Eye View) representations are preferred for 3D detection because:", options:["They are easier to visualise","Object shapes and positions are scale-consistent and non-overlapping, unlike perspective view","They require less computation","Cameras directly capture BEV images"], answer:1 },
      { q:"InstantNGP accelerates NeRF training by replacing the MLP positional encoding with:", options:["SIREN (sinusoidal activations)","Multi-resolution hash encoding of 3D positions","Voxel grids","Fourier features"], answer:1 },
      { q:"LLaVA (Large Language and Vision Assistant) architecture connects:", options:["CLIP image encoder to GPT-2","A vision encoder (CLIP ViT) to a large language model (LLaMA/Vicuna) via a linear projection","Two CLIP models","A GAN to a text encoder"], answer:1 },
      { q:"Medical image segmentation has lower data availability than ImageNet due to:", options:["Hardware limitations","Requirement for expert clinical annotation and strict patient privacy regulations","Medical images being lower quality","Limited camera technology"], answer:1 },
      { q:"ControlNet was designed specifically to add what to pretrained diffusion models?", options:["Text conditioning","Spatial conditioning (edges, depth, pose) without modifying pretrained weights","Higher resolution","Faster sampling"], answer:1 },
      { q:"In PatchCore (anomaly detection), the memory bank stores features from:", options:["Anomalous test images","Normal training image patches (CNN features), used as reference at test time","The discriminator of a GAN","Text descriptions of defects"], answer:1 },
      { q:"FAISS (Facebook AI Similarity Search) is used in image retrieval for:", options:["Training image classifiers","Efficiently searching millions of embeddings using approximate nearest neighbour algorithms (IVF, HNSW)","Data augmentation","Depth estimation"], answer:1 },
      { q:"The Chamfer Distance is used to compare:", options:["Image quality","Two point clouds (average nearest-neighbour distance from each point in A to B and vice versa)","Segmentation masks","Optical flow fields"], answer:1 },
      { q:"In document understanding, LayoutLMv3 encodes which additional information beyond text tokens?", options:["Audio transcription","2D bounding box positions of text elements and image patch features","Video frames","3D document structure"], answer:1 },
      { q:"The SA-1B dataset used to train SAM contains approximately:", options:["11K images and 11K masks","1M images and 10M masks","11M images and 1.1B masks","100M images and 1T masks"], answer:2 },
      { q:"Which metric is used to evaluate novel view synthesis quality in NeRF papers?", options:["mAP and IOU","PSNR, SSIM, and LPIPS (measuring pixel quality, structure, and perceptual quality of rendered views)","FID and IS","Chamfer Distance only"], answer:1 },
      { q:"VGGFace2 is used to train:", options:["General image classifiers","Face recognition models (contains 3.31M images of 9131 identities)","Object detection models","Generative models for face synthesis"], answer:1 },
      { q:"SAMURAI extends SAM2 to video by adding:", options:["A larger image encoder","Kalman filter-based motion modelling for temporal consistency in single-object tracking","Text conditioning","Multi-object tracking heads"], answer:1 },
      { q:"DiT (Diffusion Transformer) replaces the U-Net backbone in diffusion models with:", options:["A ResNet backbone","A Vision Transformer operating on latent patches","A GAN discriminator","A RNN sequence model"], answer:1 },
      { q:"The P3 level of a Feature Pyramid Network (FPN) has what spatial resolution relative to the input?", options:["1/2 of input","1/8 of input","1/32 of input","Same as input"], answer:1 },
      { q:"Open-vocabulary detection (e.g. GLIP, Grounding DINO) enables detecting:", options:["Only COCO classes","Any object described by free-form text at inference, without retraining","Objects in 3D point clouds only","Anomalies in industrial images"], answer:1 },
      { q:"In monocular depth estimation, 'metric depth' refers to:", options:["Relative depth ordering only","Absolute depth in real-world units (metres), requiring calibration or learning from scale-aware data","Depth measured in pixels","Depth normalised to [0,1]"], answer:1 },
    ]
  },
  {
    id:"Q6", title:"Practical ML, Deployment, and Ethics", color:P.accent6,
    questions:[
      { q:"Post-Training Quantisation (PTQ) reduces model size without retraining. The accuracy drop for INT8 is typically:", options:["Negligible (< 1%)","Moderate (5-10%)","Severe (> 20%)","Zero"], answer:0 },
      { q:"What is model calibration in the context of deployment?", options:["Adjusting model weights for a new dataset","Ensuring that the model's confidence scores reflect true probabilities","Reducing inference latency","Increasing model throughput"], answer:1 },
      { q:"Expected Calibration Error (ECE) is computed as:", options:["Mean squared error between predictions and ground truth","Weighted average of the absolute difference between accuracy and confidence per confidence bin","1 - Top-1 accuracy","KL divergence between predicted and true distributions"], answer:1 },
      { q:"TensorRT is used to:", options:["Train models faster","Optimise and compile models specifically for NVIDIA GPU inference (kernel fusion, mixed precision, batching)","Export models to mobile","Visualise model architecture"], answer:1 },
      { q:"ONNX (Open Neural Network Exchange) is:", options:["A training framework","An open format for representing ML models, enabling cross-framework deployment","A dataset format","A GPU library"], answer:1 },
      { q:"Which of these reduces inference latency but NOT model size?", options:["Weight pruning","INT8 quantisation","Batching multiple requests together","Knowledge distillation"], answer:2 },
      { q:"Conformal prediction provides:", options:["A point estimate of model output","Distribution-free prediction sets with guaranteed coverage at a specified confidence level","A calibrated probability distribution","Uncertainty estimates only for regression tasks"], answer:1 },
      { q:"Temperature scaling for calibration works by:", options:["Adding noise to logits","Dividing all logits by a learned scalar T before softmax, found via validation set","Retraining the final layer","Applying label smoothing"], answer:1 },
      { q:"Data augmentation during training primarily helps with:", options:["Faster training","Reducing overfitting by artificially increasing training data diversity","Improving inference speed","Reducing model parameters"], answer:1 },
      { q:"Which augmentation strategy specifically improves small object detection performance?", options:["Random horizontal flip","Random erasing / cutout","Mosaic augmentation (combining 4 images, used in YOLOv5+)","Mixup"], answer:2 },
      { q:"Class Activation Mapping (CAM) and Grad-CAM are used for:", options:["Data augmentation","Visualising which image regions influenced a classification decision (interpretability)","Model compression","Speed benchmarking"], answer:1 },
      { q:"The EU AI Act classifies autonomous vehicle perception systems as:", options:["Prohibited AI","Low-risk AI","High-risk AI (critical safety applications require conformity assessment)","Minimal risk AI"], answer:2 },
      { q:"Which of these is a key challenge when deploying CV models in clinical settings?", options:["Too much data","Strict regulatory requirements (FDA, CE marking), privacy regulations (HIPAA/GDPR), and bias across patient demographics","High GPU cost","Colour calibration"], answer:1 },
      { q:"Model bias in face recognition systems has been shown to:", options:["Affect all demographic groups equally","Perform worse on darker-skinned individuals and women, reflecting dataset imbalances in training data","Only affect age estimation","Not exist in modern systems"], answer:1 },
      { q:"torch.compile() in PyTorch 2.0 improves inference speed by:", options:["Reducing model parameters","Compiling the model graph (kernel fusion, operation scheduling) ahead of time","Quantising weights","Pruning unused layers"], answer:1 },
      { q:"Deepfakes pose which primary societal risk?", options:["Lower image quality","Manipulation of public discourse through synthetic media that makes real events appear false or vice versa","Increased storage requirements","Reduced model accuracy"], answer:1 },
      { q:"Why is FLANN preferred over brute-force matching for SIFT in large-scale retrieval?", options:["Higher accuracy","Approximate nearest-neighbour search with sub-linear time complexity, much faster for large databases","Better handling of rotation invariance","Produces fewer false positives"], answer:1 },
      { q:"The principle of 'Data Minimisation' in GDPR means:", options:["Using smaller models","Only collecting and processing the minimum personal data necessary for the specified purpose","Reducing dataset size for faster training","Using fewer sensors"], answer:1 },
      { q:"Which metric is most appropriate for evaluating detection performance on a dataset with severe class imbalance?", options:["Pixel accuracy","mAP (mean Average Precision, which averages over the full precision-recall curve)","Simple accuracy","MSE"], answer:1 },
      { q:"In federated learning for medical image analysis:", options:["Patient data is centralised on one server","Models are trained locally on each hospital's data; only model updates (gradients) are shared, preserving privacy","Data is encrypted then sent to a cloud","Only inference is done locally"], answer:1 },
      { q:"Structured pruning removes entire CNN filters/channels. Its advantage over unstructured pruning is:", options:["Better accuracy preservation","Direct reduction in FLOPs and wall-clock time (hardware-friendly), unlike sparse unstructured pruning","Simpler implementation","Smaller storage savings"], answer:1 },
      { q:"Which of these is NOT a consideration when deploying an object detection model on an edge device (e.g. Raspberry Pi)?", options:["Model size (RAM constraint)","Inference latency per frame","INT8 quantisation compatibility","ImageNet top-1 accuracy (not directly relevant to deployment task)"], answer:3 },
      { q:"MixUp augmentation creates training samples by:", options:["Random cropping and pasting","Linear interpolation between two random training images and their labels (x=lambda*x1+(1-lambda)*x2)","Adding random noise","Randomly dropping patches (CutOut)"], answer:1 },
      { q:"The 'LIME' explanation method explains a model prediction by:", options:["Visualising gradients","Perturbing the input, observing prediction changes, and fitting a local linear model to the resulting neighbourhood","Computing SHAP values","Backpropagating the class score to the input"], answer:1 },
      { q:"When a CV model fails silently on out-of-distribution data, the recommended mitigation is:", options:["Retrain on more data only","Add uncertainty estimation (ensembles, MC Dropout, or conformal prediction) plus OOD detection to flag uncertain predictions","Increase model size","Reduce confidence threshold"], answer:1 },
    ]
  },
];

/* ================================================================
   REACT COMPONENT
================================================================ */
export default function App() {
  const [tab, setTab]           = useState("domains");
  const [selectedDomain, setSelectedDomain] = useState(null);
  const [domainSearch, setDomainSearch]     = useState("");
  const [selectedModule, setSelectedModule] = useState(null);
  const [chFilter, setChFilter] = useState("All");
  const [selectedCh, setSelectedCh]         = useState(null);
  const [quizSection, setQuizSection]       = useState(null);
  const [quizIdx, setQuizIdx]   = useState(0);
  const [selected, setSelected] = useState(null);
  const [score, setScore]       = useState(0);
  const [answered, setAnswered] = useState(false);
  const [showAll, setShowAll]   = useState(false);
  const [copied, setCopied]     = useState(false);
  const [quizDone, setQuizDone] = useState(false);
  const [wrongAns, setWrongAns] = useState([]);

  const filteredDomains = DOMAINS.filter(d =>
    d.name.toLowerCase().includes(domainSearch.toLowerCase()) ||
    d.tagline.toLowerCase().includes(domainSearch.toLowerCase())
  );

  const filteredChallenges = chFilter === "All"
    ? CHALLENGES
    : CHALLENGES.filter(c => c.difficulty === chFilter);

  const copyCode = (code) => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true); setTimeout(() => setCopied(false), 2000);
    });
  };

  const handleAnswer = (idx) => {
    if (answered) return;
    setSelected(idx); setAnswered(true);
    const q = quizSection.questions[quizIdx];
    if (idx === q.answer) setScore(s => s+1);
    else setWrongAns(w => [...w, { q: q.q, correct: q.options[q.answer], chosen: q.options[idx] }]);
  };

  const nextQuestion = () => {
    if (quizIdx + 1 >= quizSection.questions.length) {
      setQuizDone(true);
    } else {
      setQuizIdx(i => i+1); setSelected(null); setAnswered(false);
    }
  };

  const resetQuiz = () => {
    setQuizSection(null); setQuizIdx(0); setSelected(null);
    setScore(0); setAnswered(false); setQuizDone(false); setWrongAns([]);
  };

  /* ── Styles ── */
  const S = {
    app:    { minHeight:"100vh", background:P.bg, color:P.text, fontFamily:"'JetBrains Mono','Fira Code',monospace" },
    header: { background:`linear-gradient(135deg,${P.surface} 0%,#0a1830 100%)`,
              padding:"2rem 2rem 1.5rem", borderBottom:`1px solid ${P.border}` },
    heroTitle:{ fontSize:"clamp(1.6rem,3vw,2.6rem)", fontWeight:800, letterSpacing:"-0.02em",
                background:`linear-gradient(90deg,${P.accent1},${P.accent2},${P.accent3})`,
                WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent" },
    heroSub:{ color:P.muted, fontSize:"0.9rem", marginTop:"0.4rem" },
    stats:  { display:"flex", gap:"2rem", marginTop:"1.2rem", flexWrap:"wrap" },
    stat:   { textAlign:"center" },
    statN:  { fontSize:"1.6rem", fontWeight:800, color:P.accent1 },
    statL:  { fontSize:"0.72rem", color:P.muted, textTransform:"uppercase", letterSpacing:"0.08em" },
    tabs:   { display:"flex", gap:"0.3rem", padding:"1rem 2rem 0", background:P.surface,
              borderBottom:`1px solid ${P.border}`, overflowX:"auto" },
    tab:    (active) => ({ padding:"0.65rem 1.4rem", borderRadius:"8px 8px 0 0", cursor:"pointer",
              fontWeight:600, fontSize:"0.82rem", letterSpacing:"0.04em",
              background: active ? P.card : "transparent",
              color: active ? P.accent1 : P.muted,
              border: active ? `1px solid ${P.border}` : "1px solid transparent",
              borderBottom: active ? `1px solid ${P.card}` : "1px solid transparent",
              transition:"all 0.2s", whiteSpace:"nowrap" }),
    content:{ padding:"2rem", maxWidth:"1400px", margin:"0 auto" },
    grid3:  { display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(280px,1fr))", gap:"1.2rem" },
    grid2:  { display:"grid", gridTemplateColumns:"repeat(auto-fill,minmax(380px,1fr))", gap:"1.2rem" },
    card:   (col) => ({ background:P.card, border:`1px solid ${col}33`, borderRadius:12,
              padding:"1.4rem", cursor:"pointer", transition:"all 0.2s",
              borderLeft:`3px solid ${col}` }),
    badge:  (col) => ({ display:"inline-block", background:`${col}22`, color:col,
              borderRadius:20, padding:"0.18rem 0.7rem", fontSize:"0.72rem",
              fontWeight:700, letterSpacing:"0.06em" }),
    btn:    (col,sm) => ({ padding: sm?"0.4rem 0.9rem":"0.65rem 1.4rem",
              background:`${col}22`, color:col, border:`1px solid ${col}55`,
              borderRadius:8, cursor:"pointer", fontWeight:700,
              fontSize: sm?"0.75rem":"0.85rem", transition:"all 0.15s" }),
    input:  { background:P.card2, border:`1px solid ${P.border}`, borderRadius:8,
              color:P.text, padding:"0.6rem 1rem", fontSize:"0.85rem", width:"100%",
              outline:"none", boxSizing:"border-box" },
    code:   { background:"#020509", border:`1px solid ${P.border}`, borderRadius:8,
              padding:"1.2rem", fontSize:"0.78rem", overflowX:"auto",
              whiteSpace:"pre", lineHeight:1.65, color:"#a8d8a8", maxHeight:480,
              overflowY:"auto", display:"block" },
    panel:  { background:P.card, border:`1px solid ${P.border}`, borderRadius:14, padding:"2rem" },
    tag:    (col) => ({ display:"inline-block", background:`${col}15`, color:col,
              border:`1px solid ${col}33`, borderRadius:6, padding:"0.12rem 0.5rem",
              fontSize:"0.7rem", fontWeight:600 }),
    diffBadge: (d) => {
      const m={Easy:P.ok,Medium:P.warn,Hard:P.accent4};
      return { display:"inline-block", background:`${m[d]}22`, color:m[d],
               borderRadius:20, padding:"0.15rem 0.65rem", fontSize:"0.7rem", fontWeight:700 };
    },
    back:   { background:"none", border:"none", color:P.accent1, cursor:"pointer",
              fontSize:"0.85rem", marginBottom:"1.2rem", display:"flex", alignItems:"center", gap:"0.4rem" },
    sectionTitle: { fontSize:"1.4rem", fontWeight:800, marginBottom:"1.2rem",
                    color:P.text, letterSpacing:"-0.01em" },
    listItem: (col) => ({ background:P.card2, border:`1px solid ${col}33`, borderRadius:8,
               padding:"0.5rem 0.9rem", marginBottom:"0.4rem", fontSize:"0.82rem", color:P.muted }),
    quizOpt:  (state) => {
      const base = { padding:"0.9rem 1.2rem", borderRadius:10, cursor:"pointer",
                     marginBottom:"0.6rem", fontSize:"0.88rem", fontWeight:500,
                     border:"1px solid", transition:"all 0.15s" };
      if (state==="correct")   return {...base, background:`${P.ok}22`,    color:P.ok,    borderColor:P.ok};
      if (state==="wrong")     return {...base, background:`${P.accent4}22`,color:P.accent4,borderColor:P.accent4};
      if (state==="neutral")   return {...base, background:P.card2,         color:P.text,  borderColor:P.border};
      if (state==="unselected-correct") return {...base, background:`${P.ok}11`, color:P.ok, borderColor:`${P.ok}55`};
      return {...base, background:P.card2, color:P.text, borderColor:P.border};
    },
  };

  /* ── DOMAIN VIEW ── */
  if (selectedDomain) {
    const d = selectedDomain;
    return (
      <div style={S.app}>
        <div style={{...S.content, paddingTop:"1.5rem"}}>
          <button style={S.back} onClick={()=>setSelectedDomain(null)}>← Back to Domains</button>
          <div style={{display:"flex",alignItems:"center",gap:"1rem",marginBottom:"0.5rem"}}>
            <h1 style={{fontSize:"1.8rem",fontWeight:900,color:d.color,margin:0}}>{d.name}</h1>
            <span style={S.badge(d.color)}>{d.tagline}</span>
          </div>
          <div style={{...S.panel, marginBottom:"1.5rem"}}>
            <h3 style={{color:d.color,marginTop:0}}>Theory</h3>
            <p style={{lineHeight:1.9,color:P.muted,whiteSpace:"pre-line",fontSize:"0.9rem"}}>{d.theory}</p>
          </div>
          <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(240px,1fr))",gap:"1rem",marginBottom:"1.5rem"}}>
            {[["Key Architectures",d.architectures,d.color],["Metrics",d.metrics,P.accent2],["Datasets",d.datasets,P.accent3]].map(([title,items,col])=>(
              <div key={title} style={{...S.panel}}>
                <h4 style={{color:col,marginTop:0,marginBottom:"0.8rem"}}>{title}</h4>
                {items.map((it,i)=><div key={i} style={S.listItem(col)}>{it}</div>)}
              </div>
            ))}
          </div>
          <div style={S.panel}>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"0.8rem"}}>
              <h3 style={{color:d.color,margin:0}}>Google Colab Code</h3>
              <button style={S.btn(d.color,true)} onClick={()=>copyCode(d.colab)}>{copied?"Copied!":"Copy"}</button>
            </div>
            <code style={S.code}>{d.colab}</code>
          </div>
        </div>
      </div>
    );
  }

  /* ── MODULE VIEW ── */
  if (selectedModule) {
    const m = selectedModule;
    return (
      <div style={S.app}>
        <div style={{...S.content,paddingTop:"1.5rem"}}>
          <button style={S.back} onClick={()=>setSelectedModule(null)}>← Back to Modules</button>
          <div style={{display:"flex",alignItems:"center",gap:"1rem",marginBottom:"1rem",flexWrap:"wrap"}}>
            <h1 style={{fontSize:"1.6rem",fontWeight:900,color:m.color,margin:0}}>{m.title}</h1>
            <span style={S.badge(m.color)}>{m.level}</span>
            <span style={S.badge(P.accent6)}>{m.time}</span>
          </div>
          {m.sections.map((sec,i)=>(
            <div key={i} style={{...S.panel,marginBottom:"1.2rem"}}>
              <h3 style={{color:m.color,marginTop:0}}>{sec.heading}</h3>
              <p style={{lineHeight:1.9,color:P.muted,whiteSpace:"pre-line",fontSize:"0.88rem"}}>{sec.body}</p>
            </div>
          ))}
          <div style={S.panel}>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"0.8rem"}}>
              <h3 style={{color:m.color,margin:0}}>Hands-On Code</h3>
              <button style={S.btn(m.color,true)} onClick={()=>copyCode(m.code)}>{copied?"Copied!":"Copy"}</button>
            </div>
            <code style={S.code}>{m.code}</code>
          </div>
        </div>
      </div>
    );
  }

  /* ── CHALLENGE VIEW ── */
  if (selectedCh) {
    const ch = selectedCh;
    const diffColor = {Easy:P.ok,Medium:P.warn,Hard:P.accent4}[ch.difficulty];
    return (
      <div style={S.app}>
        <div style={{...S.content,paddingTop:"1.5rem"}}>
          <button style={S.back} onClick={()=>setSelectedCh(null)}>← Back to Challenges</button>
          <div style={{display:"flex",alignItems:"center",gap:"1rem",flexWrap:"wrap",marginBottom:"1rem"}}>
            <span style={{fontSize:"1rem",fontWeight:800,color:P.muted}}>{ch.id}</span>
            <h1 style={{fontSize:"1.5rem",fontWeight:900,color:diffColor,margin:0}}>{ch.title}</h1>
            <span style={S.diffBadge(ch.difficulty)}>{ch.difficulty}</span>
            <span style={S.tag(P.accent2)}>{ch.tag}</span>
            <span style={S.badge(P.accent6)}>{ch.points} pts</span>
          </div>
          <div style={{color:P.muted,fontSize:"0.78rem",marginBottom:"1.2rem"}}>Asked by: {ch.company}</div>
          <div style={{...S.panel,marginBottom:"1rem"}}>
            <h3 style={{color:diffColor,marginTop:0}}>Problem Statement</h3>
            <p style={{lineHeight:1.8,color:P.text,fontSize:"0.9rem"}}>{ch.desc}</p>
            {ch.example && (<>
              <h4 style={{color:P.accent3}}>Example</h4>
              <code style={{...S.code,maxHeight:160,fontSize:"0.8rem"}}>{ch.example}</code>
            </>)}
          </div>
          <div style={{...S.panel,marginBottom:"1rem",background:P.card2}}>
            <h4 style={{color:P.accent6,marginTop:0}}>Hint</h4>
            <p style={{color:P.muted,fontSize:"0.88rem",lineHeight:1.7}}>{ch.hint}</p>
          </div>
          <div style={S.panel}>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"0.8rem",flexWrap:"wrap",gap:"0.5rem"}}>
              <h3 style={{color:diffColor,margin:0}}>Full Solution (with image loading)</h3>
              <button style={S.btn(diffColor,true)} onClick={()=>copyCode(ch.fullSolution)}>{copied?"Copied!":"Copy"}</button>
            </div>
            <code style={S.code}>{ch.fullSolution}</code>
            <div style={{marginTop:"1rem",display:"grid",gridTemplateColumns:"1fr 1fr",gap:"1rem"}}>
              <div style={{background:P.card2,borderRadius:8,padding:"0.8rem"}}>
                <div style={{color:P.accent3,fontSize:"0.75rem",fontWeight:700,marginBottom:"0.3rem"}}>COMPLEXITY</div>
                <div style={{color:P.muted,fontSize:"0.82rem"}}>{ch.complexity}</div>
              </div>
              <div style={{background:P.card2,borderRadius:8,padding:"0.8rem"}}>
                <div style={{color:P.accent1,fontSize:"0.75rem",fontWeight:700,marginBottom:"0.3rem"}}>FOLLOW-UP</div>
                <div style={{color:P.muted,fontSize:"0.82rem"}}>{ch.followup}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  /* ── QUIZ FLOW ── */
  if (quizSection) {
    const q = quizSection.questions[quizIdx];
    const totalQ = quizSection.questions.length;
    if (quizDone) {
      const pct = Math.round(score/totalQ*100);
      return (
        <div style={S.app}>
          <div style={{...S.content,paddingTop:"3rem",maxWidth:600}}>
            <div style={{...S.panel,textAlign:"center"}}>
              <div style={{fontSize:"4rem",marginBottom:"0.5rem"}}>{pct>=80?"🏆":pct>=60?"🎯":"📚"}</div>
              <h2 style={{color:quizSection.color,fontSize:"1.8rem"}}>Quiz Complete!</h2>
              <div style={{fontSize:"3rem",fontWeight:900,color:pct>=80?P.ok:pct>=60?P.warn:P.accent4,margin:"1rem 0"}}>{score}/{totalQ}</div>
              <div style={{color:P.muted,marginBottom:"1.5rem"}}>{pct}% correct</div>
              {wrongAns.length>0 && (
                <div style={{textAlign:"left",marginBottom:"1.5rem"}}>
                  <div style={{color:P.accent4,fontWeight:700,marginBottom:"0.8rem",fontSize:"0.85rem"}}>Review Incorrect Answers:</div>
                  {wrongAns.map((w,i)=>(
                    <div key={i} style={{background:P.card2,borderRadius:8,padding:"0.8rem",marginBottom:"0.5rem",fontSize:"0.8rem"}}>
                      <div style={{color:P.text,marginBottom:"0.4rem"}}>{w.q}</div>
                      <div style={{color:P.accent4}}>You answered: {w.chosen}</div>
                      <div style={{color:P.ok}}>Correct: {w.correct}</div>
                    </div>
                  ))}
                </div>
              )}
              <button style={S.btn(quizSection.color)} onClick={resetQuiz}>Back to Quiz Sections</button>
            </div>
          </div>
        </div>
      );
    }
    return (
      <div style={S.app}>
        <div style={{...S.content,paddingTop:"1.5rem",maxWidth:720}}>
          <button style={S.back} onClick={resetQuiz}>← Quiz Sections</button>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"1.5rem"}}>
            <div>
              <div style={{color:quizSection.color,fontWeight:800,fontSize:"1rem"}}>{quizSection.title}</div>
              <div style={{color:P.muted,fontSize:"0.78rem"}}>Question {quizIdx+1} of {totalQ}</div>
            </div>
            <div style={{fontSize:"1.2rem",fontWeight:800,color:P.ok}}>Score: {score}</div>
          </div>
          <div style={{background:P.card2,borderRadius:6,height:6,marginBottom:"1.5rem",overflow:"hidden"}}>
            <div style={{height:"100%",width:`${(quizIdx/totalQ)*100}%`,background:quizSection.color,transition:"width 0.4s"}}/>
          </div>
          <div style={{...S.panel,marginBottom:"1.2rem"}}>
            <p style={{fontSize:"1.05rem",fontWeight:600,lineHeight:1.7,color:P.text,margin:0}}>{q.q}</p>
          </div>
          {q.options.map((opt,i)=>{
            let state="neutral";
            if (answered) {
              if (i===q.answer) state="correct";
              else if (i===selected && i!==q.answer) state="wrong";
              else state="neutral";
            }
            return (
              <div key={i} style={S.quizOpt(state)} onClick={()=>handleAnswer(i)}>
                <span style={{marginRight:"0.6rem",fontWeight:800,opacity:0.6}}>{String.fromCharCode(65+i)}.</span>
                {opt}
                {answered && i===q.answer && <span style={{float:"right"}}>✓</span>}
                {answered && i===selected && i!==q.answer && <span style={{float:"right"}}>✗</span>}
              </div>
            );
          })}
          {answered && (
            <div style={{marginTop:"1rem",textAlign:"right"}}>
              <button style={S.btn(quizSection.color)} onClick={nextQuestion}>
                {quizIdx+1>=totalQ?"See Results":"Next Question →"}
              </button>
            </div>
          )}
        </div>
      </div>
    );
  }

  /* ── MAIN LAYOUT ── */
/* ── MAIN LAYOUT ── */
return (
  <div style={S.app}>
    
    {/* Header */}
    <div style={S.header}>
      <div style={{maxWidth:1400, margin:"0 auto"}}>

        {/* Brand */}
        <div style={{
          color:P.amber,
          fontSize:14,
          fontWeight:700,
          letterSpacing:"0.18em",
          textTransform:"uppercase",
          marginBottom:18
        }}>
          Daintymindz Laboratory
        </div>

        {/* Main Title */}
        <div style={S.heroTitle}>
          Computer Vision Mastery
        </div>

        {/* Subtitle */}
        <div style={{
          fontSize:22,
          lineHeight:1.6,
          color:P.textSoft,
          maxWidth:1100,
          marginTop:18,
          fontWeight:500
        }}>
          A comprehensive beginner to advanced computer vision curriculum developed by Daintymindz Laboratory. This academy is designed to teach the foundations, mathematics, engineering principles, coding practices, research concepts, and modern deep learning systems that power real world computer vision applications across healthcare, robotics, autonomous systems, agriculture, satellite imaging, smart cities, industrial inspection, biometrics, multimodal AI, and intelligent digital twin systems.
        </div>

        {/* Description */}
        <div style={{
          marginTop:32,
          maxWidth:1250,
          color:P.muted,
          fontSize:16,
          lineHeight:1.9
        }}>
          This tutorial series combines theoretical intuition, detailed Python implementations, interactive coding exercises, visual learning, research oriented explanations, and industry focused projects into a unified educational experience. Learners progress from understanding pixels and image representations to building advanced systems such as convolutional neural networks, object detectors, segmentation models, transformers, multimodal vision systems, explainable AI pipelines, and modern generative vision architectures.

          <br /><br />

          The curriculum is intentionally structured for absolute beginners, intermediate learners, researchers, interns, and engineering professionals who want a practical and deeply detailed understanding of computer vision. Every section includes carefully explained code walkthroughs, real datasets, challenge driven learning, quiz based assessments, and progressively difficult problem solving exercises inspired by real technical interviews, research workflows, and production AI systems.

          <br /><br />

          Topics covered include classical computer vision, image preprocessing, feature extraction, edge detection, segmentation, object detection, image restoration, optical flow, 3D vision, medical imaging, satellite imagery, OCR, video understanding, multimodal AI, explainable computer vision, generative AI for vision, autonomous driving perception systems, visual language models, and advanced deep learning architectures using PyTorch, OpenCV, NumPy, and modern AI frameworks.

          <br /><br />

          Developed by Daintymindz Laboratory, this platform reflects our mission of advancing intelligent systems research, engineering education, innovation, scientific rigor, and accessible AI learning for global communities.
        </div>

        {/* Stats */}
        <div style={S.stats}>
          {[
            ["30+","Computer Vision Domains",P.amber],
            ["10+","Detailed Learning Modules",P.amberSoft],
            ["100+","Coding Challenges",P.graphite3],
            ["150+","Quiz Questions",P.graphite2]
          ].map(([n,l,c]) => (
            <div key={l} style={S.stat}>
              <div style={{...S.statN, color:c}}>
                {n}
              </div>
              <div style={S.statL}>
                {l}
              </div>
            </div>
          ))}
        </div>

      </div>
    </div>

      {/* Tabs */}
      <div style={S.tabs}>
        {[["domains","30 Domains"],["modules","10 Modules"],["challenges","100 Challenges"],["quiz","150 Quizzes"]].map(([id,label])=>(
          <button key={id} style={S.tab(tab===id)} onClick={()=>setTab(id)}>{label}</button>
        ))}
      </div>

      <div style={S.content}>

        {/* DOMAINS TAB */}
        {tab==="domains" && (
          <>
            <div style={{marginBottom:"1.5rem"}}>
              <input style={S.input} placeholder="Search domains (e.g. segmentation, restoration, GAN)..."
                value={domainSearch} onChange={e=>setDomainSearch(e.target.value)}/>
            </div>
            <div style={S.grid3}>
              {filteredDomains.map(d=>(
                <div key={d.id} style={S.card(d.color)} onClick={()=>setSelectedDomain(d)}
                  onMouseEnter={e=>{e.currentTarget.style.transform="translateY(-3px)";e.currentTarget.style.boxShadow=`0 8px 24px ${d.color}22`}}
                  onMouseLeave={e=>{e.currentTarget.style.transform="";e.currentTarget.style.boxShadow=""}}>
                  <div style={{display:"flex",justifyContent:"space-between",alignItems:"start",marginBottom:"0.8rem"}}>
                    <span style={{fontSize:"0.72rem",color:P.muted,fontWeight:700}}>Domain #{d.id}</span>
                    <span style={S.badge(d.color)}>theory + code</span>
                  </div>
                  <h3 style={{margin:"0 0 0.4rem",color:d.color,fontSize:"1rem",fontWeight:800}}>{d.name}</h3>
                  <p style={{margin:"0 0 1rem",color:P.muted,fontSize:"0.8rem",lineHeight:1.5}}>{d.tagline}</p>
                  <div style={{display:"flex",flexWrap:"wrap",gap:"0.3rem"}}>
                    {d.architectures.slice(0,3).map(a=><span key={a} style={S.tag(d.color)}>{a}</span>)}
                    {d.architectures.length>3 && <span style={S.tag(d.color)}>+{d.architectures.length-3} more</span>}
                  </div>
                </div>
              ))}
            </div>
            {filteredDomains.length===0 && (
              <div style={{textAlign:"center",color:P.muted,padding:"3rem"}}>No domains match "{domainSearch}"</div>
            )}
          </>
        )}

        {/* MODULES TAB */}
        {tab==="modules" && (
          <>
            <h2 style={S.sectionTitle}>Learning Modules</h2>
            <div style={S.grid2}>
              {MODULES.map((m,i)=>(
                <div key={m.id} style={S.card(m.color)} onClick={()=>setSelectedModule(m)}
                  onMouseEnter={e=>{e.currentTarget.style.transform="translateY(-3px)"}}
                  onMouseLeave={e=>{e.currentTarget.style.transform=""}}>
                  <div style={{display:"flex",justifyContent:"space-between",marginBottom:"0.8rem",alignItems:"center"}}>
                    <span style={{fontSize:"0.72rem",color:P.muted,fontWeight:700}}>Module {i}</span>
                    <span style={S.badge(m.color)}>{m.time}</span>
                  </div>
                  <h3 style={{margin:"0 0 0.4rem",color:m.color,fontSize:"1rem",fontWeight:800}}>{m.title}</h3>
                  <div style={{marginBottom:"0.8rem"}}><span style={S.tag(m.color)}>{m.level}</span></div>
                  <div style={{color:P.muted,fontSize:"0.78rem"}}>{m.sections.length} sections • Hands-on code included</div>
                </div>
              ))}
            </div>
          </>
        )}

        {/* CHALLENGES TAB */}
        {tab==="challenges" && (
          <>
            <div style={{display:"flex",gap:"0.6rem",marginBottom:"1.5rem",flexWrap:"wrap",alignItems:"center"}}>
              <span style={{color:P.muted,fontSize:"0.82rem",fontWeight:600}}>Filter:</span>
              {["All","Easy","Medium","Hard"].map(f=>(
                <button key={f} style={S.btn(f==="Easy"?P.ok:f==="Medium"?P.warn:f==="Hard"?P.accent4:P.accent1,true)}
                  onClick={()=>setChFilter(f)}>{f} {f!=="All"?`(${CHALLENGES.filter(c=>c.difficulty===f).length})`:""}</button>
              ))}
              <span style={{marginLeft:"auto",color:P.muted,fontSize:"0.8rem"}}>Showing {filteredChallenges.length} challenges</span>
            </div>
            <div style={S.grid3}>
              {(showAll?filteredChallenges:filteredChallenges.slice(0,30)).map(ch=>{
                const diffColor={Easy:P.ok,Medium:P.warn,Hard:P.accent4}[ch.difficulty];
                return (
                  <div key={ch.id} style={S.card(diffColor)} onClick={()=>setSelectedCh(ch)}
                    onMouseEnter={e=>{e.currentTarget.style.transform="translateY(-3px)"}}
                    onMouseLeave={e=>{e.currentTarget.style.transform=""}}>
                    <div style={{display:"flex",justifyContent:"space-between",marginBottom:"0.6rem",alignItems:"center"}}>
                      <span style={{fontSize:"0.7rem",color:P.muted,fontWeight:700}}>{ch.id}</span>
                      <div style={{display:"flex",gap:"0.4rem",alignItems:"center"}}>
                        <span style={S.diffBadge(ch.difficulty)}>{ch.difficulty}</span>
                        <span style={S.badge(P.accent6)}>{ch.points}pt</span>
                      </div>
                    </div>
                    <h3 style={{margin:"0 0 0.4rem",color:diffColor,fontSize:"0.92rem",fontWeight:800}}>{ch.title}</h3>
                    <div style={{marginBottom:"0.6rem"}}><span style={S.tag(P.accent2)}>{ch.tag}</span></div>
                    <div style={{color:P.muted,fontSize:"0.75rem",lineHeight:1.4}}>{ch.desc.slice(0,90)}...</div>
                    <div style={{marginTop:"0.8rem",color:P.muted,fontSize:"0.7rem"}}>🏢 {ch.company}</div>
                  </div>
                );
              })}
            </div>
            {!showAll && filteredChallenges.length>30 && (
              <div style={{textAlign:"center",marginTop:"2rem"}}>
                <button style={S.btn(P.accent1)} onClick={()=>setShowAll(true)}>
                  Show All {filteredChallenges.length} Challenges
                </button>
              </div>
            )}
          </>
        )}

        {/* QUIZ TAB */}
        {tab==="quiz" && (
          <>
            <h2 style={S.sectionTitle}>Quiz Sections</h2>
            <p style={{color:P.muted,marginBottom:"1.5rem",fontSize:"0.88rem"}}>
              Test your knowledge across all areas of computer vision. 25 questions per section, 150 total.
            </p>
            <div style={S.grid2}>
              {QUIZ_SECTIONS.map(sec=>(
                <div key={sec.id} style={S.card(sec.color)}
                  onMouseEnter={e=>{e.currentTarget.style.transform="translateY(-3px)"}}
                  onMouseLeave={e=>{e.currentTarget.style.transform=""}}>
                  <div style={{display:"flex",justifyContent:"space-between",marginBottom:"0.8rem"}}>
                    <span style={S.badge(sec.color)}>25 Questions</span>
                  </div>
                  <h3 style={{margin:"0 0 0.8rem",color:sec.color,fontSize:"1.05rem",fontWeight:800}}>{sec.title}</h3>
                  <div style={{color:P.muted,fontSize:"0.8rem",marginBottom:"1.2rem",lineHeight:1.5}}>
                    {sec.id==="Q1" && "Pixels, colour spaces, filters, classical features, and basic CV algorithms."}
                    {sec.id==="Q2" && "CNNs, ResNet, transformers, GANs, diffusion models, and training techniques."}
                    {sec.id==="Q3" && "Object detection, segmentation, pose estimation, tracking, and autonomous driving."}
                    {sec.id==="Q4" && "Image restoration, super-resolution, dehazing, inpainting, and generative models."}
                    {sec.id==="Q5" && "NeRF, 3D reconstruction, VQA, multimodal models, and point clouds."}
                    {sec.id==="Q6" && "Deployment, quantisation, calibration, ethics, bias, and practical considerations."}
                  </div>
                  <button style={S.btn(sec.color)} onClick={()=>{setQuizSection(sec);setQuizIdx(0);setScore(0);setAnswered(false);setSelected(null);setQuizDone(false);setWrongAns([]);}}>
                    Start Quiz →
                  </button>
                </div>
              ))}
            </div>
          </>
        )}

      </div>
    </div>
  );
}
