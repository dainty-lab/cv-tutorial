import { useState, useRef } from "react";

const P = {
  bg:"#06080f", surface:"#0c1120", card:"#111827", card2:"#162035",
  accent1:"#38bdf8", accent2:"#818cf8", accent3:"#34d399",
  accent4:"#f472b6", accent5:"#fb923c", accent6:"#facc15", accent7:"#a78bfa",
  text:"#e2e8f0", muted:"#64748b", border:"#1e2d45", ok:"#34d399", warn:"#fb923c",
};
const CL = [P.accent1,P.accent2,P.accent3,P.accent4,P.accent5,P.accent6,P.accent7];
const C = i => CL[i%CL.length];

/* ─────────────────────────────────────────────────────────────
   CODING CHALLENGES  (LeetCode-style, CV-flavoured)
───────────────────────────────────────────────────────────── */
const CHALLENGES = [
  {
    id:"CH01", difficulty:"Easy", tag:"Arrays / Pixels", title:"Flip Image",
    company:"Google, Meta (frequently asked)",
    desc:`Given a binary image represented as a 2-D list of 0s and 1s, horizontally flip it (reverse each row), then invert it (0→1, 1→0). Return the result.`,
    example:`Input:  [[1,1,0],[1,0,1],[0,0,0]]
Output: [[1,0,0],[0,1,0],[1,1,1]]`,
    hint:"Step 1 – reverse. Step 2 – XOR each element with 1. Can be done in a single list comprehension.",
    solution:`def flipAndInvertImage(image):
    # XOR with 1 flips bit; [::-1] reverses the row
    return [[x ^ 1 for x in row[::-1]] for row in image]

# Test
img = [[1,1,0],[1,0,1],[0,0,0]]
print(flipAndInvertImage(img))
# [[1,0,0],[0,1,0],[1,1,1]]`,
    complexity:"Time O(m·n)  Space O(1) in-place variant",
    followup:"How would you extend this to an RGB image (3-channel numpy array) in one line?",
  },
  {
    id:"CH02", difficulty:"Easy", tag:"2-D Prefix Sum", title:"Number of Black Pixels in a Rectangle",
    company:"Amazon, Bloomberg",
    desc:`Given a binary image matrix, and a list of queries [row1,col1,row2,col2], return the count of 1s (black pixels) inside each query rectangle in O(1) per query after O(m·n) preprocessing.`,
    example:`Matrix:
  1 0 1
  0 1 0
  1 1 1
Query [0,0,2,2] → 6`,
    hint:"Build a 2-D prefix sum table. answer = pre[r2+1][c2+1] - pre[r1][c2+1] - pre[r2+1][c1] + pre[r1][c1]",
    solution:`def build_prefix(mat):
    m, n = len(mat), len(mat[0])
    pre = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            pre[i+1][j+1] = mat[i][j] + pre[i][j+1] + pre[i+1][j] - pre[i][j]
    return pre

def query(pre, r1, c1, r2, c2):
    return pre[r2+1][c2+1] - pre[r1][c2+1] - pre[r2+1][c1] + pre[r1][c1]

mat = [[1,0,1],[0,1,0],[1,1,1]]
pre = build_prefix(mat)
print(query(pre, 0, 0, 2, 2))  # 6
print(query(pre, 0, 0, 1, 1))  # 2`,
    complexity:"Time O(m·n) build + O(1) per query  Space O(m·n)",
    followup:"How does this relate to Integral Images used in Haar Cascade (Viola-Jones) face detection?",
  },
  {
    id:"CH03", difficulty:"Medium", tag:"BFS / Flood Fill", title:"Flood Fill",
    company:"Amazon, Google (direct LeetCode 733)",
    desc:`Implement the flood-fill algorithm (like paint bucket in Photoshop). Given an image, a starting pixel (sr, sc), and a new colour, repaint all 4-connected pixels of the same original colour.`,
    example:`Input:  image=[[1,1,1],[1,1,0],[1,0,1]] sr=1 sc=1 color=2
Output: [[2,2,2],[2,2,0],[2,0,1]]`,
    hint:"BFS or recursive DFS from (sr,sc). Track original colour. Avoid revisiting.",
    solution:`from collections import deque

def floodFill(image, sr, sc, color):
    orig = image[sr][sc]
    if orig == color:
        return image
    m, n = len(image), len(image[0])
    q = deque([(sr, sc)])
    image[sr][sc] = color
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0<=nr<m and 0<=nc<n and image[nr][nc]==orig:
                image[nr][nc] = color
                q.append((nr, nc))
    return image

print(floodFill([[1,1,1],[1,1,0],[1,0,1]], 1, 1, 2))`,
    complexity:"Time O(m·n)  Space O(m·n)",
    followup:"How would you handle 8-connectivity (diagonals) instead of 4-connectivity?",
  },
  {
    id:"CH04", difficulty:"Medium", tag:"Sliding Window / Convolution", title:"Image Smoother",
    company:"Microsoft, Apple",
    desc:`Apply a 3×3 mean (box) filter to a grayscale image. Each output pixel is the floor of the average of its neighbourhood (excluding out-of-bounds cells). Implement WITHOUT using NumPy convolution helpers.`,
    example:`Input:  [[1,1,1],[1,0,1],[1,1,1]]
Output: [[0,0,0],[0,0,0],[0,0,0]]  (all floors to 0 since avg<1 for corners)`,
    hint:"Iterate each cell, gather valid neighbours (clip to bounds), compute floor(mean).",
    solution:`import math

def imageSmoother(img):
    m, n = len(img), len(img[0])
    res = [[0]*n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            total, count = 0, 0
            for di in [-1,0,1]:
                for dj in [-1,0,1]:
                    ni, nj = i+di, j+dj
                    if 0<=ni<m and 0<=nj<n:
                        total += img[ni][nj]; count += 1
            res[i][j] = total // count
    return res

# Verify with numpy
import numpy as np
img = [[100,200,100],[200,50,200],[100,200,100]]
print(imageSmoother(img))

# Compare to scipy
from scipy.ndimage import uniform_filter
arr = np.array(img, dtype=float)
print(uniform_filter(arr, size=3, mode='constant').astype(int))`,
    complexity:"Time O(m·n·9) = O(m·n)  Space O(m·n)",
    followup:"How would you implement a Gaussian filter instead? What kernel weights would you use?",
  },
  {
    id:"CH05", difficulty:"Medium", tag:"Sorting / NMS", title:"Non-Maximum Suppression",
    company:"Tesla, Waymo (onsite coding)",
    desc:`Implement Non-Maximum Suppression (NMS) from scratch. Given bounding boxes [[x1,y1,x2,y2], ...] and confidence scores, remove duplicate boxes with IoU > threshold, keeping the highest-confidence box.`,
    example:`boxes  = [[0,0,10,10],[1,1,11,11],[20,20,30,30]]
scores = [0.9, 0.75, 0.8]
thresh = 0.5
→ keep boxes 0 and 2  (box 1 overlaps heavily with box 0)`,
    hint:"Sort by score desc. Greedily keep box, suppress remaining boxes with IoU > thresh.",
    solution:`def iou(a, b):
    xi1, yi1 = max(a[0],b[0]), max(a[1],b[1])
    xi2, yi2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    aA = (a[2]-a[0]) * (a[3]-a[1])
    aB = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (aA + aB - inter + 1e-6)

def nms(boxes, scores, threshold=0.5):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou(boxes[i], boxes[j]) < threshold]
    return keep

boxes  = [[0,0,10,10],[1,1,11,11],[20,20,30,30]]
scores = [0.9, 0.75, 0.8]
print("Kept:", nms(boxes, scores, 0.5))  # [0, 2]`,
    complexity:"Time O(n²)  Space O(n)  — torchvision uses O(n log n) GPU sort",
    followup:"How does Soft-NMS differ? When would you use it? (hint: dense pedestrian crowds)",
  },
  {
    id:"CH06", difficulty:"Medium", tag:"Binary Search / Histograms", title:"Otsu Threshold from Scratch",
    company:"NVIDIA, Qualcomm",
    desc:`Implement Otsu's method to automatically find the optimal binary threshold for a grayscale image. Maximize inter-class variance. Return the threshold value (0-255).`,
    example:`Histogram with clear bimodal distribution → threshold ≈ midpoint of the two peaks.`,
    hint:"For each threshold t, compute weight and mean of background (pixels ≤ t) and foreground (pixels > t). Maximize: w0*w1*(μ0-μ1)²",
    solution:`import numpy as np

def otsu_threshold(img_gray):
    hist = np.bincount(img_gray.ravel(), minlength=256).astype(float)
    total = hist.sum()
    best_t, best_var = 0, 0
    w0, sum0 = 0.0, 0.0
    total_sum = sum(i * hist[i] for i in range(256))
    
    for t in range(256):
        w0 += hist[t]
        if w0 == 0: continue
        w1 = total - w0
        if w1 == 0: break
        sum0 += t * hist[t]
        mu0 = sum0 / w0
        mu1 = (total_sum - sum0) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t

# Verify against OpenCV
import cv2, urllib.request
import numpy as np
from PIL import Image
urllib.request.urlretrieve(
  "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg","dog.jpg")
gray = np.array(Image.open("dog.jpg").convert("L"))
my_t = otsu_threshold(gray)
_, cv_t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"My Otsu: {my_t}   OpenCV Otsu: {int(cv_t)}")
assert abs(my_t - cv_t) <= 1`,
    complexity:"Time O(256 + H·W)  Space O(256)",
    followup:"What is multi-level Otsu (Kapur's method)? When does Otsu fail, and why?",
  },
  {
    id:"CH07", difficulty:"Medium", tag:"Connected Components", title:"Count Connected Components in Binary Image",
    company:"Citadel, Bloomberg (image processing round)",
    desc:`Given a binary image, count the number of connected components (4-connected). Each connected region of 1s is one component. Return the count.`,
    example:`[[1,1,0,0],
 [1,0,0,1],
 [0,0,1,1],
 [0,0,1,0]]  →  3 components`,
    hint:"Union-Find OR BFS/DFS from each unvisited 1. Mark visited. Increment counter.",
    solution:`def count_components(grid):
    m, n = len(grid), len(grid[0])
    visited = [[False]*n for _ in range(m)]
    count = 0
    
    def bfs(r, c):
        from collections import deque
        q = deque([(r,c)]); visited[r][c] = True
        while q:
            x, y = q.popleft()
            for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if 0<=nx<m and 0<=ny<n and not visited[nx][ny] and grid[nx][ny]==1:
                    visited[nx][ny] = True; q.append((nx,ny))
    
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1 and not visited[i][j]:
                bfs(i, j); count += 1
    return count

grid = [[1,1,0,0],[1,0,0,1],[0,0,1,1],[0,0,1,0]]
print(count_components(grid))  # 3

# Verify with scipy
from scipy import ndimage
import numpy as np
_, n = ndimage.label(np.array(grid))
print("scipy:", n)`,
    complexity:"Time O(m·n)  Space O(m·n)",
    followup:"cv2.connectedComponentsWithStats() returns label, count, centroids, bounding boxes. Use it to filter small noise blobs.",
  },
  {
    id:"CH08", difficulty:"Hard", tag:"Dynamic Programming", title:"Seam Carving (Content-Aware Resizing)",
    company:"Adobe (classic CV interview question)",
    desc:`Implement seam carving to reduce image width by 1. Find the vertical seam (one pixel per row) with minimum cumulative energy, then remove it. Energy = sum of absolute pixel gradient magnitudes.`,
    example:`5×3 grayscale image → find lowest-energy vertical path from top to bottom → remove 1 pixel per row.`,
    hint:"1. Compute energy map. 2. DP: dp[i][j] = energy[i][j] + min(dp[i-1][j-1], dp[i-1][j], dp[i-1][j+1]). 3. Traceback minimum seam. 4. Remove.",
    solution:`import numpy as np

def seam_carve(img_gray):
    # img_gray: 2D numpy array (H x W)
    H, W = img_gray.shape
    img = img_gray.astype(float)
    
    # 1. Energy map (gradient magnitude)
    dy = np.abs(np.roll(img, 1, 0) - np.roll(img, -1, 0))
    dx = np.abs(np.roll(img, 1, 1) - np.roll(img, -1, 1))
    energy = dx + dy
    
    # 2. DP cumulative energy (top-down)
    dp = energy.copy()
    for i in range(1, H):
        for j in range(W):
            lo = max(0, j-1); hi = min(W-1, j+1)
            dp[i, j] += dp[i-1, lo:hi+1].min()
    
    # 3. Traceback — find minimum seam
    seam = []
    j = dp[-1].argmin()
    for i in range(H-1, -1, -1):
        seam.append(j)
        if i > 0:
            lo = max(0, j-1); hi = min(W-1, j+1)
            j = lo + dp[i-1, lo:hi+1].argmin()
    seam = seam[::-1]
    
    # 4. Remove seam
    out = np.zeros((H, W-1), dtype=img_gray.dtype)
    for i, col in enumerate(seam):
        out[i] = np.delete(img_gray[i], col)
    return out, seam

# Demo
from PIL import Image
import urllib.request, matplotlib.pyplot as plt
urllib.request.urlretrieve(
  "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg","dog.jpg")
img = np.array(Image.open("dog.jpg").convert("L"))
# Remove 20 seams
result = img.copy()
for _ in range(20):
    result, _ = seam_carve(result)
print(f"Original: {img.shape}  After carving: {result.shape}")`,
    complexity:"Time O(H·W) per seam  Space O(H·W)",
    followup:"How would you vectorise the DP step with numpy.minimum? How do you extend to height reduction?",
  },
  {
    id:"CH09", difficulty:"Hard", tag:"Matrix / Transforms", title:"Rotate Image 90°",
    company:"Amazon, Apple, Google (LeetCode 48)",
    desc:`Rotate an n×n image matrix 90° clockwise IN-PLACE. Extend to arbitrary angle using affine transforms.`,
    example:`Input:  [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]`,
    hint:"Transpose (swap rows/cols), then reverse each row. For arbitrary angles, use an affine rotation matrix.",
    solution:`def rotate90(matrix):
    n = len(matrix)
    # Step 1: Transpose
    for i in range(n):
        for j in range(i+1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # Step 2: Reverse each row
    for row in matrix:
        row.reverse()
    return matrix

m = [[1,2,3],[4,5,6],[7,8,9]]
print(rotate90(m))  # [[7,4,1],[8,5,2],[9,6,3]]

# Affine rotation (arbitrary angle) with OpenCV
import cv2, numpy as np
from PIL import Image
import urllib.request
urllib.request.urlretrieve(
  "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg","dog.jpg")
img = cv2.imread("dog.jpg")
h, w = img.shape[:2]
for angle in [30, 45, 90, 180]:
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, scale=1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    print(f"Rotated {angle}°: shape {rotated.shape}")`,
    complexity:"Time O(n²) in-place  Space O(1)",
    followup:"What's the difference between warpAffine and warpPerspective in OpenCV?",
  },
  {
    id:"CH10", difficulty:"Hard", tag:"Convolution / Deep Learning", title:"Implement Conv2D Forward Pass",
    company:"FAANG ML Engineering, NVIDIA (systems round)",
    desc:`Implement a single 2D convolution forward pass (no padding, stride=1) in pure Python/NumPy. Then compare speed with torch.nn.Conv2d on the same inputs.`,
    example:`Input:  (1, 1, 5, 5) tensor  — batch=1, channels=1, H=5, W=5
Kernel: (1, 1, 3, 3)         — out_ch=1, in_ch=1, kH=3, kW=3
Output: (1, 1, 3, 3)         — valid convolution (no padding)`,
    hint:"4 nested loops: out_h × out_w × kernel_h × kernel_w. Output = sum(input_patch * kernel) + bias.",
    solution:`import numpy as np
import time

def conv2d_numpy(x, w, b):
    # x: (N,C,H,W)  w: (F,C,kH,kW)  b: (F,)
    N, C, H, W = x.shape
    F, _, kH, kW = w.shape
    oH, oW = H - kH + 1, W - kW + 1
    out = np.zeros((N, F, oH, oW))
    for n in range(N):
        for f in range(F):
            for i in range(oH):
                for j in range(oW):
                    patch = x[n, :, i:i+kH, j:j+kW]
                    out[n, f, i, j] = np.sum(patch * w[f]) + b[f]
    return out

# Speed comparison
import torch, torch.nn as nn

x_np = np.random.randn(1, 1, 28, 28).astype(np.float32)
w_np = np.random.randn(8, 1, 3, 3).astype(np.float32)
b_np = np.zeros(8, dtype=np.float32)

t0 = time.time()
out_np = conv2d_numpy(x_np, w_np, b_np)
print(f"NumPy conv: {time.time()-t0:.3f}s  shape={out_np.shape}")

x_t = torch.from_numpy(x_np)
conv = nn.Conv2d(1, 8, 3, bias=True)
conv.weight.data = torch.from_numpy(w_np)
conv.bias.data   = torch.from_numpy(b_np)

t0 = time.time()
with torch.no_grad():
    out_t = conv(x_t)
print(f"PyTorch conv: {time.time()-t0:.4f}s  shape={out_t.shape}")
print(f"Max diff: {np.abs(out_np - out_t.numpy()).max():.6f}")`,
    complexity:"Time O(N·F·C·oH·oW·kH·kW)  Space O(N·F·oH·oW)",
    followup:"What is im2col? How does it convert convolution into matrix multiplication (GEMM) for GPU efficiency?",
  },
  {
    id:"CH11", difficulty:"Medium", tag:"Hashing / Feature Matching", title:"Image Hash Similarity (pHash)",
    company:"Dropbox, Pinterest (duplicate detection)",
    desc:`Implement perceptual hashing (pHash) to detect near-duplicate images. Two images with Hamming distance ≤ 10 on their 64-bit hash are considered similar.`,
    example:`Original image and its slightly brightened version → Hamming distance < 5 (similar).
Original image and a completely different image → Hamming distance > 30 (different).`,
    hint:"1. Resize to 32×32 grayscale. 2. Apply DCT. 3. Keep top-left 8×8 coefficients. 4. Threshold by mean → 64-bit hash.",
    solution:`import numpy as np
from scipy.fft import dctn
from PIL import Image
import urllib.request

def phash(img_pil, size=32, hash_size=8):
    # 1. Resize to size×size, convert to grayscale
    img = img_pil.convert("L").resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=float)
    
    # 2. Discrete Cosine Transform 2D
    dct = dctn(arr, norm='ortho')
    
    # 3. Take top-left hash_size×hash_size (low frequencies)
    low_freq = dct[:hash_size, :hash_size].flatten()
    
    # 4. Binarize by median
    median = np.median(low_freq)
    return (low_freq > median).astype(np.uint8)

def hamming(h1, h2):
    return int(np.sum(h1 != h2))

# Test
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg"
urllib.request.urlretrieve(url, "dog.jpg")
original = Image.open("dog.jpg")

from PIL import ImageEnhance, ImageFilter
bright   = ImageEnhance.Brightness(original).enhance(1.3)
blurred  = original.filter(ImageFilter.GaussianBlur(2))
diff_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"
urllib.request.urlretrieve(diff_url, "ant.jpg")
different = Image.open("ant.jpg")

h_orig = phash(original)
for name, img in [("Brightened", bright), ("Blurred", blurred), ("Different image", different)]:
    d = hamming(h_orig, phash(img))
    sim = "SIMILAR ✅" if d <= 10 else "DIFFERENT ❌"
    print(f"{name:20s} Hamming={d:2d}  {sim}")`,
    complexity:"Time O(n² log n) for DCT  Space O(n²)",
    followup:"Compare pHash, dHash (difference hash), and aHash. Which is fastest? Which is most robust to JPEG compression?",
  },
  {
    id:"CH12", difficulty:"Hard", tag:"Graph / Shortest Path", title:"Shortest Path in Binary Image (BFS)",
    company:"Google, Uber (LeetCode 1091 variant)",
    desc:`In an n×n binary grid, find the length of the shortest clear path from (0,0) to (n-1,n-1). A clear path uses only 0-cells and can move in 8 directions. Return -1 if no clear path.`,
    example:`grid = [[0,1],[1,0]]  →  2  (path: (0,0)→(1,1))`,
    hint:"BFS guarantees shortest path on unweighted grids. Use 8-directional neighbours. Distance = steps + 1.",
    solution:`from collections import deque

def shortestPathBinaryMatrix(grid):
    n = len(grid)
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1
    if n == 1:
        return 1
    
    q = deque([(0, 0, 1)])  # row, col, dist
    visited = {(0, 0)}
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    
    while q:
        r, c, d = q.popleft()
        for dr, dc in dirs:
            nr, nc = r+dr, c+dc
            if 0<=nr<n and 0<=nc<n and (nr,nc) not in visited and grid[nr][nc]==0:
                if nr == n-1 and nc == n-1:
                    return d + 1
                visited.add((nr,nc))
                q.append((nr, nc, d+1))
    return -1

# Tests
print(shortestPathBinaryMatrix([[0,0,0],[1,1,0],[1,1,0]]))  # 4
print(shortestPathBinaryMatrix([[0,1],[1,0]]))               # 2
print(shortestPathBinaryMatrix([[1,0,0],[1,1,0],[1,1,0]]))  # -1

# Visual: relate to CV path planning in robotics
print("\\nIn robotics CV: 0=free space, 1=obstacle, BFS = wavefront planner")`,
    complexity:"Time O(n²)  Space O(n²)",
    followup:"How would A* with Euclidean heuristic improve this? When does BFS beat A* in practice?",
  },
];

/* ─────────────────────────────────────────────────────────────
   REAL-WORLD PROJECTS  (hands-on, real data)
───────────────────────────────────────────────────────────── */
const PROJECTS = [
  {
    id:"P01", title:"Build a Plant Disease Classifier", emoji:"🌿",
    color:P.accent3, difficulty:"Beginner", time:"2–3 hrs",
    dataset:"PlantVillage (54,306 real leaf images, 38 classes)",
    datasetUrl:"https://www.kaggle.com/datasets/emmarex/plantdisease",
    skills:["Image classification","Transfer learning","EfficientNet","Data augmentation"],
    outcome:"App that classifies tomato / maize leaf diseases with >95% accuracy",
    steps:[
      "Download PlantVillage from Kaggle (free)",
      "Explore class distribution & visualise sample images",
      "Apply augmentation: flip, rotate, colour jitter",
      "Fine-tune EfficientNet-B0 (pretrained on ImageNet)",
      "Evaluate: confusion matrix, per-class F1",
      "Export model to ONNX for deployment",
    ],
    code:`# ══ CELL 1 — Download & Mount Data ══════════════════════════
# Option A: Kaggle API
!pip install -q kaggle
# Upload your kaggle.json, then:
# !kaggle datasets download -d emmarex/plantdisease --unzip

# Option B: Use torchvision built-in subset (faster for demo)
!pip install -q torch torchvision matplotlib scikit-learn

import torch, torchvision, torch.nn as nn
import torchvision.transforms as T
from torchvision import models, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt, numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ══ CELL 2 — Load Dataset ════════════════════════════════════
DATA_DIR = "/content/PlantVillage"   # adjust to your path

transform_train = T.Compose([
    T.Resize(256), T.RandomCrop(224),
    T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
    T.ColorJitter(0.3, 0.3, 0.3, 0.1),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
transform_val = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

full_ds = datasets.ImageFolder(DATA_DIR)
NUM_CLASSES = len(full_ds.classes)
print(f"Classes: {NUM_CLASSES}  |  Total images: {len(full_ds)}")

n_val = int(len(full_ds) * 0.2)
train_ds, val_ds = random_split(full_ds, [len(full_ds)-n_val, n_val])
train_ds.dataset.transform = transform_train
val_ds.dataset.transform   = transform_val

train_dl = DataLoader(train_ds, 32, shuffle=True,  num_workers=2)
val_dl   = DataLoader(val_ds,   64, shuffle=False, num_workers=2)

# ══ CELL 3 — Visualise Class Distribution ════════════════════
from collections import Counter
labels = [full_ds.targets[i] for i in range(len(full_ds))]
counts = Counter(labels)
plt.figure(figsize=(18,4))
plt.bar(range(NUM_CLASSES), [counts[i] for i in range(NUM_CLASSES)], color='steelblue')
plt.xticks(range(NUM_CLASSES), full_ds.classes, rotation=90, fontsize=6)
plt.title("Samples per class"); plt.tight_layout(); plt.show()

# ══ CELL 4 — Fine-tune EfficientNet-B0 ═══════════════════════
backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, NUM_CLASSES)
model = backbone.to(device)

# Unfreeze last 2 blocks + classifier
for name, p in model.named_parameters():
    p.requires_grad = "features.7" in name or "features.8" in name or "classifier" in name

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

best_acc, hist = 0, {"loss":[], "tacc":[], "vacc":[]}
for epoch in range(15):
    model.train(); rl = rc = rt = 0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(); out = model(x)
        loss = criterion(out, y); loss.backward(); optimizer.step()
        rl += loss.item()*len(y); rc += out.argmax(1).eq(y).sum().item(); rt += len(y)
    hist["loss"].append(rl/rt); hist["tacc"].append(100*rc/rt)

    model.eval(); vc = vt = 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            vc += model(x).argmax(1).eq(y).sum().item(); vt += len(y)
    vacc = 100*vc/vt; hist["vacc"].append(vacc)
    if vacc > best_acc:
        best_acc = vacc; torch.save(model.state_dict(), "best_plant.pth")
    scheduler.step()
    print(f"Ep {epoch+1:2d} | loss={hist['loss'][-1]:.3f} | train={hist['tacc'][-1]:.1f}% | val={vacc:.1f}%")

print(f"\\n✅ Best val accuracy: {best_acc:.2f}%")

# ══ CELL 5 — Confusion Matrix ════════════════════════════════
model.load_state_dict(torch.load("best_plant.pth")); model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in val_dl:
        preds = model(x.to(device)).argmax(1).cpu()
        all_preds.extend(preds); all_labels.extend(y)

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(16,14))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
            xticklabels=full_ds.classes, yticklabels=full_ds.classes)
plt.title("Confusion Matrix"); plt.xticks(rotation=90, fontsize=6)
plt.yticks(rotation=0, fontsize=6); plt.tight_layout(); plt.show()

# ══ CELL 6 — Export to ONNX ══════════════════════════════════
model.eval()
dummy = torch.randn(1, 3, 224, 224).to(device)
torch.onnx.export(model, dummy, "plant_classifier.onnx",
    input_names=["image"], output_names=["logits"],
    dynamic_axes={"image":{0:"batch"}}, opset_version=11)
print("✅ Exported to plant_classifier.onnx")`,
  },
  {
    id:"P02", title:"Real-Time Traffic Sign Detection", emoji:"🚦",
    color:P.accent4, difficulty:"Intermediate", time:"3–4 hrs",
    dataset:"GTSDB — German Traffic Sign Detection Benchmark (900 real road images)",
    datasetUrl:"https://benchmark.ini.rub.de/gtsdb_news.html",
    skills:["Object detection","YOLOv8 fine-tuning","Custom datasets","mAP evaluation"],
    outcome:"Detector that finds traffic signs in dashcam images at >30 FPS",
    steps:[
      "Download GTSDB and convert annotations to YOLO format",
      "Configure YOLOv8 dataset YAML",
      "Fine-tune YOLOv8n for 50 epochs",
      "Evaluate with mAP@0.5 on test split",
      "Run inference on a dashcam video clip",
      "Export to TensorRT for edge deployment",
    ],
    code:`# ══ CELL 1 — Setup ═══════════════════════════════════════════
!pip install -q ultralytics roboflow opencv-python-headless

from ultralytics import YOLO
import yaml, cv2, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request

# ══ CELL 2 — Download GTSDB via Roboflow ══════════════════════
# Sign up free at roboflow.com, then:
# from roboflow import Roboflow
# rf = Roboflow(api_key="YOUR_KEY")
# project = rf.workspace("roboflow-100").project("german-traffic-sign")
# dataset = project.version(1).download("yolov8")

# For demo: use a small publicly available sign dataset
# We'll simulate the workflow with YOLO's built-in COCO demo
model = YOLO("yolov8n.pt")

# ══ CELL 3 — Create Dataset YAML ════════════════════════════
dataset_yaml = """
path: /content/gtsdb
train: images/train
val:   images/val
test:  images/test

nc: 43
names:
  0: speed_limit_20    1: speed_limit_30    2: speed_limit_50
  3: speed_limit_60    4: speed_limit_70    5: speed_limit_80
  6: end_speed_limit   7: speed_limit_100   8: speed_limit_120
  9: no_overtaking    10: no_overtaking_3.5t 11: right_of_way
  12: priority_road   13: give_way          14: stop
  15: no_vehicles     16: no_trucks         17: no_entry
  18: general_danger  19: bend_left         20: bend_right
  21: double_bend     22: rough_road        23: slippery_road
  24: road_narrows    25: roadwork          26: traffic_signals
  27: pedestrians     28: children          29: bicycles
  30: icy_road        31: wild_animals      32: end_restrictions
  33: turn_right      34: turn_left         35: ahead_only
  36: straight_or_right 37: straight_or_left 38: keep_right
  39: keep_left       40: roundabout        41: end_no_overtaking
  42: end_no_overtaking_3.5t
"""
with open("/content/gtsdb.yaml", "w") as f:
    f.write(dataset_yaml)

# ══ CELL 4 — Fine-tune YOLOv8 ════════════════════════════════
model = YOLO("yolov8n.pt")

results = model.train(
    data="/content/gtsdb.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=0.01,
    lrf=0.001,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    augment=True,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    project="traffic_sign_det",
    name="gtsdb_v8n",
    save=True,
    plots=True,
)
print("Training complete!")
print(f"Best mAP@50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.3f}")

# ══ CELL 5 — Evaluate on Test Set ════════════════════════════
metrics = model.val(data="/content/gtsdb.yaml", split="test")
print(f"Test mAP@0.5:      {metrics.box.map50:.3f}")
print(f"Test mAP@0.5:0.95: {metrics.box.map:.3f}")
print(f"Precision:         {metrics.box.mp:.3f}")
print(f"Recall:            {metrics.box.mr:.3f}")

# ══ CELL 6 — Inference on Video ══════════════════════════════
# Replace with your dashcam video
VIDEO_PATH = "/content/sample_dashcam.mp4"
model = YOLO("traffic_sign_det/gtsdb_v8n/weights/best.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_writer = cv2.VideoWriter("/content/detected.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    results = model(frame, conf=0.4, verbose=False)
    annotated = results[0].plot()
    out_writer.write(annotated)
    frame_count += 1

cap.release(); out_writer.release()
print(f"✅ Processed {frame_count} frames → /content/detected.mp4")`,
  },
  {
    id:"P03", title:"Chest X-Ray Pathology Screener", emoji:"🩻",
    color:P.accent1, difficulty:"Intermediate", time:"4–5 hrs",
    dataset:"NIH ChestX-ray14 (112,120 real chest X-rays, 14 diseases)",
    datasetUrl:"https://www.kaggle.com/datasets/nih-chest-xrays/data",
    skills:["Multi-label classification","Medical imaging","Class imbalance handling","GradCAM explainability"],
    outcome:"Model that screens for 14 chest conditions with AUC > 0.80 on 5 major pathologies",
    steps:[
      "Download ChestX-ray14 subset from Kaggle",
      "Handle severe class imbalance with weighted sampling",
      "Fine-tune DenseNet-121 (same architecture as CheXNet paper)",
      "Evaluate per-class AUC (ROC curve)",
      "Visualise GradCAM heatmaps for model explainability",
      "Compare to radiologist performance (from original paper)",
    ],
    code:`# ══ CELL 1 — Setup ═══════════════════════════════════════════
!pip install -q torch torchvision grad-cam matplotlib scikit-learn pandas pillow

import torch, torch.nn as nn, pandas as pd, numpy as np
import torchvision.transforms as T
from torchvision import models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from PIL import Image
import warnings; warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"
PATHOLOGIES = ['Atelectasis','Cardiomegaly','Effusion','Infiltration',
               'Mass','Nodule','Pneumonia','Pneumothorax',
               'Consolidation','Edema','Emphysema','Fibrosis',
               'Pleural_Thickening','Hernia']

# ══ CELL 2 — Custom Dataset ══════════════════════════════════
class ChestXrayDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(f"{self.img_dir}/{row['Image Index']}").convert("RGB")
        if self.transform: img = self.transform(img)
        # Multi-hot label vector
        label = torch.zeros(14)
        for i, p in enumerate(PATHOLOGIES):
            if p in str(row['Finding Labels']): label[i] = 1.0
        return img, label

# Load metadata CSV
df = pd.read_csv("/content/Data_Entry_2017.csv")
print(f"Total images: {len(df)}")
print("\\nClass distribution (% positive):")
for p in PATHOLOGIES:
    pct = df['Finding Labels'].str.contains(p).mean() * 100
    print(f"  {p:25s} {pct:.1f}%")

# ══ CELL 3 — Transforms & Loaders ════════════════════════════
tf_train = T.Compose([
    T.Resize(256), T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
tf_val = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# Use official train/val split files
train_df = pd.read_csv("/content/train_val_list.txt", header=None, names=["Image Index"])
val_df   = pd.read_csv("/content/test_list.txt", header=None, names=["Image Index"])
train_df = df[df["Image Index"].isin(train_df["Image Index"])]
val_df   = df[df["Image Index"].isin(val_df["Image Index"])]

IMG_DIR = "/content/images"
train_ds = ChestXrayDataset(train_df, IMG_DIR, tf_train)
val_ds   = ChestXrayDataset(val_df,   IMG_DIR, tf_val)

# Weighted sampler for imbalance
normal_idx  = train_df['Finding Labels'].eq('No Finding').values
weights = np.where(normal_idx, 0.1, 0.9)
sampler = WeightedRandomSampler(weights, len(weights))

train_dl = DataLoader(train_ds, 32, sampler=sampler, num_workers=2)
val_dl   = DataLoader(val_ds,   64, shuffle=False,   num_workers=2)
print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

# ══ CELL 4 — CheXNet: DenseNet-121 ════════════════════════════
backbone = models.densenet121(weights='IMAGENET1K_V1')
backbone.classifier = nn.Sequential(
    nn.Linear(backbone.classifier.in_features, 14),
    nn.Sigmoid()       # multi-label → sigmoid, not softmax
)
model = backbone.to(device)
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Focal loss for class imbalance
class FocalBCE(nn.Module):
    def __init__(self, gamma=2, pos_weight=None):
        super().__init__(); self.gamma=gamma; self.pw=pos_weight
    def forward(self, pred, target):
        bce = nn.functional.binary_cross_entropy(pred, target, reduction='none')
        pt  = torch.where(target==1, pred, 1-pred)
        return ((1-pt)**self.gamma * bce).mean()

criterion = FocalBCE(gamma=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# ══ CELL 5 — Train & Evaluate per-class AUC ═══════════════════
for epoch in range(5):
    model.train(); rl = rt = 0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(); pred = model(x)
        loss = criterion(pred, y); loss.backward(); optimizer.step()
        rl += loss.item(); rt += 1
    
    model.eval(); all_p, all_y = [], []
    with torch.no_grad():
        for x, y in val_dl:
            p = model(x.to(device)).cpu(); all_p.append(p); all_y.append(y)
    all_p = torch.cat(all_p).numpy(); all_y = torch.cat(all_y).numpy()
    
    aucs = [roc_auc_score(all_y[:,i], all_p[:,i])
            for i in range(14) if all_y[:,i].sum() > 0]
    print(f"Epoch {epoch+1} | Loss={rl/rt:.4f} | Mean AUC={np.mean(aucs):.3f}")

# Per-class AUC
print("\\nPer-class AUC:")
for i, p in enumerate(PATHOLOGIES):
    if all_y[:,i].sum() > 0:
        auc = roc_auc_score(all_y[:,i], all_p[:,i])
        bar = "█"*int(auc*20)
        print(f"  {p:25s} {bar} {auc:.3f}")

# ══ CELL 6 — GradCAM Explainability ══════════════════════════
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

target_layer = [model.features.denseblock4.denselayer16.conv2]
cam = GradCAM(model=model, target_layers=target_layer)

# Show heatmap for first validation image
x_sample, y_sample = val_ds[0]
grayscale_cam = cam(input_tensor=x_sample.unsqueeze(0).to(device))
img_np = x_sample.permute(1,2,0).numpy() * [0.229,0.224,0.225] + [0.485,0.456,0.406]
img_np = np.clip(img_np, 0, 1)
vis = show_cam_on_image(img_np, grayscale_cam[0], use_rgb=True)

detected = [PATHOLOGIES[i] for i in range(14) if y_sample[i].item() > 0.5]
fig, ax = plt.subplots(1,2,figsize=(10,4))
ax[0].imshow(img_np, cmap='gray'); ax[0].set_title("X-Ray")
ax[1].imshow(vis); ax[1].set_title(f"GradCAM: {detected or ['Normal']}")
for a in ax: a.axis('off')
plt.tight_layout(); plt.show()`,
  },
  {
    id:"P04", title:"Street Scene Segmentation (Cityscapes)", emoji:"🌆",
    color:P.accent2, difficulty:"Intermediate", time:"3–4 hrs",
    dataset:"Cityscapes (2,975 train / 500 val finely annotated urban images, 19 classes)",
    datasetUrl:"https://www.cityscapes-dataset.com/",
    skills:["Semantic segmentation","SegFormer","mIoU metric","Cityscapes API","Deployment prep"],
    outcome:"Segment every pixel in city driving scenes: road, car, person, building, sky…",
    steps:[
      "Register and download Cityscapes dataset",
      "Write custom Dataset class for the 19-class label encoding",
      "Fine-tune SegFormer-B2 from HuggingFace",
      "Track mIoU with the official cityscapes evaluation script",
      "Visualise colour-coded predictions overlaid on input",
    ],
    code:`# ══ CELL 1 — Setup ═══════════════════════════════════════════
!pip install -q transformers torch torchvision matplotlib pillow numpy

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import torch, torch.nn as nn, numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request

device = "cuda" if torch.cuda.is_available() else "cpu"

# Cityscapes 19-class colour palette
CS_PALETTE = np.array([
    [128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],
    [153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],
    [70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],
    [0,60,100],[0,80,100],[0,0,230],[119,11,32]
], dtype=np.uint8)

CS_CLASSES = ["road","sidewalk","building","wall","fence","pole",
              "traffic light","traffic sign","vegetation","terrain",
              "sky","person","rider","car","truck","bus",
              "train","motorcycle","bicycle"]

# ══ CELL 2 — Load Pretrained SegFormer ═══════════════════════
# Using cityscapes-pretrained model directly
model_name = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
processor  = SegformerImageProcessor.from_pretrained(model_name)
model      = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
print("SegFormer-B2 loaded ✅")
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# ══ CELL 3 — Run Inference on Street Image ════════════════════
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Above_Gotham.jpg/800px-Above_Gotham.jpg"
urllib.request.urlretrieve(url, "street.jpg")

img = Image.open("street.jpg").convert("RGB")
inputs = processor(images=img, return_tensors="pt").to(device)

with torch.no_grad():
    logits = model(**inputs).logits   # (1, 19, H/4, W/4)

# Upsample to original size
upsampled = nn.functional.interpolate(
    logits, size=img.size[::-1], mode="bilinear", align_corners=False)
pred_seg = upsampled.argmax(1)[0].cpu().numpy()   # (H, W)

# ══ CELL 4 — Visualise Segmentation ══════════════════════════
colour_seg = CS_PALETTE[pred_seg]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(img); axes[0].set_title("Input Image")
axes[1].imshow(colour_seg); axes[1].set_title("Segmentation Map")
overlay = (np.array(img) * 0.5 + colour_seg * 0.5).astype(np.uint8)
axes[2].imshow(overlay); axes[2].set_title("Overlay (50/50)")
for ax in axes: ax.axis('off')
plt.tight_layout(); plt.show()

# Print class coverage
unique, counts = np.unique(pred_seg, return_counts=True)
total = pred_seg.size
print("\\nClass coverage:")
for u, cnt in sorted(zip(unique, counts), key=lambda x:-x[1])[:8]:
    pct = cnt/total*100
    print(f"  {CS_CLASSES[u]:20s} {pct:.1f}%  {'█'*int(pct/2)}")

# ══ CELL 5 — mIoU Calculation ════════════════════════════════
def compute_iou(pred, target, num_classes=19):
    ious = []
    for cls in range(num_classes):
        tp = ((pred == cls) & (target == cls)).sum()
        fp = ((pred == cls) & (target != cls)).sum()
        fn = ((pred != cls) & (target == cls)).sum()
        if tp + fp + fn == 0: continue
        ious.append(tp / (tp + fp + fn))
    return np.mean(ious)

# Simulate ground truth for demo
gt_fake = pred_seg.copy()   # replace with real labels
print(f"\\nmIoU (demo self-evaluation): {compute_iou(pred_seg, gt_fake):.3f}")
print("With real Cityscapes GT, SegFormer-B2 achieves ~47.6 mIoU")

# ══ CELL 6 — Batch Inference Speed Test ══════════════════════
import time
imgs = [img] * 4
t0 = time.time()
for im in imgs:
    inp = processor(images=im, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**inp).logits
elapsed = time.time() - t0
print(f"\\n4 images in {elapsed:.2f}s → {4/elapsed:.1f} FPS on {device}")`,
  },
  {
    id:"P05", title:"Real-Time People Counter with Tracking", emoji:"👥",
    color:P.accent5, difficulty:"Intermediate", time:"3 hrs",
    dataset:"MOT17 benchmark + any CCTV/crowd footage",
    datasetUrl:"https://motchallenge.net/data/MOT17/",
    skills:["Object detection","ByteTrack multi-object tracking","Line counting","OpenCV drawing"],
    outcome:"System that counts people entering/leaving a zone in a video — ready for retail/event analytics",
    steps:[
      "Set up YOLOv8 + BoxMOT (ByteTrack)",
      "Define a counting line in the frame",
      "Detect people per frame, assign track IDs",
      "Detect when track crosses the line (up/down direction)",
      "Overlay count statistics on output video",
    ],
    code:`# ══ CELL 1 — Setup ═══════════════════════════════════════════
!pip install -q ultralytics boxmot opencv-python-headless

from ultralytics import YOLO
import cv2, numpy as np, collections
import matplotlib.pyplot as plt

# ══ CELL 2 — People Counter Class ════════════════════════════
class PeopleCounter:
    def __init__(self, line_y_ratio=0.5):
        self.line_y_ratio = line_y_ratio
        self.track_history = collections.defaultdict(list)
        self.count_in  = 0
        self.count_out = 0
        self.counted_ids = set()

    def update(self, frame, detections):
        h, w = frame.shape[:2]
        line_y = int(h * self.line_y_ratio)

        # Draw counting line
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
        cv2.putText(frame, "COUNTING LINE", (10, line_y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        for det in detections:
            if len(det) < 6: continue
            x1,y1,x2,y2,track_id,conf = det[:6]
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            tid = int(track_id)

            self.track_history[tid].append(cy)
            if len(self.track_history[tid]) > 30:
                self.track_history[tid].pop(0)

            # Crossing detection
            if len(self.track_history[tid]) >= 2 and tid not in self.counted_ids:
                prev_y = self.track_history[tid][-2]
                if prev_y < line_y <= cy:        # moving down = IN
                    self.count_in += 1; self.counted_ids.add(tid)
                elif prev_y > line_y >= cy:       # moving up = OUT
                    self.count_out += 1; self.counted_ids.add(tid)

            # Draw bounding box and trail
            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,200,0), 2)
            cv2.putText(frame, f"ID:{tid}", (int(x1),int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,0), 1)
            points = self.track_history[tid]
            for i in range(1, min(len(points),10)):
                alpha = i / 10
                cv2.circle(frame, (cx, points[-i]), 2, (0,200,200), -1)

        # Stats overlay
        cv2.rectangle(frame, (5,5), (220,80), (0,0,0), -1)
        cv2.putText(frame, f"IN:  {self.count_in}",  (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,100), 2)
        cv2.putText(frame, f"OUT: {self.count_out}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,100,255), 2)
        return frame

# ══ CELL 3 — Run on Video ════════════════════════════════════
VIDEO_PATH = "/content/crowd_video.mp4"  # replace with your video
model   = YOLO("yolov8n.pt")
counter = PeopleCounter(line_y_ratio=0.55)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("/content/counted.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
frame_n = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_n > 1000: break
    # YOLO detect persons only (class 0), with ByteTrack
    results = model.track(frame, classes=[0], persist=True,
                          tracker="bytetrack.yaml", conf=0.4, verbose=False)
    dets = []
    if results[0].boxes.id is not None:
        boxes  = results[0].boxes.xyxy.cpu().numpy()
        ids    = results[0].boxes.id.cpu().numpy()
        confs  = results[0].boxes.conf.cpu().numpy()
        dets   = [[*b, i, c] for b,i,c in zip(boxes,ids,confs)]
    frame = counter.update(frame, dets)
    out.write(frame); frame_n += 1

cap.release(); out.release()
print(f"✅ Done — {frame_n} frames processed")
print(f"Final count → IN: {counter.count_in}  OUT: {counter.count_out}")
print(f"Net change: {counter.count_in - counter.count_out} people")`,
  },
  {
    id:"P06", title:"Satellite Deforestation Monitor", emoji:"🛰️",
    color:P.accent6, difficulty:"Advanced", time:"4–5 hrs",
    dataset:"LEVIR-CD change detection dataset / Hansen Global Forest Cover",
    datasetUrl:"https://justchenhao.github.io/LEVIR/",
    skills:["Change detection","Satellite image processing","ChangeFormer / BIT","GeoTIFF with rasterio"],
    outcome:"Pipeline that compares two satellite images of the same location and highlights deforestation/construction changes",
    steps:[
      "Load paired bi-temporal satellite images (rasterio for GeoTIFF)",
      "Preprocess: normalise, handle multi-spectral bands",
      "Run ChangeFormer model for change detection",
      "Post-process: morphological ops to clean predictions",
      "Compute change area in hectares using GSD (ground sampling distance)",
      "Export change map as GeoTIFF with geographic coordinates",
    ],
    code:`# ══ CELL 1 — Setup ═══════════════════════════════════════════
!pip install -q rasterio torch torchvision matplotlib numpy pillow

import numpy as np, matplotlib.pyplot as plt
from PIL import Image
import torch, torch.nn as nn
import urllib.request, cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

# ══ CELL 2 — Load Bi-Temporal Images ════════════════════════
# Simulate with two images of same scene (before / after)
url_before = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Amazon_forest.jpg/640px-Amazon_forest.jpg"
url_after  = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Above_Gotham.jpg/640px-Above_Gotham.jpg"
urllib.request.urlretrieve(url_before, "before.jpg")
urllib.request.urlretrieve(url_after,  "after.jpg")

SIZE = (256, 256)
before = np.array(Image.open("before.jpg").convert("RGB").resize(SIZE))
after  = np.array(Image.open("after.jpg").convert("RGB").resize(SIZE))

# ══ CELL 3 — Classical Change Detection (Baseline) ══════════
# Difference image approach
diff = cv2.absdiff(before, after)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
_, change_mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

# Morphological cleanup
kernel = np.ones((5,5), np.uint8)
change_clean = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)
change_clean = cv2.morphologyEx(change_clean, cv2.MORPH_CLOSE, kernel)

# ══ CELL 4 — Deep Change Detection (Siamese Network) ════════
class SiameseCD(nn.Module):
    """Minimal Siamese change detector — twin encoder + diff decoder"""
    def __init__(self):
        super().__init__()
        import torchvision.models as M
        enc = M.resnet18(weights='IMAGENET1K_V1')
        self.encoder = nn.Sequential(*list(enc.children())[:-3])  # up to layer3
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 128, 1), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 32, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x1, x2):
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        diff = torch.abs(f1 - f2)   # feature difference
        return self.decoder(diff)   # change map

model_cd = SiameseCD().to(device)
print(f"Siamese CD params: {sum(p.numel() for p in model_cd.parameters())/1e6:.1f}M")

# Normalise and forward pass
from torchvision import transforms as T
tf = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
t1 = tf(Image.fromarray(before)).unsqueeze(0).to(device)
t2 = tf(Image.fromarray(after)).unsqueeze(0).to(device)
with torch.no_grad():
    logits = model_cd(t1, t2)
change_deep = torch.sigmoid(logits)[0,0].cpu().numpy()
change_deep_vis = (change_deep * 255).astype(np.uint8)
change_deep_vis = cv2.resize(change_deep_vis, SIZE)

# ══ CELL 5 — Visualise Results ════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for ax, im, t in zip(axes[0],
    [before, after, diff],
    ["Before", "After", "Pixel Difference"]):
    ax.imshow(im); ax.set_title(t); ax.axis('off')

# Overlay change on before image
overlay_classical = before.copy()
overlay_classical[change_clean > 0] = [255, 50, 50]
overlay_deep = before.copy()
heat = change_deep_vis[:,:,None] * np.array([1,0,0])
overlay_deep = np.clip(overlay_deep * 0.5 + heat, 0, 255).astype(np.uint8)

for ax, im, t in zip(axes[1],
    [change_clean, change_deep_vis, overlay_deep],
    ["Classical Change Mask", "Deep Change Map", "Deep Change Overlay"]):
    ax.imshow(im, cmap='hot' if im.ndim==2 else None); ax.set_title(t); ax.axis('off')

plt.suptitle("Bi-Temporal Change Detection", fontsize=14)
plt.tight_layout(); plt.show()

# ══ CELL 6 — Change Area Estimation ══════════════════════════
GSD_METERS = 10   # e.g. Sentinel-2 = 10m/pixel
changed_pixels = (change_clean > 0).sum()
area_m2 = changed_pixels * (GSD_METERS ** 2)
area_ha = area_m2 / 10000
print(f"Changed pixels: {changed_pixels:,}")
print(f"Changed area:   {area_m2:,.0f} m²  ({area_ha:.1f} hectares)")
print(f"(At GSD = {GSD_METERS}m/pixel, as used by Sentinel-2)")`,
  },
];

/* ─────────────────────────────────────────────────────────────
   HOSTING GUIDE
───────────────────────────────────────────────────────────── */
const HOSTING = [
  {
    platform:"Google Colab", icon:"🟡", cost:"Free / $10/mo Pro",
    bestFor:"Learning, experimentation, sharing notebooks",
    pros:["Zero setup, runs in browser","Free GPU (T4) for ~12 hrs/day","Share notebook link instantly","nbviewer-style rendered output"],
    cons:["Session resets after inactivity","Limited RAM/disk on free tier","No persistent endpoints"],
    steps:[
      'Go to colab.research.google.com',
      'File → Upload notebook OR paste cells',
      'Runtime → Change runtime type → T4 GPU',
      'Share: File → Share → "Anyone with link"',
      'For persistent files: mount Google Drive',
    ],
    code:`# Mount Google Drive to persist data across sessions
from google.colab import drive
drive.mount('/content/drive')
# Now save models to /content/drive/MyDrive/cv_models/
torch.save(model.state_dict(), '/content/drive/MyDrive/cv_models/best.pth')`,
  },
  {
    platform:"Hugging Face Spaces", icon:"🤗", cost:"Free (CPU) / $0.60/hr GPU",
    bestFor:"Deploying interactive CV demos with Gradio or Streamlit",
    pros:["Free CPU hosting, instant GPU on demand","Built-in Gradio/Streamlit support","Public URL shared instantly","Git-based deployment"],
    cons:["Cold start delay on free tier","Compute limits on free plan"],
    steps:[
      'Create account at huggingface.co',
      'New Space → choose Gradio template',
      'Upload app.py + requirements.txt',
      'Push to Space repo → auto-deploys in ~2 min',
      'Share URL: username.hf.space/space-name',
    ],
    code:`# app.py — Gradio CV demo (upload to HF Spaces)
import gradio as gr
from PIL import Image
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect(image):
    results = model(np.array(image), conf=0.4, verbose=False)
    annotated = results[0].plot()
    return Image.fromarray(annotated[..., ::-1])

demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Image(label="Detections"),
    title="YOLOv8 Object Detector",
    examples=["dog.jpg", "street.jpg"],
)
demo.launch()
# requirements.txt: ultralytics gradio`,
  },
  {
    platform:"Streamlit Cloud", icon:"🔴", cost:"Free",
    bestFor:"CV apps with rich UI — file upload, sliders, charts",
    pros:["Free hosting for public repos","Pure Python, no web dev needed","Fast iteration","GitHub integration"],
    cons:["App sleeps after 7 days inactive","Limited memory on free tier","No GPU"],
    steps:[
      'pip install streamlit',
      'Write app.py with st.* widgets',
      'Push to GitHub',
      'Go to share.streamlit.io → Deploy from GitHub',
      'Get public URL immediately',
    ],
    code:`# app.py — Streamlit CV app
import streamlit as st
import cv2, numpy as np
from PIL import Image
from ultralytics import YOLO

st.title("🎯 Real-Time Object Detector")
st.sidebar.header("Settings")
conf = st.sidebar.slider("Confidence", 0.1, 0.9, 0.4)

@st.cache_resource
def load_model(): return YOLO("yolov8n.pt")
model = load_model()

uploaded = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input", use_column_width=True)
    with st.spinner("Detecting..."):
        results = model(np.array(img), conf=conf, verbose=False)
        ann = results[0].plot()
        st.image(ann[..., ::-1], caption=f"Detected {len(results[0].boxes)} objects")
        st.json({model.names[int(b.cls)]: float(b.conf)
                 for b in results[0].boxes})`,
  },
  {
    platform:"FastAPI + Docker + Cloud Run", icon:"🐳", cost:"Free tier / Pay-per-request",
    bestFor:"Production CV APIs — scalable, portable, professional",
    pros:["Full control over infra","Auto-scaling with Cloud Run / Railway","GPU support on cloud VMs","REST API for integration"],
    cons:["More setup required","Cost scales with traffic"],
    steps:[
      'Write main.py (FastAPI app)',
      'Write Dockerfile',
      'docker build -t cv-api . && docker run -p 8000:8000 cv-api',
      'Test: POST /predict with image file',
      'Deploy: gcloud run deploy OR railway up',
    ],
    code:`# main.py — FastAPI CV inference server
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io, numpy as np
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="CV Inference API")
model = YOLO("yolov8n.pt")

@app.get("/health")
async def health(): return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    results = model(img, conf=0.4, verbose=False)[0]
    detections = [
        {"class": model.names[int(b.cls)], "confidence": round(float(b.conf), 3),
         "bbox": [round(v) for v in b.xyxy[0].tolist()]}
        for b in results.boxes
    ]
    return JSONResponse({"count": len(detections), "detections": detections})

# Dockerfile:
# FROM python:3.11-slim
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install fastapi uvicorn ultralytics python-multipart
# COPY . .
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Test with curl:
# curl -X POST localhost:8000/predict -F "file=@dog.jpg"`,
  },
  {
    platform:"ONNX Runtime / TensorRT (Edge)", icon:"⚡", cost:"Free",
    bestFor:"Deploying CV models on devices — Raspberry Pi, Jetson Nano, mobile",
    pros:["10–100× faster than PyTorch CPU","Runs on CPU/GPU/NPU","ONNX is framework-agnostic","TensorRT for NVIDIA edge devices"],
    cons:["Export pipeline needed","Limited dynamic shapes","Quantization needs careful testing"],
    steps:[
      'Export model to ONNX',
      'Optimise: onnxsim (simplifier)',
      'Optional: quantise to INT8 for faster CPU',
      'Run with onnxruntime: ort.InferenceSession()',
      'For Jetson: build TensorRT engine from ONNX',
    ],
    code:`# Export PyTorch model to ONNX
import torch
model.eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, "model.onnx",
    opset_version=17,
    input_names=["image"], output_names=["logits"],
    dynamic_axes={"image": {0: "batch_size"}})

# Simplify
!pip install onnx onnxsim
!python -m onnxsim model.onnx model_simplified.onnx

# Run with ONNX Runtime
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("model_simplified.onnx",
       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

img = np.random.randn(1, 3, 224, 224).astype(np.float32)
output = sess.run(None, {"image": img})
print("ONNX output shape:", output[0].shape)

# INT8 Quantization (CPU speedup ~2-4×)
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic("model_simplified.onnx", "model_int8.onnx",
                 weight_type=QuantType.QInt8)
print("INT8 model saved!")

# YOLOv8 has built-in ONNX/TensorRT export:
# from ultralytics import YOLO
# model = YOLO("best.pt")
# model.export(format="onnx")    # → best.onnx
# model.export(format="engine")  # → best.engine (TensorRT)`,
  },
];

/* ─────────────────────────────────────────────────────────────
   SMALL QUIZ BANK
───────────────────────────────────────────────────────────── */
const QUIZ_BANK = [
  {q:"A 256×256 RGB image has how many total numbers?",          a:"196608",                             hint:"256×256×3"},
  {q:"What operation does MaxPooling perform?",                   a:"downsamples / takes maximum",        hint:"Reduces spatial size while keeping strongest activations"},
  {q:"What does IoU stand for?",                                  a:"intersection over union",            hint:"Measures bounding box overlap"},
  {q:"Which segmentation assigns one class per pixel not per instance?", a:"semantic segmentation",     hint:"As opposed to instance segmentation"},
  {q:"What metric measures image restoration quality in dB?",    a:"psnr",                               hint:"Peak Signal to Noise Ratio"},
  {q:"What does NMS stand for in detection?",                    a:"non maximum suppression",            hint:"Removes duplicate bounding boxes"},
  {q:"What is transfer learning?",                               a:"reusing pretrained weights",         hint:"Start from ImageNet weights, not random"},
  {q:"Name the U-Net skip connection purpose.",                  a:"preserve spatial detail",            hint:"Connect encoder to decoder at matching resolution"},
  {q:"What does CLIP stand for?",                                a:"contrastive language image pretraining", hint:"OpenAI model for zero-shot vision"},
  {q:"In NMS, what triggers a box to be suppressed?",           a:"high iou with a higher-confidence box", hint:"IoU > threshold → remove the lower-score box"},
  {q:"What is the output shape of Conv2d(3,64,3) on a 32×32 input?", a:"64×30×30",                    hint:"(H - kernel + 1) with no padding, 64 filters"},
  {q:"What does Otsu's method automatically determine?",         a:"optimal threshold",                  hint:"Maximises inter-class variance"},
  {q:"Name one advantage of ONNX export.",                      a:"faster inference / framework agnostic / edge deployment", hint:"Used on CPU, GPU, and edge devices"},
  {q:"What is mIoU?",                                           a:"mean intersection over union",        hint:"Standard segmentation metric averaged over classes"},
  {q:"What kind of noise does a Median filter remove best?",    a:"salt and pepper",                    hint:"Impulsive, random black/white pixels"},
  {q:"What Python library provides YOLOv8?",                    a:"ultralytics",                        hint:"pip install ultralytics"},
  {q:"What is seam carving used for?",                          a:"content-aware image resizing",        hint:"Remove seams of low energy to shrink width/height"},
  {q:"Name the loss function used for multi-label classification.", a:"binary cross entropy",            hint:"BCEWithLogitsLoss in PyTorch"},
  {q:"What does GradCAM visualise?",                            a:"which pixels influence a prediction", hint:"Gradient-weighted Class Activation Mapping"},
  {q:"What is Hamming distance used for in image hashing?",     a:"measure similarity between two hashes", hint:"Count of differing bits"},
];

/* ─────────────────────────────────────────────────────────────
   MAIN COMPONENT
───────────────────────────────────────────────────────────── */
export default function CVTutorial() {
  const [tab, setTab] = useState("overview");
  const [selChallenge, setSelChallenge] = useState(null);
  const [selProject, setSelProject]     = useState(null);
  const [selHosting, setSelHosting]     = useState(null);
  const [showSolution, setShowSolution] = useState(false);
  const [copiedId, setCopiedId]         = useState(null);
  const [searchCh, setSearchCh]         = useState("");
  const [filterDiff, setFilterDiff]     = useState("All");
  const [quiz, setQuiz] = useState({active:false,q:0,score:0,done:false,answered:false,correct:null});
  const [qAns, setQAns] = useState("");

  const copy = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id); setTimeout(()=>setCopiedId(null),2000);
  };

  const submitQ = () => {
    const cur = QUIZ_BANK[quiz.q];
    const ok = cur.a.toLowerCase().split("/").some(ans =>
      qAns.toLowerCase().trim().replace(/[^a-z0-9 ]/g,"").includes(ans.trim().split(" ")[0]) ||
      ans.toLowerCase().includes(qAns.toLowerCase().trim())
    );
    setQuiz(s=>({...s, answered:true, correct:ok, score:s.score+(ok?1:0)}));
  };

  const nextQ = () => {
    setQAns(""); const n=quiz.q+1;
    if(n>=QUIZ_BANK.length) setQuiz(s=>({...s,done:true,answered:false}));
    else setQuiz(s=>({...s,q:n,answered:false,correct:null}));
  };

  // ── helpers ──
  const Tag = ({label,color,bg}) => (
    <span style={{background:bg||`${color}18`,border:`1px solid ${color}40`,
      color,padding:"2px 9px",borderRadius:"100px",fontSize:"11px",fontWeight:600,whiteSpace:"nowrap"}}>
      {label}
    </span>
  );

  const Card = ({style,children}) => (
    <div style={{background:P.card,border:`1px solid ${P.border}`,borderRadius:"12px",padding:"20px",...style}}>
      {children}
    </div>
  );

  const CodeBlock = ({code,id}) => (
    <div style={{position:"relative",marginBottom:"12px"}}>
      <pre style={{background:"#0b0f1a",border:`1px solid ${P.border}`,borderRadius:"8px",
        padding:"14px",color:"#e2eaf7",fontSize:"11.5px",overflowX:"auto",
        whiteSpace:"pre",margin:0,lineHeight:"1.65",
        fontFamily:"'JetBrains Mono','Fira Code',monospace"}}>
        {code}
      </pre>
      <button onClick={()=>copy(code,id)} style={{
        position:"absolute",top:"8px",right:"8px",
        background:copiedId===id?P.accent3:"#1e2a3a",
        border:"none",borderRadius:"4px",padding:"3px 10px",
        color:copiedId===id?"#000":P.text,cursor:"pointer",fontSize:"11px",transition:"all .2s"}}>
        {copiedId===id?"✓ Copied":"Copy"}
      </button>
    </div>
  );

  const diffColor = d => d==="Easy"?P.accent3:d==="Medium"?P.accent6:P.accent4;

  const TABS = [
    {id:"overview",    label:"📋 Overview"},
    {id:"challenges",  label:`💻 Challenges (${CHALLENGES.length})`},
    {id:"projects",    label:`🏗️ Projects (${PROJECTS.length})`},
    {id:"hosting",     label:"🚀 Deploy & Host"},
    {id:"quiz",        label:"🧪 Quiz"},
  ];

  const filteredCh = CHALLENGES.filter(c => {
    const matchSearch = !searchCh || c.title.toLowerCase().includes(searchCh.toLowerCase()) ||
                        c.tag.toLowerCase().includes(searchCh.toLowerCase());
    const matchDiff = filterDiff==="All" || c.difficulty===filterDiff;
    return matchSearch && matchDiff;
  });

  return (
    <div style={{background:P.bg,minHeight:"100vh",fontFamily:"'Inter','Segoe UI',sans-serif",color:P.text}}>

      {/* HEADER */}
      <div style={{background:"linear-gradient(135deg,#050811 0%,#0d1940 50%,#050811 100%)",
        borderBottom:`1px solid ${P.border}`,padding:"24px 20px 20px",position:"relative",overflow:"hidden"}}>
        <div style={{position:"absolute",inset:0,
          backgroundImage:"radial-gradient(circle at 10% 50%,rgba(56,189,248,.12) 0%,transparent 45%),radial-gradient(circle at 90% 20%,rgba(129,140,248,.1) 0%,transparent 40%)",
          pointerEvents:"none"}}/>
        <div style={{maxWidth:"980px",margin:"0 auto",position:"relative"}}>
          <div style={{display:"flex",alignItems:"center",gap:"14px",marginBottom:"8px"}}>
            <span style={{fontSize:"36px"}}>👁️</span>
            <div>
              <div style={{fontSize:"10px",letterSpacing:"3px",color:P.accent2,textTransform:"uppercase",marginBottom:"3px"}}>
                Complete Beginner → Advanced Course · v3
              </div>
              <h1 style={{margin:0,fontSize:"24px",fontWeight:800,
                background:`linear-gradient(90deg,${P.accent1},${P.accent2},${P.accent3})`,
                WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"}}>
                Computer Vision: Zero → Practitioner
              </h1>
            </div>
          </div>
          <p style={{color:P.muted,margin:"6px 0 14px",fontSize:"13px"}}>
            Coding challenges · Real-world projects with real datasets · 88 domains · Deploy anywhere
          </p>
          <div style={{display:"flex",gap:"7px",flexWrap:"wrap"}}>
            {[["12 Challenges",P.accent1],["6 Real Projects",P.accent2],["Real Datasets",P.accent3],
              ["5 Hosting Options",P.accent4],["20-Q Quiz",P.accent5],["Google Colab",P.accent6]
            ].map(([l,c])=><Tag key={l} label={l} color={c}/>)}
          </div>
        </div>
      </div>

      {/* TABS */}
      <div style={{borderBottom:`1px solid ${P.border}`,background:P.surface,position:"sticky",top:0,zIndex:10}}>
        <div style={{maxWidth:"980px",margin:"0 auto",display:"flex",overflowX:"auto"}}>
          {TABS.map(t=>(
            <button key={t.id} onClick={()=>{setTab(t.id);setSelChallenge(null);setSelProject(null);setSelHosting(null);setShowSolution(false);}}
              style={{background:"none",border:"none",cursor:"pointer",padding:"13px 18px",
                fontSize:"13px",fontWeight:600,whiteSpace:"nowrap",
                color:tab===t.id?P.accent1:P.muted,
                borderBottom:tab===t.id?`2px solid ${P.accent1}`:"2px solid transparent",
                transition:"all .2s"}}>
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* BODY */}
      <div style={{maxWidth:"980px",margin:"0 auto",padding:"24px 16px"}}>

        {/* ══════════════ OVERVIEW ══════════════ */}
        {tab==="overview" && (
          <div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(240px,1fr))",gap:"14px",marginBottom:"20px"}}>
              {[
                {icon:"💻",color:P.accent1,title:"Coding Challenges",body:`${CHALLENGES.length} LeetCode-style CV problems — from flood fill to seam carving to implementing Conv2D from scratch. Each has full solution, complexity analysis, and a follow-up question for interview prep.`},
                {icon:"🏗️",color:P.accent2,title:"Real-World Projects",body:`${PROJECTS.length} hands-on projects on real datasets: plant disease (54K images), chest X-rays (112K images), Cityscapes street scenes, traffic signs, people counting, and satellite deforestation monitoring.`},
                {icon:"🚀",color:P.accent3,title:"Deploy Anywhere",body:"Complete guide to hosting CV models: Google Colab sharing, Hugging Face Spaces (Gradio), Streamlit Cloud, FastAPI + Docker + Cloud Run, and ONNX/TensorRT for edge devices."},
              ].map(c=>(
                <div key={c.title} style={{background:P.card,border:`1px solid ${P.border}`,
                  borderRadius:"12px",padding:"18px",borderTop:`3px solid ${c.color}`}}>
                  <div style={{fontSize:"24px",marginBottom:"8px"}}>{c.icon}</div>
                  <h3 style={{margin:"0 0 8px",fontSize:"14px",color:c.color}}>{c.title}</h3>
                  <p style={{margin:0,color:P.muted,fontSize:"12px",lineHeight:"1.7"}}>{c.body}</p>
                </div>
              ))}
            </div>

            <Card style={{marginBottom:"16px"}}>
              <h3 style={{margin:"0 0 14px",color:P.accent2}}>💻 Challenges at a Glance</h3>
              <div style={{display:"grid",gap:"8px"}}>
                {CHALLENGES.map(ch=>(
                  <div key={ch.id} onClick={()=>{setTab("challenges");setSelChallenge(ch);setShowSolution(false);}}
                    style={{display:"flex",alignItems:"center",gap:"10px",padding:"10px 14px",
                      background:P.card2,borderRadius:"8px",cursor:"pointer",
                      borderLeft:`3px solid ${diffColor(ch.difficulty)}`}}>
                    <Tag label={ch.difficulty} color={diffColor(ch.difficulty)}/>
                    <Tag label={ch.tag} color={P.muted}/>
                    <span style={{fontWeight:600,fontSize:"13px",flex:1}}>{ch.title}</span>
                    <span style={{color:P.muted,fontSize:"11px"}}>{ch.company}</span>
                    <span style={{color:P.accent1}}>→</span>
                  </div>
                ))}
              </div>
            </Card>

            <Card style={{marginBottom:"16px"}}>
              <h3 style={{margin:"0 0 14px",color:P.accent3}}>🏗️ Projects at a Glance</h3>
              <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(260px,1fr))",gap:"10px"}}>
                {PROJECTS.map(p=>(
                  <div key={p.id} onClick={()=>{setTab("projects");setSelProject(p);}}
                    style={{background:P.card2,border:`1px solid ${P.border}`,
                      borderRadius:"10px",padding:"14px",cursor:"pointer",
                      borderTop:`3px solid ${p.color}`}}>
                    <div style={{fontSize:"28px",marginBottom:"6px"}}>{p.emoji}</div>
                    <div style={{fontWeight:700,fontSize:"13px",color:P.text,marginBottom:"4px"}}>{p.title}</div>
                    <div style={{fontSize:"11px",color:P.muted,marginBottom:"6px"}}>{p.dataset.split("(")[0]}</div>
                    <div style={{display:"flex",gap:"5px",flexWrap:"wrap"}}>
                      <Tag label={p.difficulty} color={diffColor(p.difficulty)}/>
                      <Tag label={p.time} color={P.muted}/>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          </div>
        )}

        {/* ══════════════ CHALLENGES ══════════════ */}
        {tab==="challenges" && !selChallenge && (
          <div>
            <div style={{display:"flex",gap:"10px",marginBottom:"16px",flexWrap:"wrap"}}>
              <input value={searchCh} onChange={e=>setSearchCh(e.target.value)}
                placeholder="🔍  Search challenges…"
                style={{flex:1,minWidth:"180px",background:P.card,border:`1px solid ${P.border}`,
                  borderRadius:"8px",padding:"8px 14px",color:P.text,fontSize:"13px",outline:"none"}}/>
              {["All","Easy","Medium","Hard"].map(d=>(
                <button key={d} onClick={()=>setFilterDiff(d)}
                  style={{background:filterDiff===d?P.accent1:"none",
                    border:`1px solid ${filterDiff===d?P.accent1:P.border}`,
                    color:filterDiff===d?"#000":P.muted,borderRadius:"8px",
                    padding:"7px 14px",cursor:"pointer",fontSize:"12px",fontWeight:600}}>
                  {d}
                </button>
              ))}
            </div>
            <div style={{display:"grid",gap:"12px"}}>
              {filteredCh.map(ch=>(
                <div key={ch.id} onClick={()=>{setSelChallenge(ch);setShowSolution(false);}}
                  style={{background:P.card,border:`1px solid ${P.border}`,
                    borderRadius:"12px",padding:"18px",cursor:"pointer",
                    borderLeft:`4px solid ${diffColor(ch.difficulty)}`}}>
                  <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",gap:"10px",marginBottom:"8px"}}>
                    <div style={{display:"flex",alignItems:"center",gap:"8px",flexWrap:"wrap"}}>
                      <span style={{fontWeight:700,fontSize:"15px",color:P.text}}>{ch.title}</span>
                      <Tag label={ch.difficulty} color={diffColor(ch.difficulty)}/>
                      <Tag label={ch.tag} color={P.accent2}/>
                    </div>
                    <span style={{color:P.accent1,fontSize:"18px"}}>→</span>
                  </div>
                  <p style={{margin:"0 0 8px",color:P.muted,fontSize:"12px"}}>{ch.desc.slice(0,120)}…</p>
                  <div style={{fontSize:"11px",color:P.muted}}>🏢 {ch.company}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {tab==="challenges" && selChallenge && (
          <div>
            <button onClick={()=>{setSelChallenge(null);setShowSolution(false);}}
              style={{background:"none",border:`1px solid ${P.border}`,color:P.muted,
                padding:"6px 14px",borderRadius:"8px",cursor:"pointer",marginBottom:"16px",fontSize:"12px"}}>
              ← All Challenges
            </button>
            <Card style={{marginBottom:"14px"}}>
              <div style={{display:"flex",alignItems:"flex-start",justifyContent:"space-between",gap:"12px",marginBottom:"14px"}}>
                <div>
                  <h2 style={{margin:"0 0 8px",fontSize:"20px",color:P.text}}>{selChallenge.title}</h2>
                  <div style={{display:"flex",gap:"8px",flexWrap:"wrap"}}>
                    <Tag label={selChallenge.difficulty} color={diffColor(selChallenge.difficulty)}/>
                    <Tag label={selChallenge.tag} color={P.accent2}/>
                    <Tag label={`🏢 ${selChallenge.company}`} color={P.muted}/>
                  </div>
                </div>
              </div>
              <div style={{background:P.card2,borderRadius:"8px",padding:"14px",marginBottom:"12px"}}>
                <div style={{fontSize:"12px",color:P.muted,marginBottom:"6px"}}>📋 Problem Statement</div>
                <p style={{margin:0,color:P.text,fontSize:"13px",lineHeight:"1.7"}}>{selChallenge.desc}</p>
              </div>
              <div style={{background:"#0b0f1a",borderRadius:"8px",padding:"14px",marginBottom:"12px"}}>
                <div style={{fontSize:"11px",color:P.muted,marginBottom:"6px"}}>📌 Example</div>
                <pre style={{margin:0,color:"#a5f3fc",fontSize:"12px",fontFamily:"monospace",whiteSpace:"pre-wrap"}}>{selChallenge.example}</pre>
              </div>
              <div style={{background:`${P.accent6}10`,border:`1px solid ${P.accent6}30`,
                borderRadius:"8px",padding:"12px",marginBottom:"16px"}}>
                <span style={{color:P.accent6,fontSize:"12px"}}>💡 <strong>Hint:</strong> {selChallenge.hint}</span>
              </div>
              <button onClick={()=>setShowSolution(s=>!s)}
                style={{background:showSolution?P.accent4:P.accent1,color:"#000",
                  border:"none",borderRadius:"8px",padding:"9px 22px",
                  fontSize:"13px",fontWeight:700,cursor:"pointer",marginBottom:showSolution?"14px":"0"}}>
                {showSolution?"▲ Hide Solution":"▼ Show Solution"}
              </button>
              {showSolution && (
                <div>
                  <CodeBlock code={selChallenge.solution} id={`sol-${selChallenge.id}`}/>
                  <div style={{background:`${P.accent2}10`,border:`1px solid ${P.accent2}25`,
                    borderRadius:"8px",padding:"12px",marginBottom:"10px"}}>
                    <div style={{fontSize:"11px",color:P.muted,marginBottom:"4px"}}>⏱️ Complexity</div>
                    <code style={{color:P.accent2,fontSize:"12px"}}>{selChallenge.complexity}</code>
                  </div>
                  <div style={{background:`${P.accent3}10`,border:`1px solid ${P.accent3}25`,
                    borderRadius:"8px",padding:"12px"}}>
                    <div style={{fontSize:"11px",color:P.muted,marginBottom:"4px"}}>🚀 Follow-up Question</div>
                    <p style={{margin:0,color:P.text,fontSize:"13px"}}>{selChallenge.followup}</p>
                  </div>
                </div>
              )}
            </Card>
          </div>
        )}

        {/* ══════════════ PROJECTS ══════════════ */}
        {tab==="projects" && !selProject && (
          <div style={{display:"grid",gap:"14px"}}>
            <p style={{color:P.muted,fontSize:"13px",margin:"0 0 4px"}}>
              Hands-on projects using real publicly available datasets. Each project has step-by-step Colab code.
            </p>
            {PROJECTS.map(p=>(
              <div key={p.id} onClick={()=>setSelProject(p)}
                style={{background:P.card,border:`1px solid ${P.border}`,
                  borderRadius:"12px",padding:"20px",cursor:"pointer",
                  borderLeft:`5px solid ${p.color}`}}>
                <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",gap:"12px",marginBottom:"10px"}}>
                  <div style={{display:"flex",alignItems:"center",gap:"10px"}}>
                    <span style={{fontSize:"28px"}}>{p.emoji}</span>
                    <div>
                      <div style={{fontWeight:700,fontSize:"15px",color:P.text}}>{p.title}</div>
                      <div style={{fontSize:"11px",color:P.muted,marginTop:"2px"}}>{p.dataset}</div>
                    </div>
                  </div>
                  <span style={{color:P.accent1,fontSize:"18px"}}>→</span>
                </div>
                <p style={{margin:"0 0 10px",color:P.muted,fontSize:"12px",lineHeight:"1.6"}}>{p.outcome}</p>
                <div style={{display:"flex",gap:"6px",flexWrap:"wrap"}}>
                  <Tag label={p.difficulty} color={diffColor(p.difficulty)}/>
                  <Tag label={`⏱ ${p.time}`} color={P.muted}/>
                  {p.skills.slice(0,3).map(s=><Tag key={s} label={s} color={p.color}/>)}
                </div>
              </div>
            ))}
          </div>
        )}

        {tab==="projects" && selProject && (
          <div>
            <button onClick={()=>setSelProject(null)}
              style={{background:"none",border:`1px solid ${P.border}`,color:P.muted,
                padding:"6px 14px",borderRadius:"8px",cursor:"pointer",marginBottom:"16px",fontSize:"12px"}}>
              ← All Projects
            </button>
            <Card style={{borderTop:`4px solid ${selProject.color}`,marginBottom:"14px"}}>
              <div style={{display:"flex",alignItems:"center",gap:"12px",marginBottom:"14px"}}>
                <span style={{fontSize:"36px"}}>{selProject.emoji}</span>
                <div>
                  <h2 style={{margin:"0 0 6px",fontSize:"20px",color:selProject.color}}>{selProject.title}</h2>
                  <div style={{display:"flex",gap:"7px",flexWrap:"wrap"}}>
                    <Tag label={selProject.difficulty} color={diffColor(selProject.difficulty)}/>
                    <Tag label={`⏱ ${selProject.time}`} color={P.muted}/>
                  </div>
                </div>
              </div>
              <div style={{display:"grid",gap:"10px",marginBottom:"16px"}}>
                <div style={{background:P.card2,borderRadius:"8px",padding:"12px"}}>
                  <div style={{fontSize:"11px",color:P.muted,marginBottom:"4px"}}>📦 Dataset</div>
                  <div style={{color:P.text,fontSize:"13px"}}>{selProject.dataset}</div>
                  <div style={{color:selProject.color,fontSize:"11px",marginTop:"4px"}}>🔗 {selProject.datasetUrl}</div>
                </div>
                <div style={{background:P.card2,borderRadius:"8px",padding:"12px"}}>
                  <div style={{fontSize:"11px",color:P.muted,marginBottom:"4px"}}>🎯 Outcome</div>
                  <div style={{color:P.text,fontSize:"13px"}}>{selProject.outcome}</div>
                </div>
                <div style={{background:P.card2,borderRadius:"8px",padding:"12px"}}>
                  <div style={{fontSize:"11px",color:P.muted,marginBottom:"6px"}}>📋 Steps</div>
                  <ol style={{margin:0,paddingLeft:"18px"}}>
                    {selProject.steps.map((s,i)=>(
                      <li key={i} style={{color:P.text,fontSize:"12px",marginBottom:"4px"}}>{s}</li>
                    ))}
                  </ol>
                </div>
                <div style={{background:P.card2,borderRadius:"8px",padding:"12px"}}>
                  <div style={{fontSize:"11px",color:P.muted,marginBottom:"6px"}}>🛠️ Skills Practised</div>
                  <div style={{display:"flex",flexWrap:"wrap",gap:"5px"}}>
                    {selProject.skills.map(s=><Tag key={s} label={s} color={selProject.color}/>)}
                  </div>
                </div>
              </div>
            </Card>
            <div style={{background:P.card,border:`1px solid ${P.border}`,borderRadius:"12px",padding:"20px"}}>
              <h3 style={{margin:"0 0 14px",color:selProject.color}}>📓 Google Colab Notebook</h3>
              <div style={{background:`${P.accent3}0e`,border:`1px solid ${P.accent3}30`,
                borderRadius:"8px",padding:"10px 14px",fontSize:"12px",color:P.accent3,marginBottom:"14px"}}>
                📌 Copy each cell into Google Colab and run top-to-bottom. Adjust DATA_DIR to your dataset path.
              </div>
              <CodeBlock code={selProject.code} id={`proj-${selProject.id}`}/>
            </div>
          </div>
        )}

        {/* ══════════════ HOSTING ══════════════ */}
        {tab==="hosting" && !selHosting && (
          <div>
            <div style={{background:`linear-gradient(135deg,${P.accent1}0a,${P.accent2}0a)`,
              border:`1px solid ${P.accent1}25`,borderRadius:"12px",padding:"18px",marginBottom:"20px"}}>
              <h3 style={{margin:"0 0 8px",color:P.accent1}}>🗺️ Pick Your Deployment Strategy</h3>
              <p style={{margin:0,color:P.muted,fontSize:"13px",lineHeight:"1.7"}}>
                Your choice depends on three questions: <strong style={{color:P.text}}>Who is your audience?</strong> (just you, collaborators, or the public), <strong style={{color:P.text}}>What's your compute budget?</strong> (free vs paid GPU), and <strong style={{color:P.text}}>Do you need a REST API or a UI?</strong>
              </p>
            </div>
            <div style={{display:"grid",gap:"14px"}}>
              {HOSTING.map((h,i)=>(
                <div key={h.platform} onClick={()=>setSelHosting(h)}
                  style={{background:P.card,border:`1px solid ${P.border}`,borderRadius:"12px",
                    padding:"18px 20px",cursor:"pointer",borderLeft:`5px solid ${C(i)}`}}>
                  <div style={{display:"flex",alignItems:"center",justifyContent:"space-between"}}>
                    <div style={{display:"flex",alignItems:"center",gap:"10px"}}>
                      <span style={{fontSize:"24px"}}>{h.icon}</span>
                      <div>
                        <div style={{fontWeight:700,fontSize:"15px",color:P.text}}>{h.platform}</div>
                        <div style={{fontSize:"11px",color:C(i)}}>{h.cost}</div>
                      </div>
                    </div>
                    <span style={{color:P.accent1,fontSize:"18px"}}>→</span>
                  </div>
                  <p style={{margin:"8px 0 10px",color:P.muted,fontSize:"12px"}}>{h.bestFor}</p>
                  <div style={{display:"flex",gap:"8px",flexWrap:"wrap"}}>
                    {h.pros.slice(0,3).map(pr=>(
                      <span key={pr} style={{background:`${C(i)}12`,color:C(i),fontSize:"10px",
                        padding:"2px 8px",borderRadius:"5px"}}>✓ {pr}</span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {tab==="hosting" && selHosting && (
          <div>
            <button onClick={()=>setSelHosting(null)}
              style={{background:"none",border:`1px solid ${P.border}`,color:P.muted,
                padding:"6px 14px",borderRadius:"8px",cursor:"pointer",marginBottom:"16px",fontSize:"12px"}}>
              ← All Options
            </button>
            {(()=>{const ci=HOSTING.findIndex(h=>h.platform===selHosting.platform);const col=C(ci);return(
              <Card style={{borderTop:`4px solid ${col}`,marginBottom:"14px"}}>
                <div style={{display:"flex",alignItems:"center",gap:"12px",marginBottom:"16px"}}>
                  <span style={{fontSize:"36px"}}>{selHosting.icon}</span>
                  <div>
                    <h2 style={{margin:"0 0 4px",fontSize:"20px",color:col}}>{selHosting.platform}</h2>
                    <Tag label={selHosting.cost} color={col}/>
                  </div>
                </div>
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"10px",marginBottom:"16px"}}>
                  <div style={{background:P.card2,borderRadius:"8px",padding:"12px"}}>
                    <div style={{fontSize:"11px",color:P.muted,marginBottom:"6px"}}>✅ Pros</div>
                    {selHosting.pros.map(p=><div key={p} style={{color:P.text,fontSize:"12px",marginBottom:"3px"}}>• {p}</div>)}
                  </div>
                  <div style={{background:P.card2,borderRadius:"8px",padding:"12px"}}>
                    <div style={{fontSize:"11px",color:P.muted,marginBottom:"6px"}}>⚠️ Cons</div>
                    {selHosting.cons.map(c=><div key={c} style={{color:P.muted,fontSize:"12px",marginBottom:"3px"}}>• {c}</div>)}
                  </div>
                </div>
                <div style={{background:P.card2,borderRadius:"8px",padding:"12px",marginBottom:"16px"}}>
                  <div style={{fontSize:"11px",color:P.muted,marginBottom:"6px"}}>🚀 Step-by-step</div>
                  <ol style={{margin:0,paddingLeft:"18px"}}>
                    {selHosting.steps.map((s,i)=><li key={i} style={{color:P.text,fontSize:"12px",marginBottom:"4px"}}>{s}</li>)}
                  </ol>
                </div>
                <h4 style={{margin:"0 0 10px",color:col}}>💻 Ready-to-use Code</h4>
                <CodeBlock code={selHosting.code} id={`host-${selHosting.platform}`}/>
              </Card>
            );})()}
          </div>
        )}

        {/* ══════════════ QUIZ ══════════════ */}
        {tab==="quiz" && (
          <div>
            {!quiz.active && !quiz.done && (
              <div style={{textAlign:"center",padding:"48px 20px"}}>
                <div style={{fontSize:"64px",marginBottom:"14px"}}>🧪</div>
                <h2 style={{color:P.accent1,margin:"0 0 8px",fontSize:"22px"}}>Master Quiz</h2>
                <p style={{color:P.muted,fontSize:"14px",margin:"0 0 6px"}}>{QUIZ_BANK.length} questions spanning CV fundamentals, coding challenges, and project skills.</p>
                <p style={{color:P.muted,fontSize:"12px",margin:"0 0 28px"}}>Type your answer — partial matches accepted. Open-ended, not multiple choice.</p>
                <button onClick={()=>setQuiz({active:true,q:0,score:0,done:false,answered:false,correct:null})}
                  style={{background:P.accent1,color:"#000",border:"none",borderRadius:"10px",
                    padding:"14px 36px",fontSize:"15px",fontWeight:700,cursor:"pointer"}}>
                  Start Quiz →
                </button>
              </div>
            )}

            {quiz.active && !quiz.done && (
              <div style={{maxWidth:"600px",margin:"0 auto"}}>
                <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"14px"}}>
                  <span style={{color:P.muted,fontSize:"13px"}}>Question {quiz.q+1} / {QUIZ_BANK.length}</span>
                  <span style={{background:`${P.accent3}20`,color:P.accent3,padding:"3px 12px",borderRadius:"6px",fontSize:"13px"}}>Score: {quiz.score}</span>
                </div>
                <div style={{height:"4px",background:P.border,borderRadius:"2px",marginBottom:"20px"}}>
                  <div style={{height:"4px",borderRadius:"2px",background:P.accent1,
                    width:`${(quiz.q/QUIZ_BANK.length)*100}%`,transition:"width .4s"}}/>
                </div>
                <Card>
                  <h3 style={{color:P.text,margin:"0 0 20px",fontSize:"16px",lineHeight:"1.5"}}>
                    {QUIZ_BANK[quiz.q].q}
                  </h3>
                  {!quiz.answered?(
                    <>
                      <input value={qAns} onChange={e=>setQAns(e.target.value)}
                        onKeyDown={e=>e.key==="Enter"&&qAns.trim()&&submitQ()}
                        placeholder="Type your answer…"
                        style={{width:"100%",background:"#0b0f1a",border:`1px solid ${P.border}`,
                          borderRadius:"8px",padding:"11px 14px",color:P.text,fontSize:"14px",
                          boxSizing:"border-box",outline:"none",marginBottom:"12px"}}
                        autoFocus/>
                      <button onClick={submitQ} disabled={!qAns.trim()}
                        style={{background:qAns.trim()?P.accent1:P.border,
                          color:qAns.trim()?"#000":P.muted,border:"none",borderRadius:"8px",
                          padding:"10px 24px",fontSize:"14px",fontWeight:600,
                          cursor:qAns.trim()?"pointer":"default"}}>
                        Submit
                      </button>
                    </>
                  ):(
                    <>
                      <div style={{background:quiz.correct?`${P.accent3}15`:`${P.accent4}15`,
                        border:`1px solid ${quiz.correct?P.accent3:P.accent4}40`,
                        borderRadius:"8px",padding:"14px",marginBottom:"16px"}}>
                        <div style={{fontSize:"18px",marginBottom:"6px"}}>{quiz.correct?"✅ Correct!":"❌ Not quite"}</div>
                        <div style={{color:P.text,fontSize:"13px"}}>
                          <strong>Answer:</strong> {QUIZ_BANK[quiz.q].a}<br/>
                          <strong>Hint:</strong> {QUIZ_BANK[quiz.q].hint}
                        </div>
                      </div>
                      <button onClick={nextQ}
                        style={{background:P.accent2,color:"#fff",border:"none",borderRadius:"8px",
                          padding:"10px 24px",fontSize:"14px",fontWeight:600,cursor:"pointer"}}>
                        {quiz.q+1>=QUIZ_BANK.length?"See Results →":"Next →"}
                      </button>
                    </>
                  )}
                </Card>
              </div>
            )}

            {quiz.done && (
              <div style={{textAlign:"center",padding:"40px 20px"}}>
                <div style={{fontSize:"64px",marginBottom:"12px"}}>
                  {quiz.score>=17?"🏆":quiz.score>=12?"🥈":quiz.score>=8?"🥉":"📚"}
                </div>
                <h2 style={{color:quiz.score>=17?P.accent3:quiz.score>=12?P.accent1:P.accent4,margin:"0 0 8px",fontSize:"22px"}}>
                  {quiz.score}/{QUIZ_BANK.length} — {quiz.score>=17?"Outstanding!":quiz.score>=12?"Great work!":quiz.score>=8?"Good effort!":"Keep going!"}
                </h2>
                <div style={{background:P.card,border:`1px solid ${P.border}`,borderRadius:"12px",
                  padding:"24px 40px",display:"inline-block",margin:"16px 0 24px"}}>
                  <div style={{fontSize:"52px",fontWeight:800,
                    color:quiz.score>=17?P.accent3:quiz.score>=12?P.accent1:P.accent4}}>
                    {Math.round((quiz.score/QUIZ_BANK.length)*100)}%
                  </div>
                  <div style={{color:P.muted,fontSize:"13px"}}>Final Score</div>
                </div>
                <br/>
                <button onClick={()=>setQuiz({active:false,q:0,score:0,done:false,answered:false,correct:null})}
                  style={{background:P.surface,border:`1px solid ${P.border}`,color:P.text,
                    borderRadius:"8px",padding:"10px 22px",fontSize:"13px",cursor:"pointer",marginRight:"10px"}}>
                  Retake
                </button>
                <button onClick={()=>setTab("challenges")}
                  style={{background:P.accent1,color:"#000",border:"none",borderRadius:"8px",
                    padding:"10px 22px",fontSize:"13px",fontWeight:700,cursor:"pointer"}}>
                  Try Challenges →
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* FOOTER */}
      <div style={{borderTop:`1px solid ${P.border}`,padding:"14px 24px",textAlign:"center",
        color:P.muted,fontSize:"11px",marginTop:"28px"}}>
        CV Tutorial v3 · {CHALLENGES.length} Coding Challenges · {PROJECTS.length} Real-World Projects · 5 Hosting Options · {QUIZ_BANK.length}-Question Quiz
      </div>
    </div>
  );
}
