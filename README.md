# VGG Feature-based Template Matching with Refinement
基于 VGG 特征的模板匹配与精修算法

---

## 1. Introduction | 简介
This repository implements a template matching algorithm that combines **deep feature correlation** (using a modified VGG16) with **local feature refinement** (e.g., SIFT-based corner matching).  
该代码实现了一种模板匹配方法：先利用改进的 VGG16 网络进行深度特征相似性计算，再通过局部特征点匹配进行精修，最终得到图像对之间的区域对应关系。  

---

## 2. Algorithm Workflow | 算法流程

The complete pipeline can be described in the following steps:  
算法整体流程如下：

### Step 1. Region Splitting | 区域划分
- Input image A is divided into **M×N sub-regions** (default: 18×12).  
- Each region (≈100×100 pixels) is treated as a **template**.  
- Edge portion (10%) is ignored to reduce noise.  

### Step 2. Template Preparation | 模板准备
- Each template region is converted into a tensor.  
- From image B, a horizontal strip corresponding to the template’s vertical span is extracted.  
- The strip is vertically expanded (default: ±40%) to ensure the true match is included.  

### Step 3. Deep Feature Extraction | 深度特征提取
- Both the template and the candidate region from image B are passed through a **VGG16 network without pooling** (`CustomVGG16NoPooling`).  
- This preserves spatial resolution while extracting semantic features.  

### Step 4. Feature Correlation (Convolution Matching) | 特征卷积匹配
- The template feature map is treated as a kernel.  
- A convolution (`F.conv2d`) is performed on the feature map of image B.  
- The result is a **response map**, indicating similarity between the template and each location in B.  

### Step 5. Response Map Resizing & Placement | 响应图缩放与拼接
- The response map is resized to match the candidate region size.  
- It is inserted into a zero-initialized matrix with the same size as image B, forming a **full-image response map**.  

### Step 6. Maximum Response Localization | 最大响应定位
- The location with the **highest response** is selected as the match.  
- A border exclusion (e.g., 5%) ensures the match does not fall on image edges.  

### Step 7. Local Feature Refinement | 局部特征精修
- Around the template region in A and the matched region in B, local descriptors (e.g., SIFT) are extracted.  
- Corner matching is performed to refine alignment and reject outliers.  
- The refined corner correspondences are saved.  

### Step 8. Result Storage & Visualization | 结果保存与可视化
- Matching results are appended to `res_vgg_sift.txt`.  
- Optional visualization shows:
  - Template in A  
  - Matched location in B  
  - Refined corner correspondences  

---

## 3. Input & Output | 输入与输出

### Input
- **Image A** (reference image, divided into regions)  
- **Image B** (target image, search space)  
- Pre-trained weights: `saved_pt/vgg16_cov.pth`  

### Output
- Response maps for each region  
- Best match location per template  
- Refined corner correspondences  
- Log file: `res_vgg_sift.txt`  

---

## 4. Applications | 应用场景
- **Image Registration**: Aligning two images with local distortions.  
- **Object Tracking**: Tracking small templates across frames.  
- **Stereo Matching**: Finding correspondences between stereo image pairs.  
- **Change Detection**: Locating region shifts between sequential images.  

---

## 5. Key Advantages | 方法优势
- **High resolution features**: No pooling layers → better localization.  
- **Robust to appearance changes**: Deep VGG features capture semantics.  
- **Refinement with local descriptors**: Combines deep learning with classical vision.  

---

## 6. Workflow Diagram | 流程图

```mermaid
flowchart TD
    A[Input Image A] --> B[Split into Regions]
    B --> C[Template Selection]
    A2[Input Image B] --> D[Extract Candidate Region with Expansion]
    C --> E[Deep Feature Extraction with VGG16 No Pooling]
    D --> E
    E --> F[Feature Correlation (Conv2D)]
    F --> G[Response Map Resizing & Placement]
    G --> H[Maximum Response Localization]
    H --> I[Local Feature Refinement (SIFT Corners)]
    I --> J[Save & Visualize Results]
```

---

## 7. License | 许可证
MIT License. Free for research and educational use.  
MIT 协议，支持科研与教育使用。
