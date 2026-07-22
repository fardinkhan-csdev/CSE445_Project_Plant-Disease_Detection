# Plant Disease Model Pipeline: PlantVillage to PlantDoc

This document outlines the dataset sizes, structural differences, and a step-by-step implementation strategy for training a plant disease diagnosis model on laboratory data (**PlantVillage**) and testing it on real-world field data (**PlantDoc**).

---

## 1. Dataset Reference Sheet

### PlantDoc Dataset Overview
* **Primary Task:** Object Detection (bounding boxes around multiple leaves)
* **Total Images:** ~2,569 to 2,598 images
* **Total Annotations:** 8,800+ labeled leaf objects
* **Classes:** ~30 distinct healthy and diseased plant classes across 13 species
* **File Size Variations:**
  * **Original Version (~992 MB):** Hosted on GitHub via Pascal VOC XML annotation formats. Contains redundant web-scraping metadata and raw image headers.
  * **Roboflow Version (~980 MB):** Standardized into optimized text formats (YOLO text strings/COCO JSON). Cleaned of system artifacts and structured cleanly into train/validation/test splits.

### PlantVillage Dataset Overview
* **Primary Task:** Image Classification (one pristine centered leaf per image)
* **Total Images:** 54,306 images
* **Classes:** 38 distinct healthy and diseased classes across 14 crop species
* **Official Processing Versions:**
  1. **Color (Original RGB) [~815 MB - 1 GB]:** Raw photographs of single leaves captured in controlled lab environments against varying gray or black backdrops.
  2. **Grayscale:** 8-bit black-and-white conversion to isolate structural deformities and patterns while eliminating color bias.
  3. **Segmented:** Backgrounds digitally stripped out, leaving only raw leaf tissue on a solid black canvas to prevent the model from memorizing lab environments.

---

## 2. Cross-Dataset Discrepancy Matrix

| Feature | PlantVillage (Source/Train) | PlantDoc (Target/Test) |
| :--- | :--- | :--- |
| **Data Volume** | Massive (~54,300 images) | Small (~2,500 images) |
| **Environment** | Controlled laboratory setting | Chaotic, real-world fields |
| **Visual Focus** | Single leaf per image, perfectly centered | Multiple leaves, stems, soil, and weeds |
| **Task Design** | Image Classification | Object Detection |

---

## 3. The 3 Major Architectural Hurdles

### I. Task Mismatch
Your model will be trained on image classification, meaning it expects a single leaf and outputs one class label. PlantDoc provides whole-scene images with multiple overlapping targets, which will break standard classification inference unless pre-processed.

### II. Class Alignment Gap
The two datasets do not share identical class taxonomies. Unique classes in PlantVillage (e.g., *Squash* or *Strawberry*) must be dropped, and variations in naming syntax must be programmatically mapped to a uniform standard.

### III. Severe Domain Shift
A model trained exclusively on uniform laboratory backdrops will experience a massive accuracy drop when deployed on field images. It will struggle to isolate leaf boundaries from background noise like dirt, grass, harsh shadows, and variable outdoor lighting.

---

## 4. End-to-End Implementation Strategy

To successfully train on PlantVillage and evaluate on PlantDoc, execute the following technical workflow:

[ PlantVillage Segmented ]│▼[ Heavy Augmentations ] ──► (Random Backgrounds, Hue/Contrast Jitter)│▼[ Train Model ] (Image Classification)│▼[ Crop PlantDoc Bounding Boxes ] ──► [ Generate Uniform Test Set ] ──► [ Evaluate Accuracy ]

AI Mode conversation: plantdoc dataset sizeYou said: plantdoc dataset sizeplantdoc dataset sizeThe PlantDoc dataset consists of approximately 2,500 to 2,600 images, encompassing 13 different plant species and around 27 to 30 classes for both classification and object detection tasks. The raw dataset size is around 1 GB, while various augmented versions can reach up to 4.5 GB.Key Dataset DetailsTotal Images: ~2,569 to 2,598 images (with bounding box annotations)File Size: ~992 MB for the original version, ~980 MB for the Roboflow datasetPlant Species: 13 major types (including corn, tomato, apple, grape, potato, and bell pepper)Classes: ~30 diseased and healthy leaf classes (with about 8,800+ labeled objects)Task Focus: Primarily used for training visual object detection and image classification modelsExplore and download the dataset via the PlantDoc Hugging Face Dataset or the PlantDoc Object Detection Dataset on Roboflow.Would you like to know about:Pre-processing options (e.g., resizing to 416x416 or 100x100)?Available dataset splits (e.g., train vs. test ratios)?Model recommendations for training on this dataset (like MobileNet or YOLOv5)?8 sitesPlantDoc - Dataset NinjaThe authors of the PlantDoc: A Dataset for Visual Plant Disease Detection recognized the importance of training models with real-l...Dataset NinjaPlantDoc Augmented Dataset - KaggleDataset Specifications * Total Size: ~4.5 GB (compressed ZIP) * Storage Format: Hierarchical directory structure. * Naming Convent...KagglePlantDoc Object Detection Dataset by Singh et. al 2019About PlantDoc Dataset. Overview. The PlantDoc dataset was originally published by researchers at the Indian Institute of Technolo...Roboflow UniverseShow allFile Size: ~992 MB for the original version, ~980 MB for the Roboflow dataset

difference?The difference in file size (~992 MB vs. ~980 MB) stems from formatting optimizations, annotation types, and file structuring, rather than a difference in the actual image content. Both versions share the same base pool of 2,569 original images.The key technical variations between the two versions include:1. Annotation FormatsOriginal Version: The original release on the Pratik Kayal GitHub Repository stores labels in older, uncompressed, or verbose formats (like Pascal VOC XML). These text files generate a slightly larger overall storage footprint.Roboflow Version: The Roboflow Universe PlantDoc Page modernizes these labels into highly optimized, lightweight formats (such as YOLO text strings or COCO JSON files). This strips out redundant metadata and lowers the file size.2. Standardized Train/Test/Validation SplitsOriginal Version: The raw dataset is often distributed as one massive folder of images paired with individual annotation files, leaving splitting up to the researcher.Roboflow Version: Roboflow automatically partitions the data into structured train, valid, and test directories. This automated directory structure and index generation slightly alters the final zip overhead.3. Metadata and Hidden FilesOriginal Version: Often contains additional repository files, unoptimized image headers, or original web-scraping metadata from Google and Ecosia image indexes.Roboflow Version: Cleans out non-image files, system artifacts, and standardized image metadata headers upon ingestion.(Note: Roboflow also allows you to download a downsized 416x416 resolution export. If you choose that option, the file size will drop dramatically below 980 MB to save disk space and accelerate training).Would you like to know how to download the Roboflow version directly using a Python script, or do you need assistance converting the original annotations to a specific format like YOLOv8 or COCO?6 sitesPlantDoc Object Detection Dataset by Singh et. al 2019Overview. The PlantDoc dataset was originally published by researchers at the Indian Institute of Technology, and described in dep...Roboflow UniverseDataset size vs number of images - 🤝 Community Help - RoboflowApr 11, 2025 — However, when I look at the dataset tab, I see 267 images. Which one is accurate? What does each number reflect? ... if you are lo...RoboflowImproved PlantDoc Dataset for Plant Disease Object DetectionFeb 25, 2026 — How to Train a Plant Disease Detection Model on PlantDoc. You can go from this dataset to a trained model entirely in Roboflow, no...Roboflow BlogShow allwhat about plantvillage versions?Unlike the PlantDoc dataset, which is a small object-detection dataset (~2,500 images), PlantVillage is a massive, widely recognized benchmark dataset containing over 54,300 images.The Official PlantVillage GitHub Repository breaks down the collection into three core processing versions, alongside several third-party curated variations.The 3 Official VersionsAll three versions contain exactly 54,306 images covering 14 crop species and 38 distinct healthy/diseased classes:Color (Original RGB):What it is: The raw, unedited camera photographs of single leaves.Condition: Captured in controlled lab environments against varying solid gray or black backgrounds.Size: Around 815 MB to 1 GB depending on host packaging (e.g., TensorFlow Datasets vs Kaggle).Grayscale:What it is: The exact same image set converted entirely into 8-bit black-and-white images.Purpose: Used to train models to focus strictly on structural deformities, texture, and shapes, removing any color-based bias.Segmented:What it is: Digitally processed images where the background is completely stripped away.Condition: Only the raw leaf tissue is preserved on a stark black background to prevent the neural network from accidentally "memorizing" background noise or lighting conditions.Popular Third-Party VersionsBecause the original plantvillage.org host went offline, researchers have created heavily modified versions hosted elsewhere:Dataset VersionApprox. File SizeWhat Changed?Mendeley Data (Updated)1.2 GB to 2 GBExpanded to 61,486 images by adding 6 types of augmentations (gamma correction, flipping, noise) and a "Background without leaves" negative control class.Meta-Album (Micro/Mini)< 100 MBResized specifically for rapid benchmarking down to 128x128 pixels (from the native 256x256 resolution).Kaggle (Compressed)344 MBThe 54k color files packaged with highly optimized JPEG compression to save disk space.Summary Comparison: PlantDoc vs. PlantVillagePlantDoc: Small (~2,500 images) | Focuses on object detection (bounding boxes around multiple leaves in complex wild environments).PlantVillage: Large (~54,300 images) | Focuses on image classification (one pristine leaf centered per image in a laboratory setting).Which version of PlantVillage matches your current hardware constraints, or are you looking for a data loader script to plug one of these directly into PyTorch/TensorFlow?15 sitesDisplays sample images extracted from the three distinct versions of...The complete PlantVillage dataset encompasses three distinct versions, each offering unique variations and additions. (1) Color-54...ResearchGateGitHub - spMohanty/PlantVillage-Dataset: Dataset of diseased plant leaf images and corresponding labelsThe PlantVillage Dataset is a publicly available repository of 54,306 images of healthy and diseased plant leaves. The dataset is ...GitHubMeta-Album Plant Village Dataset (Micro) - OpenMLThe original image resolution is 256x256 px. This collection covers 26 plant diseases and 12 healthy plants. The leaves are remove...OpenMLShow alli want to train on village, then test on doc.This approach—training on PlantVillage and testing on PlantDoc—is an excellent way to evaluate how well your model generalizes to real-world scenarios.However, you will face an immediate drop in accuracy because of a severe domain gap and task mismatch. PlantVillage images are clean, laboratory-grade snapshots of single leaves on solid backgrounds, whereas PlantDoc images are messy, crowded, real-world field photographs.Here is a breakdown of the specific challenges you will encounter and a step-by-step strategy to make this training pipeline succeed.The 3 Major Hurdles1. Task Mismatch (Classification vs. Object Detection)PlantVillage: Designed for image classification. The dataset provides a single image of a leaf and a corresponding label (e.g., "Tomato_Early_blight").PlantDoc: Designed for object detection. It contains complex scenes with multiple leaves, branches, and soil, requiring bounding boxes ([x, y, w, h]) around specific infected areas.2. Class Alignment IssuesThe two datasets do not share identical class names. You will need to filter and map them manually.Overlapping Classes: You must drop any unique PlantVillage classes (like Squash or Strawberry) that do not exist in the PlantDoc test set.Label Mapping: You will need to map specific granular classes to broader ones. For example, mapping PlantVillage's Tomato___Bacterial_spot directly to PlantDoc's Tomato Bacterial Spot.3. Image Backgrounds (The Domain Gap)A model trained only on PlantVillage's solid gray/black backgrounds will struggle heavily on PlantDoc because it never learned to ignore background noise like soil, weeds, sky, or human hands.Step-by-Step Implementation StrategyStep 1: Align the LabelsDo not throw the entire PlantVillage dataset into your model. Filter it down to only the crops and diseases present in PlantDoc. Write a custom data pipeline to map the folder names of PlantVillage to match the exact string classes of PlantDoc.Step 2: Use the "Segmented" PlantVillage VersionTo combat the background domain gap, train your model using the Segmented version of PlantVillage. Because this version strips away the lab backgrounds and leaves only raw leaf tissue on a pure black canvas, it forces your deep learning model to learn leaf textures and lesion shapes rather than memorizing laboratory lighting.Step 3: Heavy Data AugmentationTo prepare your model for the chaotic nature of PlantDoc, you must aggressively augment PlantVillage during training. Apply these specific transforms using libraries like Albumentations or Torchvision:Random Background Insertion: Replace the black backgrounds of the segmented leaves with random outdoor textures (grass, dirt, gravel).Random Scaling and Cropping: Shrink the leaves randomly to simulate a camera being far away in a field.Brightness, Contrast, and Hue Jitter: Simulate natural, uneven sunlight and harsh shadows.Step 4: Convert PlantDoc to a Classification Test SetBecause your model trained on PlantVillage will output an image-level class label (and not bounding boxes), you must prepare PlantDoc as a classification test set:Parse the PlantDoc XML/JSON annotation files.Use the bounding box coordinates to crop out individual leaves from the PlantDoc images.Save these cropped leaf images into folders corresponding to their respective disease labels.Pass these cropped images to your trained model to evaluate its final classification accuracy.Would you like a Python script using OpenCV and Pandas to automatically crop the PlantDoc bounding boxes into a classification test set, or do you need the exact class mapping list between the two datasets?no, i need an md file with all these info. can you create an .md file?You can copy and save the content block below directly as a .md file (e.g., plant_dataset_pipeline.md). It compiles all the data, differences, architectural challenges, and implementation workflows discussed.markdown# Plant Disease Model Pipeline: PlantVillage to PlantDoc

This document outlines the dataset sizes, structural differences, and a step-by-step implementation strategy for training a plant disease diagnosis model on laboratory data (**PlantVillage**) and testing it on real-world field data (**PlantDoc**).

---

## 1. Dataset Reference Sheet

### PlantDoc Dataset Overview
* **Primary Task:** Object Detection (bounding boxes around multiple leaves)
* **Total Images:** ~2,569 to 2,598 images
* **Total Annotations:** 8,800+ labeled leaf objects
* **Classes:** ~30 distinct healthy and diseased plant classes across 13 species
* **File Size Variations:**
  * **Original Version (~992 MB):** Hosted on GitHub via Pascal VOC XML annotation formats. Contains redundant web-scraping metadata and raw image headers.
  * **Roboflow Version (~980 MB):** Standardized into optimized text formats (YOLO text strings/COCO JSON). Cleaned of system artifacts and structured cleanly into train/validation/test splits.

### PlantVillage Dataset Overview
* **Primary Task:** Image Classification (one pristine centered leaf per image)
* **Total Images:** 54,306 images
* **Classes:** 38 distinct healthy and diseased classes across 14 crop species
* **Official Processing Versions:**
  1. **Color (Original RGB) [~815 MB - 1 GB]:** Raw photographs of single leaves captured in controlled lab environments against varying gray or black backdrops.
  2. **Grayscale:** 8-bit black-and-white conversion to isolate structural deformities and patterns while eliminating color bias.
  3. **Segmented:** Backgrounds digitally stripped out, leaving only raw leaf tissue on a solid black canvas to prevent the model from memorizing lab environments.

---

## 2. Cross-Dataset Discrepancy Matrix

| Feature | PlantVillage (Source/Train) | PlantDoc (Target/Test) |
| :--- | :--- | :--- |
| **Data Volume** | Massive (~54,300 images) | Small (~2,500 images) |
| **Environment** | Controlled laboratory setting | Chaotic, real-world fields |
| **Visual Focus** | Single leaf per image, perfectly centered | Multiple leaves, stems, soil, and weeds |
| **Task Design** | Image Classification | Object Detection |

---

## 3. The 3 Major Architectural Hurdles

### I. Task Mismatch
Your model will be trained on image classification, meaning it expects a single leaf and outputs one class label. PlantDoc provides whole-scene images with multiple overlapping targets, which will break standard classification inference unless pre-processed.

### II. Class Alignment Gap
The two datasets do not share identical class taxonomies. Unique classes in PlantVillage (e.g., *Squash* or *Strawberry*) must be dropped, and variations in naming syntax must be programmatically mapped to a uniform standard.

### III. Severe Domain Shift
A model trained exclusively on uniform laboratory backdrops will experience a massive accuracy drop when deployed on field images. It will struggle to isolate leaf boundaries from background noise like dirt, grass, harsh shadows, and variable outdoor lighting.

---

## 4. End-to-End Implementation Strategy

To successfully train on PlantVillage and evaluate on PlantDoc, execute the following technical workflow:

Use code with caution.[ PlantVillage Segmented ]│▼[ Heavy Augmentations ] ──► (Random Backgrounds, Hue/Contrast Jitter)│▼[ Train Model ] (Image Classification)│▼[ Crop PlantDoc Bounding Boxes ] ──► [ Generate Uniform Test Set ] ──► [ Evaluate Accuracy ]
### Step 1: Filter and Map Classes
Examine the folders in both datasets. Drop any classes unique to PlantVillage. Create a Python dictionary mapper to align string labels (e.g., mapping PlantVillage's `Tomato___Bacterial_spot` directly to PlantDoc's `Tomato Bacterial Spot`).

### Step 2: Leverage "Segmented" Data
Utilize the **Segmented** version of the PlantVillage dataset for training. Because the background is a consistent black canvas, it forces the neural network to focus its weight optimization purely on leaf morphology, lesion color, and tissue damage patterns.

### Step 3: Apply Domain-Agnostic Augmentations
During your PyTorch/TensorFlow training loops, implement aggressive augmentations using libraries like `Albumentations`:
* **Background Blending:** Replace the solid black background of the segmented leaves with random outdoor textures (soil, grass, gravel) to simulate field conditions.
* **Spatial Distortion:** Use random scaling, cropping, and rotations to account for varying camera distances.
* **Photometric Jitter:** Heavily adjust brightness, contrast, and hue to mimic varying solar conditions and camera sensors.

### Step 4: Convert PlantDoc into a Classification Test Set
Because your model outputs classification categories rather than bounding coordinates, you must alter the testing data:
1. Parse the PlantDoc annotation source files (XML or JSON formats).
2. Read the coordinate bounds `[xmin, ymin, xmax, ymax]` for every annotated leaf object.
3. Use an image processing library (like OpenCV or PIL) to **crop out individual leaves** based on those coordinates.
4. Export these cropped leaf fragments into categorized subfolders corresponding to your aligned labels.
5. Feed these cropped real-world images into your trained classification model to measure final testing accuracy.