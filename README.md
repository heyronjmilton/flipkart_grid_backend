# Automated Grocery Quality Assessment using Computer Vision and OCR

This project leverages advanced computer vision models and OCR technology to automate the quality assessment of groceries, ensuring accuracy in both packed goods and fresh produce. It consists of three models: an object detection model trained on 49 classes to identify various packed items, a freshness detection model for fruits and vegetables with indices of good, average, and bad, and a segmentation model to extract important details like batch numbers, manufacturing, and expiration dates from product packaging. A custom dataset was created due to the limitations of publicly available ones, and the models were trained on an NVIDIA RTX Ada 4000 series GPU.

The project’s pipeline begins with object detection, followed by segmenting the region of interest using the segmentation model. OCR is employed through Gemini API to extract textual details like expiration dates. A freshness index is calculated for each detected fruit, and a real-time item count is reflected on the frontend for the operator. This system improves efficiency in tracking product quality, offering a seamless user interface for visualizing real-time data and detailed insights.

# Dataset Generation and Training Process for Grocery Quality Assessment Project

## 1. Challenges with Existing Datasets

The publicly available datasets for grocery items, fresh produce, and packaging details did not meet the specific requirements for the project. These datasets often lacked the diversity, detail, and context needed for accurate object detection, freshness assessment, and extraction of critical text (e.g., batch numbers, manufacturing and expiration dates) from packaging. Hence, the team decided to generate a custom dataset, tailored to the unique goals of the project.

## 2. Custom Dataset Generation

### a) Object Detection Model Dataset

The first model was designed to detect packed grocery items. It required a dataset that could recognize various products across different conditions.

- **Individual Item Photos:** The team captured separate individual photos for each of the 49 packed items. To ensure diversity, the images were taken under varied angles, lighting conditions, and backgrounds. This step was crucial because objects in real-life environments often appear with different visual properties, so the model needed to generalize well.
- **Diverse Backgrounds and Lighting:** By photographing items in different environments—indoors, outdoors, in natural light, and artificial light—the dataset was made robust enough to account for common variations. For example, an item placed on a kitchen counter might have different lighting than the same item placed on a grocery shelf.
- **Grouped Item Photos:** Beyond individual item captures, the dataset included mixed images where multiple items were grouped together in the frame. This added complexity to the task, ensuring the model could recognize individual items within larger sets of products, making the object detection more versatile and practical for real-world grocery applications.
- **Manual Annotation:** Each of the images was manually annotated using tools like LabelImg, ensuring the bounding boxes were accurately drawn around the items of interest. The precise annotations were necessary for the object detection model to learn the spatial characteristics of each product accurately.

### b) Freshness Detection Model Dataset

For the freshness detection model, the aim was to classify fruits and vegetables based on freshness indices: good, average, and bad. Creating this dataset required careful planning and real-world scenarios:

- **Freshness Index Photos:** The team collected images of fruits and vegetables in various states of freshness. Fresh produce was documented over time, from the day of harvest (good) through gradual deterioration (average) until signs of spoilage (bad). Each item was labeled accordingly.
- **Environmental Variations:** Just like with the packed items, the images were captured under different lighting and background settings. Freshness indicators, like slight color changes or texture alterations, can be subtle, so including various environmental conditions ensured the model could accurately assess freshness in diverse scenarios.
- **Manual Annotation:** Similar to the object detection model, annotations were created to highlight the fruits and vegetables in the images. These annotations were combined with freshness labels (good, average, bad) for supervised training.

### c) Segmentation Model Dataset

For the segmentation model, the goal was to detect and extract text regions like batch numbers, manufacturing dates, and expiration dates from packed products.

- **Focus on Textual Regions:** For this dataset, close-up images of the packaging, specifically focusing on areas with important textual details (batch number, mfg date, exp date), were taken. These regions were manually marked as regions of interest (ROIs).
- **Varied Packaging Conditions:** Since text clarity can vary based on packaging design, fonts, and backgrounds, photos were taken under different lighting conditions and angles. This helped the model handle blurry or low-resolution text that might occur in real-world scenarios.
- **Manual Annotation and ROI Segmentation:** Annotators marked the regions where batch numbers, manufacturing dates, and expiration dates appeared. These annotated regions served as the segmentation targets, allowing the model to focus specifically on extracting and analyzing the textual information.

## 3. Model Training Process

### a) Object Detection Model (YOLOv11)

The object detection model was trained to recognize the 49 classes of packed items using **YOLOv11**, a state-of-the-art object detection architecture known for its speed and accuracy.

- **Training Framework:** The team used YOLOv11 with a pre-trained model and applied transfer learning techniques. YOLOv11 was chosen because of its ability to detect objects in real-time while maintaining high precision, making it suitable for grocery item detection in dynamic environments.
- **Training on Diverse Data:** The mixed image sets with single items, multiple items, different angles, and lighting conditions ensured the model could generalize well. During training, data augmentation techniques such as flipping, rotating, and scaling were applied to artificially expand the dataset and improve the model's robustness.
- **GPU Acceleration:** Given the large dataset and complexity of object detection, the model was trained on NVIDIA RTX Ada 4000 series GPUs, which significantly reduced training time and enabled the handling of larger batches.

### b) Freshness Detection Model

The freshness detection model was trained to classify fruits and vegetables into three categories: good, average, and bad.

- **CNN Architecture:** A convolutional neural network (CNN) was used to extract features from the images and classify them into the appropriate freshness category. Pre-trained models such as ResNet or MobileNet might have been used as the backbone, with custom layers added for classification.
- **Balanced Dataset:** Since freshness is subjective and can vary over time, the dataset was balanced to avoid bias toward any particular class (good, average, bad). The team ensured that there was an equal representation of fruits and vegetables in each category to prevent overfitting.
- **Augmentation:** The dataset was augmented with transformations such as color jittering, brightness adjustment, and rotation to account for natural variations in fruit appearance and ensure the model could detect subtle changes in freshness.

### c) Segmentation Model

The segmentation model was trained to detect and segment the regions containing batch numbers, manufacturing dates, and expiration dates.

- **UNet or Mask R-CNN Architecture:** A model such as UNet or Mask R-CNN might have been used for this task, as these architectures are well-suited for semantic segmentation. The model was trained to accurately detect and isolate text regions on the packaging.
- **Region-based Training:** The annotated regions of interest (ROIs) were used as ground truth, and the model was trained to learn the pixel-wise classification of these areas. The output of the segmentation model was used to guide the OCR system for extracting text information.

## 4. Model Evaluation and Fine-tuning

Once training was complete, the models were evaluated using test sets that included images not seen during training. Metrics such as precision, recall, and F1 score were used to evaluate the object detection and segmentation models, while accuracy and confusion matrices were used for the freshness detection model.

- **Fine-tuning:** If the models exhibited any performance gaps, the team fine-tuned the hyperparameters or retrained specific layers. The models were continuously tested in simulated real-world conditions to ensure they met the required performance benchmarks.

## 5. Real-time Inference and Deployment

After training, the models were deployed in a real-time system where they worked in tandem. The object detection model first identified items, followed by the segmentation model, which extracted and analyzed the text on packaging. For fresh produce, the freshness detection model assigned an index to each fruit, and the results were displayed on the frontend in real-time for operator reference.

- **Backend:** The backend of the system was developed using **FastAPI**, a modern web framework that provided fast and asynchronous API calls, essential for handling real-time inference and high traffic loads.
- **Frontend:** The frontend was built using **HTML, CSS, and JavaScript**, where the real-time results from the models (such as item detection, freshness index, and segmentation details) were displayed for the operator's reference.

This approach ensured that the system was capable of efficiently and accurately assessing grocery quality, handling diverse real-world scenarios with high performance.

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

**gemini_key**

## Installation

clone the repository

```bash
  git clone <git-link-here>
```

create a virtual environment

```bash
python -m venv venv
```

activate the environment

Windows

```bash
venv\Scripts\activate
```

Linux

```bash
. venv/bin/activate
```

Now install requirements

```bash
pip install -r requirements.txt
```

now install torch with CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

```

now running the backend, move to the directory where main.py is located

```bash
uvicorn main:app --reload
```

## Extra Files which needs to be downloaded

- **Models:** https://drive.google.com/file/d/1duIRqbyAFprE5DBZPbeg4mFn-JUhauvM/view?usp=sharing
  unzip the models.zip and paste the models into the folder **/model**
- **Dataset:** https://drive.google.com/drive/folders/10N_75hbKhGKgwnYc2fSAPpB9-_6x4C7I?usp=sharing
