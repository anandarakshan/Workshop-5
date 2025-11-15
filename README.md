# Workshop-5
## Name: Ananda Rakshan K V
## Reg.No: 212223230014
# AIM:
To implement a License Plate Detection system using OpenCV’s Haar Cascade Classifier.
# Algorithm:
1.Read the input image containing the vehicle using OpenCV.

2.Convert the image to grayscale to simplify processing.

3.Load the Haar Cascade classifier for license plate detection.

4.Apply the classifier using detectMultiScale() to locate plate regions.

5.Draw bounding boxes around the detected license plates.

6.Crop and save the detected plate area as a separate image for further use.

# Program:
```
import cv2
import matplotlib.pyplot as plt
import numpy as np
# Read input image
img = cv2.imread('car.jpg')

# Convert BGR → RGB for matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title('Input Image')
plt.axis('off')
plt.show()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()
# Load pre-trained Haar Cascade
plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

# Detect license plates
plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

print(f"Detected {len(plates)} plates.")
for (x, y, w, h) in plates:
    cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)

plt.imshow(img_rgb)
plt.title('Detected License Plates')
plt.axis('off')
plt.show()
for i, (x, y, w, h) in enumerate(plates):
    plate_roi = gray[y:y+h, x:x+w]
    cv2.imwrite(f"plate_{i+1}.png", plate_roi)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
equalized = cv2.equalizeHist(blurred)

plt.imshow(equalized, cmap='gray')
plt.title('Preprocessed Image')
plt.axis('off')
plt.show()

plates_improved = plate_cascade.detectMultiScale(equalized, scaleFactor=1.1, minNeighbors=4)
print(f"Plates detected after preprocessing: {len(plates_improved)}")
fig, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].imshow(img_rgb)
axs[0].set_title('Original Detection')
axs[1].imshow(equalized, cmap='gray')
axs[1].set_title('Improved Detection (Preprocessed)')
for ax in axs: ax.axis('off')
plt.show()
```

# Output:
<img width="645" height="512" alt="image" src="https://github.com/user-attachments/assets/aad3bf85-cbf3-4cb3-b1fe-d81ebd2771fc" />

<img width="662" height="500" alt="image" src="https://github.com/user-attachments/assets/23778b1e-a7ea-44c6-8e2e-4d4c6a2ba620" />

<img width="651" height="511" alt="image" src="https://github.com/user-attachments/assets/4010502c-d225-4147-ae6a-2435729ac096" />

<img width="652" height="512" alt="image" src="https://github.com/user-attachments/assets/7bd0d757-52b4-4e16-bef8-6f7286ca4ee9" />

<img width="819" height="321" alt="image" src="https://github.com/user-attachments/assets/d3d72773-6e89-4286-ab84-afc0c5824597" />

# Result:
The Haar Cascade classifier successfully detected the license plate region from the input image. After preprocessing (Gaussian Blur and Histogram Equalization), the detection became more stable and accurate.
