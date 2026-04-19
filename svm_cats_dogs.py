import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ======================
# DATASET PATH
# ======================
DATADIR = r"train/train"   # ⚠️ your structure

IMG_SIZE = 64
data = []

print("Loading images...")

files = os.listdir(DATADIR)

cats = []
dogs = []

# Separate classes
for img in files:
    if img.lower().startswith("cat"):
        cats.append(img)
    elif img.lower().startswith("dog"):
        dogs.append(img)

# Balanced dataset
cats = cats[:1000]
dogs = dogs[:1000]

print("Total cats:", len(cats))
print("Total dogs:", len(dogs))

selected_files = cats + dogs
random.shuffle(selected_files)

# ======================
# LOAD COLOR IMAGES
# ======================
for img in selected_files:
    try:
        img_path = os.path.join(DATADIR, img)

        label = 0 if img.lower().startswith("cat") else 1

        img_array = cv2.imread(img_path)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        data.append([resized, label])

    except:
        continue

print("Total images loaded:", len(data))

# ======================
# PREPARE DATA
# ======================
random.shuffle(data)

X = []
y = []

for features, label in data:
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Save images for display
X_images = X.copy()

# Flatten for SVM
X = X.reshape(-1, IMG_SIZE * IMG_SIZE * 3)
X = X / 255.0

# Split
X_train, X_test, y_train, y_test, X_train_img, X_test_img = train_test_split(
    X, y, X_images, test_size=0.2, random_state=42
)

# ======================
# TRAIN MODEL
# ======================
print("\nTraining SVM model...")
model = SVC(kernel='rbf')
model.fit(X_train, y_train)

print("Model trained!")

# ======================
# PREDICTION
# ======================
y_pred = model.predict(X_test)

# ======================
# RESULTS
# ======================
print("\n===== RESULTS =====")
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ======================
# 🖼️ DISPLAY IMAGES (3 SCREENS)
# ======================
images_per_screen = 12
total_screens = 3

for screen in range(total_screens):
    plt.figure(figsize=(12, 8))

    start = screen * images_per_screen
    end = start + images_per_screen

    for i in range(start, end):
        if i >= len(X_test_img):
            break

        plt.subplot(3, 4, i - start + 1)

        plt.imshow(X_test_img[i])

        actual = "Cat" if y_test[i] == 0 else "Dog"
        predicted = "Cat" if y_pred[i] == 0 else "Dog"

        color = "green" if actual == predicted else "red"

        plt.title(f"A:{actual}\nP:{predicted}", fontsize=8, color=color)
        plt.axis('off')

    plt.suptitle(f"Predictions - Screen {screen+1}", fontsize=14)
    plt.tight_layout()
    plt.show()

# ======================
# 📊 CONFUSION MATRIX
# ======================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Cat", "Dog"],
            yticklabels=["Cat", "Dog"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ======================
# 📈 ACCURACY GRAPH
# ======================
plt.figure()
plt.bar(["SVM Model"], [accuracy])
plt.ylim(0, 1)
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.show()