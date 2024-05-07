import os
import cv2
import math
import numpy as np
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances as L2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array


database_path = 'D:\\bishe/input/custom_database'
querybase_path = 'D:\\bishe/input/ds-kimds'

embeddings = tf.keras.models.load_model("D:\\bishe/input/facenet/keras-facenet/model/facenet_keras.h5")
embeddings.load_weights('D:\\bishe/input/facenet/keras-facenet/weights/facenet_keras_weights.h5')

##
img_size = 160

for layer in embeddings.layers:
    if layer.name == 'Block8_3_ScaleSum':
        break
    layer.trainable = False

image_input1 = Input(shape=(img_size, img_size, 3), name='Image1')
image_input2 = Input(shape=(img_size, img_size, 3), name='Image2')
image_input3 = Input(shape=(img_size, img_size, 3), name='Image3')

anchor = embeddings(image_input1)
positive = embeddings(image_input2)
negative = embeddings(image_input3)

siamese_network = Model(inputs=[image_input1, image_input2, image_input3], outputs=[anchor, positive, negative])

siamese_network.load_weights("model.h5")

face_embeddings = siamese_network.layers[-1]
##

detector = MTCNN()


def euclidean(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def align(x1, y1, x2, y2, img):
    x3, y3 = 0, 0
    adj, hyp = 0, euclidean(x1, y1, x2, y2)
    angle = 0

    if y1 > y2:
        x3, y3 = x1, y2
        adj = euclidean(x3, y3, x2, y2)
        angle = -math.degrees(math.acos(adj / hyp))
    else:
        x3, y3 = x2, y1
        adj = euclidean(x3, y3, x1, y1)
        angle = math.degrees(math.acos(adj / hyp))

    M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, 1)
    out = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return out


def read_image(directory, image_path):
    img = load_img(os.path.join(directory, image_path))
    img = img_to_array(img)
    faces = detector.detect_faces(img)
    x1, y1 = faces[0]['keypoints']['left_eye']
    x2, y2 = faces[0]['keypoints']['right_eye']
    img = align(x1, y1, x2, y2, img)
    faces = detector.detect_faces(img)
    x, y, w, h = faces[0]['box']
    img = img[y:y + h, x:x + w, :]
    img /= 255.
    img = cv2.resize(img, (160, 160))
    return img


database_images = os.listdir(database_path)
querybase_images = os.listdir(querybase_path)

db_images = sorted(database_images)
query_images = sorted(querybase_images)

db_names = map(lambda x: x.split('.')[0], db_images)
db_names = list(db_names)

detected_faces = []
directory = database_path
query_directory = querybase_path
for image in db_images:
    img = read_image(directory, image)
    detected_faces.append(img)

detected_faces = np.array(detected_faces)
db_embeddings = face_embeddings.predict(detected_faces)


##
def query_face(directory, filename, db_embeddings, db_names, thresh):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
    color = (255, 255, 255)
    thickness = 1

    # 1
    query_img = read_image(directory, filename)
    query_img = np.expand_dims(query_img, axis=0)
    # 2
    query_embedding = face_embeddings.predict(query_img)
    # 3
    distances = L2(db_embeddings, query_embedding)
    # 4
    index = np.argmin(distances, axis=0)
    q_index = index[0] if distances[index] < thresh else None
    label = 'Unknown'
    try:
        label = db_names[q_index]
        label = label.split('_')[0] + "_" + label.split('_')[1]
    except TypeError:
        pass

    # 5
    (width, height), b = cv2.getTextSize(label, font, fontScale, thickness)

    img = cv2.imread(os.path.join(directory, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img)
    x, y, w, h = faces[0]['box']

    img = cv2.rectangle(img, (x, y), (x + w, y + h), (51, 153, 255), 5)
    img = cv2.rectangle(img, (x, y - height), (x + width, y), (51, 153, 255), -1)
    img = cv2.putText(img, label, (x, y), font,
                      fontScale, color, thickness, cv2.LINE_AA)

    return img


plt.figure(figsize=(60, 60))
n = 0
for image in query_images[0:10]:
    n += 1
    plt.subplot(2, 5, n)
    img = query_face(query_directory, image, db_embeddings, db_names, 1.65)
    plt.imshow(img)
    plt.axis("off")
plt.show()
