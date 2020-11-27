import cv2 as cv
import os
from PIL import Image


class Face:
    image_path = ""
    age = None
    gender = None
    is_wearing_mask = None

    def __init__(self, image_path):
        self.image_path = image_path


"""
Returns a list of faces found from an image, given a filepath to the image,
returns an empty list in the cast of no faces detected
"""
def find_faces(image_path):
    casc_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv.CascadeClassifier(casc_path)
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    face_coords = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    # face_coords has x,y,w,h
    # need to extract face from x,y to x+w,y+h for each face and create an instance of Face
    im = Image.open(image_path)
    faces = list()
    filename = os.path.splitext(image_path)[0]
    file_extension = os.path.splitext(image_path)[1]
    i = 1
    for (x, y, w, h) in face_coords:
        image_of_face = im.crop((x, y, x+w, y+h))  # image of face extracted from image
        image_of_face_path = os.path.join(filename + '_face_' + str(i) + file_extension)
        i += 1
        image_of_face.save(image_of_face_path)
        faces.append(Face(image_of_face_path))

    for face in faces:
        print(face.image_path)

    return faces
