import face_recognition
import cv2
import numpy as np

# This is a super simple (but slow) example of running face recognition on live video from your webcam.
# There's a second example that's a little more complicated but runs faster.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Load a second sample picture and learn how to recognize it.
amit_image = face_recognition.load_image_file("amit.png")
amit_face_encoding = face_recognition.face_encodings(amit_image)[0]

# Load a second sample picture and learn how to recognize it.
aarna_image = face_recognition.load_image_file("aarna.jpg")
aarna_face_encoding = face_recognition.face_encodings(aarna_image)[0]

# Load a second sample picture and learn how to recognize it.
shikha_image = face_recognition.load_image_file("shikha.jpg")
shikha_face_encoding = face_recognition.face_encodings(shikha_image)[0]

# Load a second sample picture and learn how to recognize it.
albert_image = face_recognition.load_image_file("albert.jpg")
albert_face_encoding = face_recognition.face_encodings(albert_image)[0]

# Load a second sample picture and learn how to recognize it.
gautam_image = face_recognition.load_image_file("gautam.jpg")
gautam_face_encoding = face_recognition.face_encodings(gautam_image)[0]

# Load a second sample picture and learn how to recognize it.
kristina_image = face_recognition.load_image_file("kristina.jpg")
kristina_face_encoding = face_recognition.face_encodings(kristina_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    amit_face_encoding,
    aarna_face_encoding,
    shikha_face_encoding,
    albert_face_encoding,
    gautam_face_encoding,
    kristina_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "Amit G",
    "Superhero Aarna",
    "Shikha G",
    "Albert",
    "Gautam",
    "Kristina"
]
import glob
import os
def filebrowser(indir,ext=""):
    "Returns files with an extension"
    return [f for f in glob.glob(indir+'/*'+ext)]


def get_face_encodings(indir=r'C:\Work\per\dview\face_recognition\faces_db'):
    print('Reading known faces from:',indir)
    #This function will return the face encoding and names to allow matching
    #it loads faces from a database of facedir where all images of database are kept
    # Load a sample picture and learn how to recognize it.
    #let's get all images
    f_jpg = filebrowser(indir, 'jpg')
    f_jpeg = filebrowser(indir, 'jpeg')
    f_png = filebrowser(indir, 'png')

    f_all = f_jpg + f_jpeg + f_png

    known_face_encodings = []
    known_face_names = []

    for f in f_all:
        print(f)
        image_data = face_recognition.load_image_file(f)
        image_encoding = face_recognition.face_encodings(image_data)[0]
        f_name=os.path.basename(f).split('.')[0]+'-'+os.path.basename(f).split('.')[1]
        known_face_encodings.append(image_encoding)
        known_face_names.append(f_name)
    # obama_image = face_recognition.load_image_file("obama.jpg")
    # obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
    #
    # # Load a second sample picture and learn how to recognize it.
    # biden_image = face_recognition.load_image_file("biden.jpg")
    # biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
    #
    # # Load a second sample picture and learn how to recognize it.
    # amit_image = face_recognition.load_image_file("amit.png")
    # amit_face_encoding = face_recognition.face_encodings(amit_image)[0]
    #
    # # Load a second sample picture and learn how to recognize it.
    # aarna_image = face_recognition.load_image_file("aarna.jpg")
    # aarna_face_encoding = face_recognition.face_encodings(aarna_image)[0]
    #
    # # Load a second sample picture and learn how to recognize it.
    # shikha_image = face_recognition.load_image_file("shikha.jpg")
    # shikha_face_encoding = face_recognition.face_encodings(shikha_image)[0]
    #
    # # Load a second sample picture and learn how to recognize it.
    # albert_image = face_recognition.load_image_file("albert.jpg")
    # albert_face_encoding = face_recognition.face_encodings(albert_image)[0]
    #
    # # Load a second sample picture and learn how to recognize it.
    # gautam_image = face_recognition.load_image_file("gautam.jpg")
    # gautam_face_encoding = face_recognition.face_encodings(gautam_image)[0]
    #
    # # Load a second sample picture and learn how to recognize it.
    # kristina_image = face_recognition.load_image_file("kristina.jpg")
    # kristina_face_encoding = face_recognition.face_encodings(kristina_image)[0]
    #
    # # Create arrays of known face encodings and their names
    # known_face_encodings = [
    #     obama_face_encoding,
    #     biden_face_encoding,
    #     amit_face_encoding,
    #     aarna_face_encoding,
    #     shikha_face_encoding,
    #     albert_face_encoding,
    #     gautam_face_encoding,
    #     kristina_face_encoding
    # ]
    # known_face_names = [
    #     "Barack Obama",
    #     "Joe Biden",
    #     "Amit G",
    #     "Superhero Aarna",
    #     "Shikha G",
    #     "Albert",
    #     "Gautam",
    #     "Kristina"
    # ]
    return known_face_names, known_face_encodings

def match_faces(frame,face_locations,face_encodings,kfe,kfn,show_matches=True):
    known_face_encodings = kfe
    known_face_names =kfn

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            print('Found a match:',name)
            frame_match = frame[top:bottom,left:right,:]
            if (show_matches):
                cv2.imshow(name, frame_match)
                # Hit 'q' on the keyboard to quit!
                cv2.waitKey(0)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        return frame

kfn,kfe = get_face_encodings()


known_face_names = kfn
known_face_encodings = kfe

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
