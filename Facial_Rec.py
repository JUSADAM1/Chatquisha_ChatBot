#imported opencv cause they have the facial REC feature.
import cv2
import os
import sys


def face_rec():
    cap = cv2.VideoCapture(0)

    # Create the haar cascade
    #######################################################################################################
    # This came from Github!!!
    ########################################################################################################
    #This file has data based on the facial rec stuff.
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    prev_num = 0
    img_counter = 0

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        num_faces = len(faces)
        print("Found {0} faces!".format(num_faces))

        if num_faces != prev_num:
            print("New Face!")

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


        prev_num = num_faces
        # Display the resulting frame
        cv2.imshow('frame', frame)
        # Display quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 256 == 32:
            img_name = "facedetect_webcam_{}.png".format(img_counter)
            cv2.imshow(img_name, frame)
            print("{} written!!".format(img_name))
            img_counter += 1


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
# Tiding it together with the others
if __name__ == '__main__':
     face_rec()
