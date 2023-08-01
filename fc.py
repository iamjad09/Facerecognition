# Import necessary libraries
import cv2
import datetime
import numpy as np
import csv
import os
from tensorflow.keras.models import load_model
import face_recognition

# Load the face recognition model from a pre-trained file
model = load_model('ThestrongestTeam.h5')

# Initialize empty lists to store face encodings and corresponding names
faces = []
names = []

# Loop through each image in the 'faces' directory, resize it, and encode it using the loaded model
faces_dir = os.path.join('faces')
for i in os.listdir(faces_dir):
    # Check if the file extension is valid
    split_name = os.path.splitext(i)
    if split_name[1] not in ['.jpeg', '.jpg', '.png']:
        print(f"Skipping invalid file: {i}")
        continue
    # Load and resize the image to match the input size of the loaded model
    image = cv2.imread(os.path.join(faces_dir, i))
    if image is None:
        print(f"Failed to load image: {i}")
        continue
    if image.shape[:2] != (224, 224):
        image = cv2.resize(image, (224, 224))
    # Preprocess the image by normalizing pixel values and expanding its dimensions
    face_img = np.expand_dims(image, axis=0)
    face_img = face_img / 255.0
    # Encode the face image using the loaded model and append the result to the lists
    face_encoding = model.predict(face_img)
    faces.append(face_encoding)
    names.append(split_name[0])

# Create a new CSV file to record attendance
with open('attendance.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Time', 'Image File'])

# Initialize an empty list to keep track of recorded names
recorded_names = []

# Define the project directory and open the video capture device
project_dir = os.getcwd()
captures_dir = os.path.join(project_dir, 'captures')
cap = cv2.VideoCapture(0)

# Set the frame size and encoding of the video capture device
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

# Check if the video capture device was successfully opened
if not cap.isOpened():
    print("Failed to open video capture")
    exit(1)

# Start the main loop to capture video frames and recognize faces
while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from video capture")
        break

    # Convert the BGR color format to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings in the RGB frame using the face_recognition library
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each detected face and recognize it based on the stored encodings
    for face_location in face_locations:
        # Extract the coordinates of the face bounding box
        top, right, bottom, left = face_location
        # Extract the face image from the frame and resize it to match the input size of the loaded model
        face_img = frame[top:bottom, left:right]
        face_img = cv2.resize(face_img, (224, 224))
        # Preprocess the face image by normalizing pixel values and expanding its dimensions
        face_img = np.expand_dims(face_img, axis=0)
        face_img = face_img / 255.0
        # Encode the face image using the loaded model
        face_encoding = model.predict(face_img)
        # Compute the Euclidean distance between the encoded face and each stored encoding
        matches = []
        for known_face_encoding in faces:
            distance = np.linalg.norm(face_encoding - known_face_encoding)
            # If the distance is below a certain threshold, the face is considered a match
            if distance < 0.6:
                matches.append(True)
            else:
                matches.append(False)
        # Set the name of the recognized face to "Unknown" by default
        name = "Unknown"
        # If there is a match, record the attendance and save a capture of the face image
        if True in matches:
            index = matches.index(True)
            name = names[index]
            if name not in recorded_names:
                # Generate a unique filename based on the current date and time
                filename = f"{name}{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                # Save the capture of theface image to the 'captures' directory
                cv2.imwrite(os.path.join(captures_dir, filename), frame)
                # Record the attendance in the CSV file
                with open('attendance.csv', 'a', newline='') as file:
                    writer= csv.writer(file)
                    writer.writerow([name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), filename])
                # Add the name to the list of recorded names to avoid duplicates
                recorded_names.append(name)

        # Draw a rectangle around the face bounding box and display the recognized name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Show the video frame with recognized faces
    cv2.imshow('Attendance System', frame)

    # Wait for a key press and check if the 'q' or 's' key was pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save a capture of the current frame with a unique filename based on the current date and time
        filename = f"capture{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        cv2.imwrite(os.path.join(captures_dir, filename), frame)
        print(f"Saved capture to {filename}")

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()