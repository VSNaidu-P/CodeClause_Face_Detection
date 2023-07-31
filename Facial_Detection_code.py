import cv2
import dlib

# Load the pre-trained DLIB face detector
detector = dlib.get_frontal_face_detector()

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    
    # Convert the frame to grayscale (DLIB requires grayscale input)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform facial detection using the DLIB model
    faces = detector(gray_frame)
    
    for face in faces:
        # Get the bounding box coordinates of the detected face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Real-time Facial Detection', frame)
    
    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
