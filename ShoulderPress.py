import cv2
import mediapipe as mp

# Initialize MediaPipe Hands & Face Detection
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    hand_results = hands.process(frame_rgb)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Draw landmarks on hands
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Draw a green dot on the wrist (landmark 0)
            wrist = hand_landmarks.landmark[0]
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
            cv2.circle(frame, (wrist_x, wrist_y), 5, (0, 255, 0), -1)

    # Detect face
    face_results = face_detection.process(frame_rgb)
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            
            # Draw a red rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Hand & Face Tracking", frame)

    # Close window when 'q' is pressed or if manually closed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("Hand & Face Tracking", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
