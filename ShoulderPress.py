import cv2
import mediapipe as mp

# Initialize MediaPipe Hands & Face Detection
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# intialize hands and face detection and pose which will be used to track forearms
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# State variables for the workflow
highest_point_set = False  # First step: set highest point with one finger
started = False  # Second step: start with two fingers
completed = False  # Final step: mark as completed with one finger

# Variables to store wrist positions
top_left_wrist = None
top_right_wrist = None
bottom_left_wrist = None
bottom_right_wrist = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Mirror the frame horizontally
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    left_hand_x = None   # Store left-hand X position
    left_hand_y = None   # Store left-hand Y position
    right_hand_x = None  # Store right-hand X position
    right_hand_y = None  # Store right-hand Y position
    face_x = None        # Store face X position
    face_y = None        # Store face Y position
    
    # Reset finger gesture detection
    left_hand_one_finger = False
    right_hand_one_finger = False
    left_hand_two_fingers = False
    right_hand_two_fingers = False

    # Detect hands
    hand_results = hands.process(frame_rgb)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            label = handedness.classification[0].label  # "Left" or "Right"
            
            # Get wrist position (landmark 0)
            wrist = hand_landmarks.landmark[0]
            
            # Get finger positions
            index_tip = hand_landmarks.landmark[8].y
            middle_tip = hand_landmarks.landmark[12].y
            ring_tip = hand_landmarks.landmark[16].y
            pinky_tip = hand_landmarks.landmark[20].y
            
            # Check finger states
            index_up = index_tip < hand_landmarks.landmark[6].y
            middle_up = middle_tip < hand_landmarks.landmark[10].y
            ring_down = ring_tip > hand_landmarks.landmark[14].y
            pinky_down = pinky_tip > hand_landmarks.landmark[18].y
            middle_down = middle_tip > hand_landmarks.landmark[10].y
            
            # Define gestures
            one_finger = index_up and middle_down and ring_down and pinky_down
            two_fingers = index_up and middle_up and ring_down and pinky_down
            
            if label == "Left":
                left_hand_x = int(wrist.x * w)
                left_hand_y = int(wrist.y * h)
                left_hand_one_finger = one_finger
                left_hand_two_fingers = two_fingers
                
            elif label == "Right":
                right_hand_x = int(wrist.x * w)
                right_hand_y = int(wrist.y * h)
                right_hand_one_finger = one_finger
                right_hand_two_fingers = two_fingers

            # Draw landmarks on hands
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Handle state transitions based on gestures
    
    # Step 1: Set highest point with one finger on both hands
    if not highest_point_set and left_hand_one_finger and right_hand_one_finger:
        highest_point_set = True
        top_left_wrist = (left_hand_x, left_hand_y)
        top_right_wrist = (right_hand_x, right_hand_y)
    
    # Step 2: Start the activity with two fingers on both hands
    if highest_point_set and not started and left_hand_two_fingers and right_hand_two_fingers:
        started = True
        bottom_left_wrist = (left_hand_x, left_hand_y)
        bottom_right_wrist = (right_hand_x, right_hand_y)
    
    # Step 3: Mark as completed with one finger on both hands
    if started and not completed and left_hand_one_finger and right_hand_one_finger:
        completed = True
    
    # Draw markers for highest and lowest points
    if top_left_wrist:
        cv2.circle(frame, top_left_wrist, 10, (0, 0, 255), -1)  # Red dot
        cv2.putText(frame, "Highest Point", (top_left_wrist[0] - 40, top_left_wrist[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if top_right_wrist:
        cv2.circle(frame, top_right_wrist, 10, (0, 0, 255), -1)  # Red dot
        cv2.putText(frame, "Highest Point", (top_right_wrist[0] - 40, top_right_wrist[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    if bottom_left_wrist:
        cv2.circle(frame, bottom_left_wrist, 10, (255, 0, 0), -1)  # Blue dot
        cv2.putText(frame, "Lowest Point", (bottom_left_wrist[0] - 40, bottom_left_wrist[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if bottom_right_wrist:
        cv2.circle(frame, bottom_right_wrist, 10, (255, 0, 0), -1)  # Blue dot
        cv2.putText(frame, "Lowest Point", (bottom_right_wrist[0] - 40, bottom_right_wrist[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Update status text based on the current state
    if completed:
        status_text = "Completed"
        status_color = (0, 255, 0)  # Green
    elif started:
        status_text = "Activity in progress - Raise ONE finger when done"
        status_color = (0, 165, 255)  # Orange
    elif highest_point_set:
        status_text = "Raise TWO fingers to start"
        status_color = (0, 255, 0)  # Green
    else:
        status_text = "Raise ONE finger to set the highest point"
        status_color = (0, 165, 255)  # Orange
        
    # Display status at the top of the screen
    cv2.putText(frame, status_text, (w//2 - 250, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Detect face
    face_results = face_detection.process(frame_rgb)
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            face_x = int((bboxC.xmin + bboxC.width / 2) * w)
            face_y = int((bboxC.ymin + bboxC.height / 2) * h)
            
            # Draw face rectangle
            x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)

    # Draw lines between hands and face
    if left_hand_x is not None and face_x is not None:
        x_distance = abs(left_hand_x - face_x)
        cv2.line(frame, (left_hand_x, left_hand_y), (face_x, face_y), (0, 255, 0), 2)
        mid_x = (left_hand_x + face_x) // 2
        mid_y = (left_hand_y + face_y) // 2
        cv2.putText(frame, f"L: {x_distance}px", (mid_x, mid_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if right_hand_x is not None and face_x is not None:
        x_distance = abs(right_hand_x - face_x)
        cv2.line(frame, (right_hand_x, right_hand_y), (face_x, face_y), (255, 0, 0), 2)
        mid_x = (right_hand_x + face_x) // 2
        mid_y = (right_hand_y + face_y) // 2
        cv2.putText(frame, f"R: {x_distance}px", (mid_x, mid_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # Show the frame
    cv2.imshow("Hand & Face Tracking with X-Distance", frame)

    # Close window when 'q' is pressed or if manually closed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("Hand & Face Tracking with X-Distance", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()