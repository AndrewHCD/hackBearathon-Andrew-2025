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

# Variable to store the status of "two fingers raised"
two_fingers_detected = False
two_finger_done = False

# varible to store the status of one finger raised
one_fingers_detected = False
one_finger_done = False

# Variables to store wrist positions when two fingers are detected at the bottom
bottom_left_wrist = None
bottom_right_wrist = None

# Variables to store wrist positions for highest point
top_left_wrist = None # testing
top_right_wrist = None # testing
activate_highest_point = False


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
    
    # sets both one finger and two finger to false at the start
    left_hand_two_fingers = False
    right_hand_two_fingers = False
    left_hand_one_fingers = False
    right_hand_one_fingers = False

    # Detect hands
    hand_results = hands.process(frame_rgb)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            label = handedness.classification[0].label  # "Left" or "Right"
            
            # Get wrist position (landmark 0)
            wrist = hand_landmarks.landmark[0]
            
            # Check for two fingers up (index and middle fingers)
            # Index finger tip (landmark 8) and middle finger tip (landmark 12)
            index_tip = hand_landmarks.landmark[8].y
            middle_tip = hand_landmarks.landmark[12].y
            ring_tip = hand_landmarks.landmark[16].y
            pinky_tip = hand_landmarks.landmark[20].y
            
            # Check if index and middle fingers are up (y position lower than other fingers)
            index_up = index_tip < hand_landmarks.landmark[6].y  # Index finger up
            middle_up = middle_tip < hand_landmarks.landmark[10].y  # Middle finger up
    
            ring_down = ring_tip > hand_landmarks.landmark[14].y  # Ring finger down
            pinky_down = pinky_tip > hand_landmarks.landmark[18].y  # Pinky finger down
            
            # used for 1 finger up
            middle_down = middle_tip > hand_landmarks.landmark[10].y  # Middle finger down
            
            # if two fingers are up these are the conditions
            two_fingers_up = index_up and middle_up and ring_down and pinky_down
            # if 1 finger is up these are the conditions 
            one_finger_up = index_up and middle_down and ring_down and pinky_down
            
            if label == "Left":
                left_hand_x = int(wrist.x * w)  # Convert to pixel coordinates
                left_hand_y = int(wrist.y * h)  # Get Y coordinate too
                
                 
                # if two fingers are raised changes it two_fingers_up
                if two_finger_done == False:
                    left_hand_two_fingers = two_fingers_up
                    two_fingers_up = True
                        
                if two_finger_done == True:
                    left_hand_one_fingers = one_finger_up
                    
            elif label == "Right":
                right_hand_x = int(wrist.x * w)  # Convert to pixel coordinates
                right_hand_y = int(wrist.y * h)  # Get Y coordinate too
                
                
                # if two fingers are raised changes it two_fingers_up
                if two_finger_done == False:
                    right_hand_two_fingers = two_fingers_up
                    two_fingers_up = True
    
                if two_finger_done == True:
                    right_hand_one_fingers = one_finger_up

            # Draw landmarks on hands
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Update the "two_fingers_detected" status only if both hands have two fingers up
    # Track the state of finger detection
    if left_hand_two_fingers and right_hand_two_fingers and not two_finger_done:
    
        if activate_highest_point == False:
            top_right_wrist = (right_hand_x, right_hand_y)  # testing
            top_left_wrist = (left_hand_x, left_hand_y) # testing
            
        if activate_highest_point == True:   
            two_fingers_detected = True
            two_finger_done = True  # Remember that two fingers were detected
            activate_highest_point = True
            right_hand_two_fingers = False
            left_hand_two_fingers = False
            # Store wrist positions
            bottom_left_wrist = (left_hand_x, left_hand_y)
            bottom_right_wrist = (right_hand_x, right_hand_y)
            
        
    # Draw blue dots at stored wrist positions this will be used to record bottom of the rep compared to top
    if bottom_left_wrist:
        cv2.circle(frame, bottom_left_wrist, 10, (255, 0, 0), -1)  # Blue dot
        cv2.putText(frame, "Lowest Point", (bottom_left_wrist[0] - 40, bottom_left_wrist[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Text below dot

    if bottom_right_wrist:
        cv2.circle(frame, bottom_right_wrist, 10, (255, 0, 0), -1)  # Blue dot
        cv2.putText(frame, "Lowest Point", (bottom_right_wrist[0] - 40, bottom_right_wrist[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Text below do
        
    # Draw red dots at the highest wrist positions
    if top_left_wrist: # testing
        cv2.circle(frame, top_left_wrist, 10, (0, 0, 255), -1)  # Red dot for highest point  # testing
        cv2.putText(frame, "Highest Point", (top_left_wrist[0] - 40, top_left_wrist[1] - 20),  # testing
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Text above dot  # testing

    if top_right_wrist:
        cv2.circle(frame, top_right_wrist, 10, (0, 0, 255), -1)  # Red dot for highest point  # testing
        cv2.putText(frame, "Highest Point", (top_right_wrist[0] - 40, top_right_wrist[1] - 20),  # testing
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Text above dot  # testing

    if two_finger_done and left_hand_one_fingers and right_hand_one_fingers:
        one_fingers_detected = True
        one_finger_done = True  # Remember that one finger was detected
        two_finger_done = False  # Reset two fingers state

    # Update status message based on detection
    if one_finger_done:
        status_text = "Completed"
        status_color = (0, 255, 0)  # Green
    elif two_fingers_detected:
        status_text = "Two fingers detected on both hands!"
        status_color = (0, 255, 0)  # Green
    else:
        status_text = "Raise TWO fingers on BOTH hands"
        status_color = (0, 165, 255)  # Orange

    # Display instruction/status at the top of the screen
    cv2.putText(frame, status_text, (w//2 - 200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Detect face
    face_results = face_detection.process(frame_rgb)
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            face_x = int((bboxC.xmin + bboxC.width / 2) * w)  # Get center X of the face
            face_y = int((bboxC.ymin + bboxC.height / 2) * h)  # Get center Y of the face
            
            # Draw face rectangle
            x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)

    # Draw a line and display distance for left hand if detected
    if left_hand_x is not None and face_x is not None:
        x_distance = abs(left_hand_x - face_x)
        
        # Draw a line between left hand and face
        cv2.line(frame, (left_hand_x, left_hand_y), (face_x, face_y), (0, 255, 0), 2)
        
        # Calculate the midpoint to place the text above the line
        mid_x = (left_hand_x + face_x) // 2
        mid_y = (left_hand_y + face_y) // 2
        
        # Display distance above the line
        cv2.putText(frame, f"L: {x_distance}px", (mid_x, mid_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Draw a line and display distance for right hand if detected
    if right_hand_x is not None and face_x is not None:
        x_distance = abs(right_hand_x - face_x)
        
        # Draw a line between right hand and face
        cv2.line(frame, (right_hand_x, right_hand_y), (face_x, face_y), (255, 0, 0), 2)
        
        # Calculate the midpoint to place the text above the line
        mid_x = (right_hand_x + face_x) // 2
        mid_y = (right_hand_y + face_y) // 2
        
        # Display distance above the line
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
