import cv2
import mediapipe as mp
import time  # Import time module to track delays
import os
from datetime import datetime

# Create directory for saving videos if it doesn't exist
output_dir = "recorded_videos"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

# Get webcam properties for video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video writer setup (will be initialized when recording starts)
video_writer = None

# State variables for the workflow
highest_point_set = False  # First step: set highest point with one finger
started = False  # Second step: start with two fingers
completed = False  # Final step: mark as completed with one finger

# Variables to store wrist positions
top_left_wrist = None
top_right_wrist = None
bottom_left_wrist = None
bottom_right_wrist = None

# Variables to track repetition counts
top_count = 0
bottom_count = 0

# Variables to track if we've recently counted a rep
# This prevents counting the same position multiple times
at_top_position = False
at_bottom_position = False

# Flag to track if top has been reached before counting bottom
top_reached_in_current_rep = False

# Time tracking for delays between rep counting
last_top_time = 0
last_bottom_time = 0
# Minimum delay required between counts (in seconds)
min_delay = 1.5

# Timer variables for tracking set duration
set_start_time = None  # Will store when the set was started
elapsed_time = 0  # Will track elapsed time during the set

# Threshold for how close the hand needs to be to count as a rep (in pixels)
proximity_threshold = 20

# NEW: Timer variables for tracking uneven hands and unbalanced height
uneven_height_timer = 0  # Total accumulated time with uneven height
unbalanced_hands_timer = 0  # Total accumulated time with unbalanced hands
uneven_height_start = None  # Will store when uneven height was first detected
unbalanced_hands_start = None  # Will store when unbalanced hands were first detected
currently_uneven_height = False  # Flag for tracking current state
currently_unbalanced_hands = False  # Flag for tracking current state

# Variable to track when the workout was completed
completion_time = None
# The delay after completion before stopping recording (in seconds)
completion_delay = 1.0

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

    # Get current time for delay calculations
    current_time = time.time()
    
    # Update elapsed time if set is in progress
    if started and not completed and set_start_time is not None:
        elapsed_time = current_time - set_start_time
        
    # Only process detection if not in completed state
    if not completed:
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

                # Draw landmarks on hands only if not completed
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Detect face only if not completed
        face_results = face_detection.process(frame_rgb)
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                face_x = int((bboxC.xmin + bboxC.width / 2) * w)
                face_y = int((bboxC.ymin + bboxC.height / 2) * h)
                
                # Draw face rectangle only if not completed
                x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            

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
        # Initialize the flag to false when starting
        top_reached_in_current_rep = False
        # Initialize the time trackers
        last_top_time = 0
        last_bottom_time = 0
        # Start the set timer
        set_start_time = time.time()
        # Reset uneven timers when starting
        uneven_height_timer = 0
        unbalanced_hands_timer = 0
        uneven_height_start = None
        unbalanced_hands_start = None
        currently_uneven_height = False
        currently_unbalanced_hands = False
        
        # Initialize video writer when set starts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.join(output_dir, f"workout_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
        video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
        print(f"Recording started: {video_filename}")
    
    # Step 3: Mark as completed with one finger on both hands
    if started and not completed and left_hand_one_finger and right_hand_one_finger:
        completed = True
        # Record the time when completed
        completion_time = current_time
        # Freeze the elapsed time at completion
        if set_start_time is not None:
            elapsed_time = current_time - set_start_time
            set_start_time = None  # Clear start time
        
        # If any uneven states are active, add their final durations
        if currently_uneven_height and uneven_height_start is not None:
            uneven_height_timer += (current_time - uneven_height_start)
            currently_uneven_height = False
            uneven_height_start = None
            
        if currently_unbalanced_hands and unbalanced_hands_start is not None:
            unbalanced_hands_timer += (current_time - unbalanced_hands_start)
            currently_unbalanced_hands = False
            unbalanced_hands_start = None
    
    # Count repetitions when activity is in progress and not completed
    if started and not completed and left_hand_x is not None and right_hand_x is not None:
        # Calculate distance to top position
        left_to_top_distance = None
        right_to_top_distance = None
        if top_left_wrist and top_right_wrist:
            left_to_top_distance = ((left_hand_x - top_left_wrist[0])**2 + (left_hand_y - top_left_wrist[1])**2)**0.5
            right_to_top_distance = ((right_hand_x - top_right_wrist[0])**2 + (right_hand_y - top_right_wrist[1])**2)**0.5
        
        # Calculate distance to bottom position
        left_to_bottom_distance = None
        right_to_bottom_distance = None
        if bottom_left_wrist and bottom_right_wrist:
            left_to_bottom_distance = ((left_hand_x - bottom_left_wrist[0])**2 + (left_hand_y - bottom_left_wrist[1])**2)**0.5
            right_to_bottom_distance = ((right_hand_x - bottom_right_wrist[0])**2 + (right_hand_y - bottom_right_wrist[1])**2)**0.5
        
        # Check if hands are close to top position
        if left_to_top_distance is not None and right_to_top_distance is not None:
            if left_to_top_distance < proximity_threshold and right_to_top_distance < proximity_threshold:
                # Check if we've recently counted a top position (time delay check)
                time_since_last_top = current_time - last_top_time
                if not at_top_position and time_since_last_top > min_delay:
                    top_count += 1
                    at_top_position = True
                    top_reached_in_current_rep = True  # Mark that top has been reached in this rep
                    last_top_time = current_time  # Update the last time we counted top
            else:
                at_top_position = False
        
        # Check if hands are close to bottom position, but only count if top has been reached first
        if left_to_bottom_distance is not None and right_to_bottom_distance is not None:
            if left_to_bottom_distance < proximity_threshold and right_to_bottom_distance < proximity_threshold:
                # Check if we've recently counted a bottom position (time delay check)
                time_since_last_bottom = current_time - last_bottom_time
                if not at_bottom_position and top_reached_in_current_rep and time_since_last_bottom > min_delay:
                    bottom_count += 1
                    at_bottom_position = True
                    top_reached_in_current_rep = False  # Reset for next rep
                    last_bottom_time = current_time  # Update the last time we counted bottom
            else:
                at_bottom_position = False
    
    # Calculate remaining cooldown time for visual feedback (if not completed)
    if not completed:
        top_cooldown = max(0, min_delay - (current_time - last_top_time))
        bottom_cooldown = max(0, min_delay - (current_time - last_bottom_time))
        
        # Draw markers for highest and lowest points only if not completed
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

        # Draw lines between hands and face only if not completed
        if left_hand_x is not None and face_x is not None:
            left_distance = abs(left_hand_x - face_x)
            cv2.line(frame, (left_hand_x, left_hand_y), (face_x, face_y), (0, 255, 0), 2)
            mid_x = (left_hand_x + face_x) // 4
            mid_y = (left_hand_y + face_y) // 2
            cv2.putText(frame, f"L: {left_distance}px", (mid_x, mid_y - 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if right_hand_x is not None and face_x is not None:
            right_distance = abs(right_hand_x - face_x)
            cv2.line(frame, (right_hand_x, right_hand_y), (face_x, face_y), (255, 0, 0), 2)
            mid_x = (right_hand_x + face_x) // 2
            mid_y = (right_hand_y + face_y) // 2
            cv2.putText(frame, f"R: {right_distance}px", (mid_x, mid_y - 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Check and update timers for unbalanced hands - ONLY IF SET HAS STARTED
        if started and left_hand_x is not None and face_x is not None and right_hand_x is not None:
            left_distance = abs(left_hand_x - face_x)
            right_distance = abs(right_hand_x - face_x)
            difference = abs(left_distance - right_distance)
            
            # Check if hands are unbalanced
            if difference > 30:
                warning_text = f"UNBALANCED HANDS: {difference}px difference"
                warning_color = (0, 0, 255)  # Red
                
                # If this is the first time we're detecting unbalanced hands, start the timer
                if not currently_unbalanced_hands:
                    unbalanced_hands_start = current_time
                    currently_unbalanced_hands = True
                
                # Calculate current unbalanced duration
                current_unbalanced_duration = current_time - unbalanced_hands_start
                total_unbalanced_time = unbalanced_hands_timer + current_unbalanced_duration
                
                # Add duration to the warning text
                warning_text += f" - Time: {total_unbalanced_time:.1f}s"
                
                # Draw the warning at the bottom of the screen
                text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.putText(frame, warning_text, (w//2 - text_size[0]//2, h - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, warning_color, 2)
            else:
                # If hands are now balanced but were unbalanced before, update the timer
                if currently_unbalanced_hands:
                    unbalanced_hands_timer += (current_time - unbalanced_hands_start)
                    currently_unbalanced_hands = False
                    
                # Display the total time even when balanced
                if unbalanced_hands_timer > 0 or currently_unbalanced_hands:
                    balanced_info = f"Total Unbalanced Hands Time: {unbalanced_hands_timer:.1f}s"
                    cv2.putText(frame, balanced_info, (w//2 - 200, h - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
        # Check and update timers for uneven heights - ONLY IF SET HAS STARTED
        if started and left_hand_y is not None and right_hand_y is not None:
            # Calculate vertical distances from bottom of frame
            left_from_bottom = h - left_hand_y
            right_from_bottom = h - right_hand_y
            vertical_difference = abs(left_from_bottom - right_from_bottom)
            
            if vertical_difference > 30:
                vertical_warning_text = f"UNEVEN HEIGHT: {vertical_difference}px difference"
                vertical_warning_color = (0, 0, 255)  # Red
                
                # If this is the first time we're detecting uneven height, start the timer
                if not currently_uneven_height:
                    uneven_height_start = current_time
                    currently_uneven_height = True
                
                # Calculate current uneven duration
                current_uneven_duration = current_time - uneven_height_start
                total_uneven_time = uneven_height_timer + current_uneven_duration
                
                # Add duration to the warning text
                vertical_warning_text += f" - Time: {total_uneven_time:.1f}s"
                
                # Draw the warning at the bottom of the screen, above the previous warning if present
                text_size = cv2.getTextSize(vertical_warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.putText(frame, vertical_warning_text, (w//2 - text_size[0]//2, h - 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, vertical_warning_color, 2)
            else:
                # If heights are now even but were uneven before, update the timer
                if currently_uneven_height:
                    uneven_height_timer += (current_time - uneven_height_start)
                    currently_uneven_height = False
                
                # Display the total time even when even
                if uneven_height_timer > 0 or currently_uneven_height:
                    even_info = f"Total Uneven Height Time: {uneven_height_timer:.1f}s"
                    cv2.putText(frame, even_info, (w//2 - 200, h - 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Update status text based on the current state
    if completed:
        status_text = "Completed"
        status_color = (0, 255, 0)  # Green
    elif started:
        if top_cooldown > 0:
            status_text = f"TOP Position Cooldown: {top_cooldown:.1f}s"
            status_color = (0, 165, 255)  # Orange
        elif bottom_cooldown > 0:
            status_text = f"BOTTOM Position Cooldown: {bottom_cooldown:.1f}s"
            status_color = (0, 165, 255)  # Orange
        elif top_reached_in_current_rep:
            status_text = "Reach The Bottom or Raise One Finger to Stop"
            status_color = (0, 165, 255)  # Orange
        else:
            status_text = "Reach The Top or Raise One Finger to Stop"
            status_color = (0, 165, 255)  # Orange
    elif highest_point_set:
        status_text = "Raise TWO fingers to Start Your Set"
        status_color = (0, 255, 0)  # Green
    else:
        status_text = "Raise ONE finger to set the highest point"
        status_color = (0, 165, 255)  # Orange
    
    # Create a slightly transparent background behind the text for better visibility
    text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    
    # Display status at the top of the screen
    cv2.putText(frame, status_text, (w//2 - text_size[0]//2, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Display timer below the status text
    if started or completed:
        # Format time as minutes:seconds
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        time_text = f"Time: {minutes:02d}:{seconds:02d}"
        
        # Calculate text width to center it properly
        time_text_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Display the time below the status text (properly centered)
        cv2.putText(frame, time_text, (w//2 - time_text_size[0]//2, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display recording status if currently recording
    if video_writer is not None:
        recording_text = "REC"
        cv2.putText(frame, recording_text, (w - 100, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Red dot for recording

    # Calculate total completed reps (minimum of top and bottom counts)
    # This ensures that a rep is only counted if both top and bottom positions were reached
    total_reps = min(top_count, bottom_count)
    
    # Display counts - always show these regardless of state
    if started or completed:
        # Create background rectangles for better visibility
        top_text = f"TOP: {top_count}"
        top_text_size = cv2.getTextSize(top_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Draw top count
        cv2.putText(
            frame,
            top_text,
            (20, 60),  # Position at top-left corner
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),  # Red
            2,
        )

        # Bottom count background
        bottom_text = f"BOTTOM: {bottom_count}"
        bottom_text_size = cv2.getTextSize(bottom_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Draw bottom count
        cv2.putText(
            frame,
            bottom_text,
            (20, 85),  # Position below top count
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),  # Blue
            2,
        )
        
        # If completed, show total reps and uneven time statistics at the bottom of the screen
        if completed:
            total_text = f"TOTAL COMPLETED REPS: {total_reps}"
            total_text_size = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            
            # Draw total reps text
            cv2.putText(
                frame,
                total_text,
                (w//2 - total_text_size[0]//2, h - 120),  # Centered at bottom
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),  # Black
                2,
            )
            
            # Display uneven timing statistics
            uneven_height_text = f"Total Time with Uneven Height: {uneven_height_timer:.1f} seconds"
            uneven_hands_text = f"Total Time with Unbalanced Hands: {unbalanced_hands_timer:.1f} seconds"
            
            cv2.putText(
                frame,
                uneven_height_text,
                (w//2 - 250, h - 80),  # Position above total reps
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),  # Red
                2,
            )
            
            cv2.putText(
                frame,
                uneven_hands_text,
                (w//2 - 250, h - 40),  # Position above total reps
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),  # Red
                2,
            )

    # Show the frame
    cv2.imshow("Shoulder Press Analyzer", frame)
    
    # Write frame to video if recording is active
    if video_writer is not None:
        video_writer.write(frame)

    # Check if we need to stop the recording after delay
    if completed and completion_time is not None:
        # Calculate time since completion
        time_since_completion = current_time - completion_time
        
        # If we've waited long enough, stop the recording
        if time_since_completion >= completion_delay and video_writer is not None:
            video_writer.release()
            print(f"Recording saved to {video_filename}")
            video_writer = None
            completion_time = None  # Reset to prevent repeated execution
    
    # Close window when 'q' is pressed or if manually closed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("Shoulder Press Analyzer", cv2.WND_PROP_VISIBLE) < 1:
        break

# Make sure to release the video writer if it's still active
if video_writer is not None:
    video_writer.release()

cap.release()
cv2.destroyAllWindows()