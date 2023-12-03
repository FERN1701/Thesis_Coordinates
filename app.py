import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands()
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)
hand_coordinates = []
pose_coordinates = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand points
    hand_results = hands.process(rgb_frame)
    pose_results = pose.process(rgb_frame)

    # If pos are detected, get landmarks and draw them
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # Extract the coordinates of each pose landmark
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cz = int(landmark.z * 100)  # Scale for better visibility
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                # for 3d coordination of the pose
                cv2.putText(frame, f"Hand {idx}: ({cx}, {cy}, {cz})", (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                # append
                hand_coordinates.append((cx, cy, cz))

    # drow pose connection from dotted
    if pose_results.pose_landmarks:
        # Draw lines between shoulder, elbow, and wrist
        mp.solutions.drawing_utils.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract XYZ coordinates for specific pose landmarks (e.g., shoulder, elbow, and wrist)
        shoulder_xyz = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x, pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y, pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z
        elbow_xyz = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x, pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y, pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].z
        wrist_xyz = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x, pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y, pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].z

        # Display the XYZ cordinates for specfic pose landmarks
        cv2.putText(frame, f"Shoulder: ({shoulder_xyz[0]:.2f}, {shoulder_xyz[1]:.2f}, {shoulder_xyz[2]*100:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Elbow: ({elbow_xyz[0]:.2f}, {elbow_xyz[1]:.2f}, {elbow_xyz[2]*100:.2f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Wrist: ({wrist_xyz[0]:.2f}, {wrist_xyz[1]:.2f}, {wrist_xyz[2]*100:.2f})", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Append coordinates to the array
        pose_coordinates.append({
            'Shoulder': shoulder_xyz,
            'Elbow': elbow_xyz,
            'Wrist': wrist_xyz
        })

    cv2.imshow("Hand and Pose Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()


print("\n\n\n Extracted Coordinates : \n\n\n")

print(pose_coordinates)