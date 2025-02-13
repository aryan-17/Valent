import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to check if hands form a heart shape
def is_heart_gesture(landmarks):
    # Extract key points (thumbs and index fingers)
    thumb_tip_1 = landmarks[4]  # Thumb tip of first hand
    index_tip_1 = landmarks[8]  # Index finger tip of first hand
    thumb_tip_2 = landmarks[20]  # Thumb tip of second hand
    index_tip_2 = landmarks[12]  # Index finger tip of second hand

    # Calculate distances
    dist_thumb_index_1 = calculate_distance(thumb_tip_1, index_tip_1)
    dist_thumb_index_2 = calculate_distance(thumb_tip_2, index_tip_2)

    # Check if the distances are within a threshold to form a heart
    return dist_thumb_index_1 < 50 and dist_thumb_index_2 < 50

# Function to draw a heart shape
def draw_heart(frame, center, size, color):
    cv2.ellipse(frame, (center[0] - size // 2, center[1]), (size // 4, size // 2), 0, 0, 180, color, -1)
    cv2.ellipse(frame, (center[0] + size // 2, center[1]), (size // 4, size // 2), 0, 0, 180, color, -1)
    cv2.rectangle(frame, (center[0] - size // 4, center[1]), (center[0] + size // 4, center[1] + size // 2), color, -1)

# Open webcam
cap = cv2.VideoCapture(0)

# Variables for floating hearts, confetti, and explosions
floating_hearts = []
confetti_particles = []
explosion_particles = []
gesture_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a more natural feel
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in hand_landmarks.landmark]

            # Check for heart gesture
            if len(results.multi_hand_landmarks) == 2:  # Ensure both hands are detected
                if is_heart_gesture(landmarks):
                    # Increment gesture counter
                    gesture_counter += 1

                    # Add floating hearts
                    floating_hearts.append([random.randint(0, frame.shape[1]), frame.shape[0], random.randint(10, 30), random.choice([(255, 0, 0), (0, 255, 0), (0, 0, 255)])])

                    # Add confetti particles
                    for _ in range(10):
                        confetti_particles.append([
                            random.randint(0, frame.shape[1]),
                            random.randint(0, frame.shape[0]),
                            random.choice([(255, 0, 0), (0, 255, 0), (0, 0, 255)]),
                            random.uniform(-2, 2),
                            random.uniform(2, 5)
                        ])

                    # Add explosion particles
                    for _ in range(50):
                        explosion_particles.append([
                            landmarks[8][0],  # Start at index finger tip
                            landmarks[8][1],
                            random.choice([(255, 0, 0), (0, 255, 0), (0, 0, 255)]),
                            random.uniform(-5, 5),
                            random.uniform(-5, 5)
                        ])

                    # Display Valentine's Day message
                    cv2.putText(frame, "Happy Valentine's Day!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for heart in floating_hearts:
        heart[1] -= 5
        draw_heart(frame, (heart[0], heart[1]), heart[2], heart[3])
    floating_hearts = [heart for heart in floating_hearts if heart[1] > 0]  # Remove off-screen hearts

    for particle in confetti_particles:
        particle[0] += particle[3]
        particle[1] += particle[4]  # Move particle vertically
        cv2.circle(frame, (int(particle[0]), int(particle[1])), 5, particle[2], -1)
    confetti_particles = [p for p in confetti_particles if 0 <= p[1] < frame.shape[0]]  # Remove off-screen particles

    # Update and draw explosion particles
    for particle in explosion_particles:
        particle[0] += particle[3]  # Move particle horizontally
        particle[1] += particle[4]  # Move particle vertically
        cv2.circle(frame, (int(particle[0]), int(particle[1])), 5, particle[2], -1)
    explosion_particles = [p for p in explosion_particles if 0 <= p[0] < frame.shape[1] and 0 <= p[1] < frame.shape[0]]

    # Display gesture counter
    # cv2.putText(frame, f"Heart Gestures: {gesture_counter}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show the frame
    cv2.putText(frame, "Happy Valentine's Day", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    x = 10
    y = 50  
    cv2.putText(frame, "aryan-17", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()