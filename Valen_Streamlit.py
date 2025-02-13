import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import random
from PIL import Image

# Set page config
st.set_page_config(page_title="Valentine's Day Gesture App", layout="wide")

# Initialize MediaPipe Hands
@st.cache_resource
def load_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2)
    mp_draw = mp.solutions.drawing_utils
    return mp_hands, hands, mp_draw

mp_hands, hands, mp_draw = load_mediapipe()

# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to check if hands form a heart shape
def is_heart_gesture(landmarks):
    thumb_tip_1 = landmarks[4]
    index_tip_1 = landmarks[8]
    thumb_tip_2 = landmarks[20]
    index_tip_2 = landmarks[12]
    
    dist_thumb_index_1 = calculate_distance(thumb_tip_1, index_tip_2)
    dist_thumb_index_2 = calculate_distance(thumb_tip_2, index_tip_1)
    
    return dist_thumb_index_1 < 50 and dist_thumb_index_2 < 50

# Function to draw a heart shape
def draw_heart(frame, center, size, color):
    cv2.ellipse(frame, (center[0] - size // 2, center[1]), (size // 4, size // 2), 0, 0, 180, color, -1)
    cv2.ellipse(frame, (center[0] + size // 2, center[1]), (size // 4, size // 2), 0, 0, 180, color, -1)
    cv2.rectangle(frame, (center[0] - size // 4, center[1]), (center[0] + size // 4, center[1] + size // 2), color, -1)

def main():
    # Streamlit UI
    st.title("💝 Valentine's Day Gesture Recognition")
    st.write("Make a heart shape with your hands to see the magic! ✨")

    # Initialize session state for particles
    if 'floating_hearts' not in st.session_state:
        st.session_state.floating_hearts = []
    if 'confetti_particles' not in st.session_state:
        st.session_state.confetti_particles = []
    if 'explosion_particles' not in st.session_state:
        st.session_state.explosion_particles = []
    if 'gesture_counter' not in st.session_state:
        st.session_state.gesture_counter = 0

    # Create a placeholder for webcam feed
    video_placeholder = st.empty()
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break

        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) 
                            for lm in hand_landmarks.landmark]
                
                if len(results.multi_hand_landmarks) == 2:
                    if is_heart_gesture(landmarks):
                        st.session_state.gesture_counter += 1
                        
                        # Add effects
                        st.session_state.floating_hearts.append([
                            random.randint(0, frame.shape[1]), 
                            frame.shape[0], 
                            random.randint(10, 30),
                            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        ])
                        
                        # Add message
                        cv2.putText(frame, "Happy Valentine's Day!", 
                                  (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Update and draw effects
        for heart in st.session_state.floating_hearts:
            heart[1] -= 5
            draw_heart(frame, (heart[0], heart[1]), heart[2], heart[3])
        
        st.session_state.floating_hearts = [heart for heart in st.session_state.floating_hearts 
                                          if heart[1] > 0]

        # Convert frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()

if __name__ == "__main__":
    main()