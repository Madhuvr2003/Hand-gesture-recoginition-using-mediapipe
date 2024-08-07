import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Function to draw text with a background
def draw_text_with_background(frame, text, position, font, scale, color, thickness, bgcolor):
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    cv2.rectangle(frame, (x, y - text_height - baseline), (x + text_width, y + baseline), bgcolor, -1)
    cv2.putText(frame, text, position, font, scale, color, thickness, cv2.LINE_AA)

# Function to detect hands and annotate frame with different styles
def detect_hands(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            handedness = hand_info.classification[0].label
            text_position = (50, 100)

            if handedness == 'Right':
                text_position = (frame.shape[1] - 250, 100)

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            if thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y and thumb_tip.y < ring_tip.y and thumb_tip.y < pinky_tip.y:
                draw_text_with_background(frame, 'Thumbs Up', text_position, cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 3, (0, 0, 0))

            if index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y and ring_tip.y > thumb_tip.y and pinky_tip.y > thumb_tip.y:
                draw_text_with_background(frame, 'Peace', text_position, cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 0, 0), 3, (0, 0, 0))

            if index_tip.y < thumb_tip.y and middle_tip.y > thumb_tip.y and ring_tip.y > thumb_tip.y and pinky_tip.y > thumb_tip.y:
                draw_text_with_background(frame, 'Warning', text_position, cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 255, 255), 3, (0, 0, 0))

            if thumb_tip.y < index_tip.y and middle_tip.y > ring_tip.y and middle_tip.y > pinky_tip.y:
                draw_text_with_background(frame, 'Washroom', text_position, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (255, 255, 0), 3, (0, 0, 0))

            if thumb_tip.y > index_tip.y and thumb_tip.y > middle_tip.y and thumb_tip.y > ring_tip.y and thumb_tip.y > pinky_tip.y:
                draw_text_with_background(frame, 'Hello', text_position, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, (255, 0, 255), 3, (0, 0, 0))

            if index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y and ring_tip.y < thumb_tip.y and pinky_tip.y > thumb_tip.y:
                draw_text_with_background(frame, 'Rock', text_position, cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 128, 255), 3, (0, 0, 0))

            if index_tip.y > thumb_tip.y and middle_tip.y < thumb_tip.y and ring_tip.y > thumb_tip.y and pinky_tip.y < thumb_tip.y:
                draw_text_with_background(frame, 'I Love You', text_position, cv2.FONT_HERSHEY_COMPLEX, 1.2, (128, 0, 128), 3, (0, 0, 0))

            if index_tip.y < thumb_tip.y and middle_tip.y > thumb_tip.y and ring_tip.y < thumb_tip.y and pinky_tip.y > thumb_tip.y:
                draw_text_with_background(frame, 'Bye', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 128), 3, (0, 0, 0))

            for point in hand_landmarks.landmark:
                x = int(point.x * frame.shape[1])
                y = int(point.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

    return frame

# Main function to capture video and detect hands
def main():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_hands(frame)
        cv2.imshow('Hand Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
