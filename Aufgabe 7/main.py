import cv2
import mediapipe as mp
import time
from enum import Enum

class SSP(Enum):
    Schere = 1
    Stein = 2
    Papier = 3  

def getRandomSymbol():
    import random
    return SSP(random.randint(1, len(SSP)))

win = {
        SSP.Schere: [SSP.Papier],
        SSP.Papier: [SSP.Stein],
        SSP.Stein: [SSP.Schere],
    }

def decypher(landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    if thumb_tip.y < index_finger_tip.y and thumb_tip.y < middle_finger_tip.y and thumb_tip.y < ring_finger_tip.y and thumb_tip.y < pinky_tip.y:
        return SSP.Stein
    elif index_finger_tip.y < middle_finger_tip.y and index_finger_tip.y < ring_finger_tip.y and index_finger_tip.y < pinky_tip.y:
        return SSP.Schere
    elif thumb_tip.y > index_finger_tip.y and thumb_tip.y > middle_finger_tip.y and thumb_tip.y > ring_finger_tip.y and thumb_tip.y > pinky_tip.y:
        return SSP.Papier


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

start_time = time.time()

recognized_symbol = None
random_symbol = None
winner = ''

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1) as hands:
  
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        current_time = time.time()
        
        if current_time - start_time >= 2:
            recognized_symbol = decypher(hand_landmarks)
            random_symbol = getRandomSymbol()

            if recognized_symbol == None:
                 continue

            if random_symbol in win[recognized_symbol]:
                 winner = 'Spieler'
            elif recognized_symbol in win[random_symbol]:
                 winner = 'Computer'
            else:
                 winner = 'Unentschieden'

            start_time = current_time

        if recognized_symbol == SSP.Stein:
                cv2.putText(image, "Stein", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif recognized_symbol == SSP.Schere:
                cv2.putText(image, "Schere", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif recognized_symbol == SSP.Papier:
                cv2.putText(image, "Papier", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if random_symbol == SSP.Stein:
                cv2.putText(image, "Stein", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        elif random_symbol == SSP.Schere:
                cv2.putText(image, "Schere", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        elif random_symbol == SSP.Papier:
                cv2.putText(image, "Papier", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.putText(image, winner, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        

    cv2.imshow('Schere, Stein, Papier', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()


