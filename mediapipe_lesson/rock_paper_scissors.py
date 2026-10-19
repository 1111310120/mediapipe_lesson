import cv2
import mediapipe as mp

# 初始化 MediaPipe Hands 模組
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # 支援最多偵測雙手
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# 判斷手指伸直與猜拳型態的函數
def get_gesture(hand_landmarks):
    # 食指(8)、中指(12)、無名指(16)、小指(20)的指尖與關節節點
    fingers = []
    finger_tips = [8, 12, 16, 20]
    
    # 若指尖 y 座標小於下方關節 y 座標，代表手指伸直 (1: 伸直, 0: 彎曲)
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # 計算四指伸直數量
    total_open = sum(fingers)

    # 分類猜拳型態
    if total_open == 0:
        return "Rock (石頭)"
    elif total_open == 4:
        return "Paper (布)"
    elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0:
        return "Scissors (剪刀)"
    else:
        return "Unknown (無法判定)"

# 開啟攝影機
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("請按 'q' 鍵結束程式...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("無法讀取攝影機畫面")
        break

    # 水平翻轉畫面，使其如同鏡像操作
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # 處理偵測到的手部
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # 1. 區分左右手 (MediaPipe 翻轉後標籤：Left 代表右手，Right 代表左手)
            label = handedness.classification[0].label
            hand_label = "Right (右手)" if label == "Left" else "Left (左手)"
            
            # 2. 判斷猜拳型態
            gesture = get_gesture(hand_landmarks)

            # 3. 繪製手部骨架
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 取得手腕點 (Landmark 0) 座標以顯示文字
            h, w, _ = frame.shape
            cx, cy = int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h)

            # 於畫面上顯示結果
            text = f"{hand_label}: {gesture}"
            cv2.putText(frame, text, (cx - 50, cy + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Rock Paper Scissors - MediaPipe", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()