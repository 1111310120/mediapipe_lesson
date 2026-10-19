import cv2
import mediapipe as mp

# Hàm tìm chỉ số camera hoạt động
def get_working_camera():
    for index in range(5):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"Đã tìm thấy camera hoạt động tại index: {index}")
                return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return None

cap = get_working_camera()

if cap is None:
    print("Khong tim thay webcam nao! Vui long kiem tra lai ket noi hoac quyen truy cap camera.")
    exit()

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

print("Nhan phim 'q' de thoat chuong trinh...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Khong the doc du lieu tu webcam")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow('MediaPipe Hands Test', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()