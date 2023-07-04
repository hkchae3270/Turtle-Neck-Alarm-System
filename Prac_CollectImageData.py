# Collect Image
# State : Normal or Abnormal

import cv2
import cvlib as cv

# Open Camera(Webcam : 0 / USB Prot Connection Camera : 1)
webcam = cv2.VideoCapture(0)

# Set Camera Frame
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

sample_num = 0
captured_num = 0

# Loop through Frames
while webcam.isOpened():

    # Read Frame from Camera
    status, frame = webcam.read()
    frame = cv2.flip(frame, 1)  # 좌우 대칭 변경
    sample_num = sample_num + 1

    if not status:
        break

    # Detect Face
    face, confidence = cv.detect_face(frame)

    print(face)
    print(confidence)

    # Loop through Faces
    for idx, f in enumerate(face):

        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]*2

        if sample_num % 8 == 0:
            captured_num = captured_num + 1
            face_in_img = frame[startY:endY, startX:endX, :]
            # Collect Images - Abnormal State
            # cv2.imwrite('./image/abnormal/face' + str(captured_num) + '.jpg', face_in_img)  # 마스크 미착용 데이터 수집시 주석처리
            # Collect Images - Normal State
            cv2.imwrite('./image/normal/face'+str(captured_num)+'.jpg', face_in_img) # 마스크 미착용 데이터 수집시 주석해제

    # Display Image
    cv2.imshow("captured frames", frame)
              
    # Press "Q" to Stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()