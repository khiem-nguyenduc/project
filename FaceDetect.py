import cv2
import os

cam = cv2.VideoCapture(0)
# Set chiều ngang, chiều dọc camera
cam.set(3, 640)
cam.set(4, 480)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Nhập id của từng người
face_id = input('\n Nhập ID của khuôn mặt để lưu trữ: ')

print("\n [INFO] Khởi tạo camera")
# Đếm số khuôn mặt
count = 0

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)     
        count += 1

        # Lưu hình ảnh vào thư mục dataset
        cv2.imwrite("dataset/ID." + str(face_id) + '.' + str(count) + ".jpg", gray[y : y + h, x : x + w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Nhấn ESC để thoát
    if k == 27:
        break
    elif count >= 50: # Lấy 50 khuôn mặt của mỗi người
         break

# Do a bit of cleanup
print("\n [INFO] Thoát")
cam.release()
cv2.destroyAllWindows()


