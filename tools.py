import cv2

def display(title, img):
    cv2.imshow(title, img)
    # Chờ một khoảng thời gian
    cv2.waitKey(0)
    # Đóng window
    cv2.destroyAllWindows()