import pywhatkit as kit
import cv2


kit.text_to_handwriting("hey shubham", save_to="writing.png")


img = cv2.imread("writing.png")


cv2.imshow("Text to Handwriting", img)
cv2.waitKey(0)
cv2.destroyAllWindows()