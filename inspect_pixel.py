import cv2
import numpy as np

img = cv2.imread('temp_orthofoto.png')
if img is None:
    print('Image not found!')
    exit(1)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

win_name = 'Click to Inspect Pixel (ESC to quit)'
cv2.namedWindow(win_name)

print('Click on a pixel to inspect its values (HSV, LAB, RGB).')

# Mouse callback
def show_pixel(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        bgr = img[y, x]
        h, s, v = hsv[y, x]
        l, a, b = lab[y, x]
        print(f'Pixel ({x},{y}): BGR={bgr}, HSV=({h},{s},{v}), LAB=({l},{a},{b})')
        cv2.circle(img, (x, y), 5, (0, 255, 255), 2)
        cv2.imshow(win_name, img)

cv2.setMouseCallback(win_name, show_pixel)

while True:
    cv2.imshow(win_name, img)
    if cv2.waitKey(10) == 27:
        break
cv2.destroyAllWindows()
