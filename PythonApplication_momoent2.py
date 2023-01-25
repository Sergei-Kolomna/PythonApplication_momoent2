import cv2
import numpy as np

# сжимаем картинку на scale_percent %
def img_zip(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dsize = (width, height)
    output = cv2.resize(img, dsize)
    return output

def nothing(*arg):
    pass

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
         print(x, ' ', y)
    if event==cv2.EVENT_RBUTTONDOWN:
         print(x, ' ', y)

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

cv2.namedWindow("result") # создаем главное окно
cv2.namedWindow("settings") # создаем окно настроек

#set a thresh
cv2.createTrackbar('threshold', 'settings', 0, 255, nothing)
cv2.setTrackbarPos('threshold', 'settings', 25)

# координаты образки экрана
x_min = 180
x_max = 450
y_min = 125
y_max = 285
# HSV для зеленого цвета - обведенного изображения
green_hsv_min = np.array((20, 175, 30), np.uint8)
green_hsv_max = np.array((175, 255, 255), np.uint8)

# Работа с видео
cap = cv2.VideoCapture("SCHOM12.mp4")
while cap.isOpened():
    ret, image = cap.read()
    image_crop = image[y_min:y_max, x_min:x_max]         # Обрезаем изображение, сперва высота, потом ширина
    image_crop = white_balance(image_crop)
    monochrome_image = cv2.cvtColor(image_crop,cv2.COLOR_BGR2GRAY)
    thresh = cv2.getTrackbarPos('threshold', 'settings')
    
    
    ret, thresh_image = cv2.threshold(monochrome_image, thresh, 255, cv2.THRESH_BINARY)
    #ищем контуры без адаптивного thresh
    contours, hierarchy = cv2.findContours(image=thresh_image.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    image_copy=image_crop.copy()
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)

# Теперь накладываем фильтр для зеленоого цвета:
    
#    green_hsv = cv2.cvtColor(image_copy, cv2.COLOR_BGR2HSV )
#    my_hsv = cv2.inRange(green_hsv, green_hsv_min, green_hsv_max)
#        # вычисляем моменты изображения
#    moments = cv2.moments(my_hsv, 1)
#    dM01 = moments['m01']
#    dM10 = moments['m10']
#    dArea = moments['m00']
#    # будем реагировать только на те моменты, которые содержать больше 100 пикселей
#    if dArea > 100:
#        x = int(dM10 / dArea)
#        y = int(dM01 / dArea)
#        cv2.circle(image_copy, (x, y), 10, (0,0,255), -1)
     
#    image[180:450, 125:285] = image_copy
    image[y_min:y_max, x_min:x_max] = image_copy
    cv2.imshow("Only_threshold", image)

    #cv2.setMouseCallback('Only_threshold', click_event)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


