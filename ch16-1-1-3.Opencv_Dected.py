import cv2
import keras

img = cv2.imread("test3.jpg")
#cv2.imshow("original", img)
#cv2.waitKey(0)
print(img.shape)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("img_gray", img_gray)
#cv2.waitKey(0)
print(img_gray.shape)

ret, img_th = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
#cv2.imshow("img_th", img_th)
#cv2.waitKey(0)

contours, hierachy = cv2.findContours(img_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, contours, -1, (255,125,125), 3)
#cv2.imshow("img", img)
#cv2.waitKey(0)
print(len(contours))

rects= [cv2.boundingRect(each) for each in contours]

for rect in rects:
    print(rect)
    cv2.rectangle(img_gray, (rect[0], rect[1]), (rect[0] + rect[2],
                            rect[1] + rect[3]), (0, 255, 0), 3)
    #cv2.imshow("drawContours",img_gray)
    #cv2.waitKey(0)

img_for_class = img_th.copy()
#cv2.imshow("img_for_class",img_for_class)
#cv2.waitKey(0)
margin_pixel=10
mnist_imgs = []
for rect in rects:
    print(rect)
    im=img_for_class[rect[1] - margin_pixel:rect[1] + rect[3]
                            + margin_pixel, rect[0] - margin_pixel:
                            rect[0] + rect[2] + margin_pixel]
#    cv2.imshow("im", im)
#    cv2.waitKey(0)
    resized_img = cv2.resize(im, dsize=(28, 28), interpolation=cv2.INTER_AREA)
    mnist_imgs.append(resized_img)
#    cv2.imshow("resized_img", resized_img)
#    cv2.waitKey(0)
print('len(mnist_imgs)===?',len(mnist_imgs))

import numpy as np
model = keras.models.load_model('MNIST_CNN.hdf5')
for i in range(len(mnist_imgs)):
    img = mnist_imgs[i]
    cv2.imshow("img", img)
    cv2.waitKey(0)
    img = img.reshape(-1, 28, 28, 1)
    input_data = (np.array(img) / 255)
    res = np.argmax(model.predict(input_data), axis=-1)
    print(res)