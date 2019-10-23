import cv2
import time


img = cv2.imread("cat.jpg")
cv2.imshow("img",img)

# img_resize = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC )
# cv2.imwrite('output/%d.jpg'%(size), img_resize)
# cv2.imshow("img_resize",img_resize)
print(img.shape)

b = img[:,:,0]#得到蓝色通道
g = img[:,:,1]#得到绿色通道
r = img[:,:,2]#得到红色通道

print(b.shape)

cv2.imshow("img_b",b)
cv2.imshow("img_g",g)
cv2.imshow("img_r",r)


b_1,g_1,r_1 = cv2.split(img)#拆分通道
cv2.imshow("img_b_1",b_1)
cv2.imshow("img_g_1",g_1)
cv2.imshow("img_r_1",r_1)


cv2.waitKey()


