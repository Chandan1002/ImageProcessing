import  cv2

cv2.namedWindow("output", cv2.WINDOW_NORMAL)
img = cv2.imread('E:\ImageProcessing\IMG_3787.jpeg', cv2.COLOR_RGB2HSV)
img_r = img[:,:,2]
thres,rimage=cv2.threshold(img_r,127,255,cv2.THRESH_BINARY)
kernelOpen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
opened = cv2.morphologyEx(rimage, cv2.MORPH_OPEN, kernelOpen)
opened = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
opened = cv2.morphologyEx(opened, cv2.MORPH_OPEN, kernelOpen)

img_g = img[:,:,1]
thres,gimage=cv2.threshold(img_g,127,255,cv2.THRESH_BINARY)
# kernelOpen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
g_opened = cv2.morphologyEx(gimage, cv2.MORPH_OPEN, kernelOpen)
g_opened = cv2.morphologyEx(g_opened, cv2.MORPH_CLOSE, kernel)
g_opened = cv2.morphologyEx(g_opened, cv2.MORPH_OPEN, kernelOpen)

g_not=cv2.bitwise_not(g_opened)
out_and=cv2.bitwise_and(opened,g_not)
final_img = cv2.bitwise_and(img,img,mask = out_and)
img123=cv2.resize(final_img, (1920, 1080))
# cv2.imshow('test', img)
# cv2.imshow('output', out_and)

# img_r.shape[300500]


contours, hierarchy = cv2.findContours(out_and, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# contours, hierarchy = cv2.findContours(final_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)


for i, c in enumerate(contours):
    if hierarchy[0][i][3] == -1:
        # calculate moments for each contour
        M = cv2.moments(c)
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a white rectangle to visualize the bounding rect
        cv2.rectangle(out_and, (x, y), (x+w, y+h), (255, 255, 255), 2)
        print(M)
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        cv2.circle(img, (cX, cY), 5, (255, 255, 0), 2)
    else:
        continue


cv2.imshow("Keypoints", img)
# cv2.waitKey(0)
cv2.imshow('output', out_and)
cv2.waitKey(0)
cv2.destroyAllWindows()

