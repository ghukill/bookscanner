# BG BACK
image = cv2.imread('/home/commander/dev/local/bookscanner/data/natural_light/P1260242.JPG')
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)
# image = cv2.blur(image,(5,5))
image = cv2.GaussianBlur(image,(5,5),0)
edged = cv2.Canny(
        image, # input image
        50, # minVal
        200 # maxVal
    )
plt.imshow(edged)
img, cnts, hierarchy = cv2.findContours(
    edged.copy(), # source image
    cv2.RETR_LIST, # contour retrieval mode
    cv2.CHAIN_APPROX_SIMPLE # contour approximation method
)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
for each in cnts:
    print(cv2.contourArea(each))



# SPLIT PAGE
image = cv2.imread('/home/commander/dev/local/bookscanner/data/natural_light_done/P1260242.JPG')
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)
# image = cv2.blur(image,(5,5))
image = cv2.GaussianBlur(image,(5,5),0)
edged = cv2.Canny(
        image, # input image
        50, # minVal
        200 # maxVal
    )
plt.imshow(edged)