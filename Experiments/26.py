import cv2

img = cv2.imread('image.jpg')
watermark = 'OpenCV Watermark'
cv2.putText(img, watermark, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
            1, (255, 255, 255), 2)

cv2.imshow('Watermarked Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
