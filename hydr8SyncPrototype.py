# Algorithm if the liquid content in a water bottle is empty. 
import cv2
import imutils

bottle1 = cv2.imread("./Water Refill Modiftn/sampleImages/Bottle1.png")

# Glass Bottle
bottle_test1 = cv2.split(bottle1)[0]
cv2.imshow("Glass Bottle1", bottle_test1)
cv2.waitKey(0)

(T, bottle_threshold) = cv2.threshold(bottle_test1, 28, 256, cv2.THRESH_BINARY_INV)
cv2.imshow("Bottle1 Threshold 256 mL", bottle_threshold)
cv2.waitKey(0)

liquidContent = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
bottle_open = cv2.morphologyEx(bottle_threshold, cv2.MORPH_OPEN, liquidContent)
# cv2.imshow("Open Bottle", bottle_open)
# cv2.waitKey(0)

contours = cv2.findContours(
    bottle_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
contours = imutils.grab_contours(contours)
bottle_clone = bottle1.copy()

cv2.drawContours(bottle_clone, contours, -1, (255, 0, 0))
# cv2.imshow("All Contours", bottle_clone)
# cv2.waitKey(0)

areas = [cv2.contourArea(contour) for contour in contours]

(contours, areas) = zip(*sorted(zip(contours, areas), key=lambda a: a[1]))

bottle_clone = bottle1.copy()
cv2.drawContours(bottle_clone, [contours[-1]], -1, (255, 0, 0))
cv2.imshow("Largest Contour (Amount of Liquid)", bottle_clone)
cv2.waitKey(0)

bottle_clone = bottle1.copy()
(x, y, w, h) = cv2.boundingRect(contours[-1])
liquidWtoHRatio = w / float(h)
if liquidWtoHRatio < 0.4:
    cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        bottle_clone,
        "Full.",
        (x + 10, y + 20),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (0, 255, 0),
        2,
    )
else:
    cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(
        bottle_clone,
        "Low. Filling...",
        (x + 10, y + 20),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (0, 0, 255),
        2,
    )

cv2.imshow("Decision", bottle_clone)
cv2.waitKey(0)
