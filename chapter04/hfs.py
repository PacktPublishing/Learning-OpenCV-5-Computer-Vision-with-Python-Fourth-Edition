import cv2

img = cv2.imread('../images/statue_small.jpg')
hfsSegmenter = cv2.hfs.HfsSegment_create(
    height=img.shape[0], width=img.shape[1],
    minRegionSizeI=50, minRegionSizeII=100)
segmented_img = hfsSegmenter.performSegmentCpu(img)
cv2.imshow("HFS segmentation", segmented_img)
cv2.waitKey()
