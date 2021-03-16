import numpy as np
import time
import cv2
import os

INPUT_FILE='IMG_4924.JPEG'
OUTPUT_FILE='predicted.jpg'
LABELS_FILE='data/coco.names'
CONFIG_FILE='cfg/yolov4.cfg'
WEIGHTS_FILE='yolov4.weights'
ressources = 'RessourcesImagesVins'
CONFIDENCE_THRESHOLD=0.3
#initialise descriptor
orb = cv2.ORB_create()

LABELS = open(LABELS_FILE).read().strip().split("\n")

np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")


net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

image = cv2.imread(INPUT_FILE)
(H, W) = image.shape[:2]

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()o
	kp2,des2 = orb.detectAndCompute(img,None)
	matchList=[]
	finalVal=-1
	for kp1,desL in desList:
		if (len(desL) == 0):
			print('desL is empty')
			exit(1)
		if (len(des2) == 0):
			print('des2 is empty')
			exit(1)
		matches = flann.knnMatch(desL, des2, k=2)
		cor = []
		# ratio test as per Lowe's paper
		for m_n in matches:
			if len(m_n) != 2:
				continue
			elif m_n[0].distance < 0.80 * m_n[1].distance:
				cor.append([kp1[m_n[0].queryIdx].pt[0], kp1[m_n[0].queryIdx].pt[1],
							kp2[m_n[0].trainIdx].pt[0], kp2[m_n[0].trainIdx].pt[1],
							m_n[0].distance])
		matchList.append(len(cor))
	print(matchList)
	if len(matchList)!=0:
		if max(matchList)>thres:
			finalVal=matchList.index(max(matchList))
	end = time.time()
	print("[INFO] Matching took {:.6f} seconds".format(end - start))
	return finalVal

# ensure at least one detection exists
nbouteilles = 0
if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		color = [int(c) for c in COLORS[classIDs[i]]]

		cv2.rectangle(image, (x, y), (x + w, y + h), color, 8)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		#print('A ' + LABELS[classIDs[i]] + ' has been found')
		if(LABELS[classIDs[i]] == 'bottle' and confidences[i] >0.8) :
			print('A bottle has been found')
			nbouteilles += 1
			crop_img = image[abs(y):y + h, abs(x):x + w]
			cv2.imwrite("example1.png", crop_img)
			crop_img_resize = cv2.resize(crop_img, (int(1920/10),int(1080/2)))
			cv2.imshow("cropped"+str(i), crop_img_resize)
			cv2.moveWindow("cropped"+str(i), int(1920/10*(nbouteilles-1)), int(1080/2))
			kp, des = orb.detectAndCompute(crop_img, None)
			crop_img_wkp = drawKeyPts(crop_img.copy(), kp, (0, 255, 0), 2)
			crop_img_wkp = cv2.resize(crop_img_wkp, (int(1920/10),int(1080/2)))
			cv2.imshow("cropped with keypoints"+str(i), crop_img_wkp)
			cv2.moveWindow("cropped with keypoints"+str(i), int((1920/2)+1920/10*(nbouteilles-1)), int(int(1080/2)))
			id = findID(crop_img, desList, thres=1)
			if id!=-1:
				correspondingImage = cv2.imread(f'{ressources}/{myList[id]}', 1)
				correspondingImage = cv2.resize(correspondingImage, (int(1920 / 8), int(1080 / 2)))
				cv2.imshow("Corresponding image found", correspondingImage)
				cv2.moveWindow("Corresponding image found", int((1920 / 4) *3),
							   int(0))
			else:
				print('Not found')

		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)



# show the output image
cv2.imwrite("example.png", image)
crop_image = cv2.resize(image, (int(1920/2),int(1080/2)))
cv2.imshow("Original", crop_image)
cv2.moveWindow("Original",0,0)
cv2.waitKey(0)