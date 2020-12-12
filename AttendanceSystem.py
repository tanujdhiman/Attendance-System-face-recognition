import numpy as np 
import os
import face_recognition as fr 
import cv2
from datetime import datetime
import csv

path = 'ImagesAttendance/'

images = []
classNames = []

names = os.listdir(path)
print(names)

for cl in names:
	curImg = cv2.imread(f'{path}{cl}')
	images.append(curImg) 
	classNames.append(os.path.splitext(cl)[0])

print(classNames)

def findEncodings(images):
	encodings = []

	for img in images:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		#faceLoc = fr.face_locations(tan_image)[0]
		Enco = fr.face_encodings(img)[0]
		encodings.append(Enco)
	return encodings

def mark_attendance(name):
	with open('Attendance.csv', 'r+') as f:
		myDataList = f.readlines()
		#print(myDataList)
		nameslist = []
		for line in myDataList:
			entry = line.split(',')
			nameslist.append(entry[0])

		if name not in nameslist:
			now = datetime.now()
			dtstring = now.strftime('%H:%M:%S') 
			f.writelines(f'\n{name}, {dtstring}')




encodelistknown = findEncodings(images)
print('Encoding complete')

cap = cv2.VideoCapture()

while True :
	sucess, img = cap.read()
	cam_img = cv2.resize(img, (0, 0), None, 0.255, 0.255)
	cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)

	cam_face_loc = fr.face_locations(cam_img)
	cam_face_enc = fr.face_encodings(cam_img, cam_face_loc)

	for encodeface, locs in zip(cam_face_enc, cam_face_loc):
		matches = fr.compare_faces(encodelistknown, encodeface)
		faces_dis = fr.face_distance(encodelistknown, encodeface)
		print(faces_dis)

		matchIndex = np.argmin(faces_dis)

		if matches[matchIndex]:
			name = classNames[matchIndex].upper()
			print(name)
			y1, x2, y2, x1 = cam_face_loc
			y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
			cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
			cv2.rectangle(img, (x1, y1 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
			cv2.putText(img, name, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
			mark_attendance(name)





	cv2.imshow('Web Cam', img)
	cv2.waitKey(1)





