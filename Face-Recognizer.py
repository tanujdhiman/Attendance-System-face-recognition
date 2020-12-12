import cv2
import numpy as np
import face_recognition as fr

org_image = fr.load_image_file('images/Sharukhan.jpg')
test_image = fr.load_image_file('images/Akshay.jpg')

org_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

faceLoc = fr.face_locations(org_image)[0]
faceEnco = fr.face_encodings(org_image)[0]
# print(faceLoc)

cv2.rectangle(org_image, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = fr.face_locations(test_image)[0]
faceEncoTest = fr.face_encodings(test_image)[0]
# print(faceLoc)

cv2.rectangle(test_image, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = fr.compare_faces([faceEnco], faceEncoTest)
facedis = fr.face_distance([faceEnco], faceEncoTest)
print(results, facedis)

cv2.putText(test_image, f'{results}, {round(facedis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))


cv2.imshow('Actual Image', org_image)
cv2.imshow('Test Image', test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()