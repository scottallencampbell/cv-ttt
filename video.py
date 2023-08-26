import math
import cv2
import numpy as np
from main import interpret

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frame = 0

while True:
    success, original = cap.read()
    final = interpret(original)
    cv2.imshow(f'Tic tac toe', final) 
    cv2.waitKey(1)
    print(frame)
    frame += 1
    
if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()
    
    
    