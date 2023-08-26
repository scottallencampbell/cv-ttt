import cv2
import numpy as np
from main import interpret

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frame = 0
votes_required = 5
candidate_board = None
board = [' '] * 9
votes = 0

while True:
    success, original = cap.read()
    final, latest_board = interpret(original, board)
    cv2.imshow(f'Tic tac toe', final) 
    cv2.waitKey(1)
    frame += 1
    
    if latest_board != None and latest_board != board:
        
        if candidate_board == None:
            candidate_board = latest_board
        else:
            if latest_board == candidate_board:
                votes += 1
            else:
                votes = 0
                candidate_board = None
            
            print(f'votes: {votes} {candidate_board}')
            
            if votes >= votes_required:
                board = candidate_board

if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()
    
    
    