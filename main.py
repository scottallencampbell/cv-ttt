import cv2
import numpy as np
import vision
import engine 
from constants import *

def play():    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    current_player = X
    board = [EMPTY] * 9
    candidate_board = None
    frame = 0
    votes = 0
    best_move = -1
    
    # Just in case we have a initial board state, let's read in the entire board
    _, original = cap.read()        
    _, _, _, initial_board = vision.interpret(original, board)
    
    if initial_board is not None:
        board = initial_board
        
    while True:
        _, original = cap.read()
        
        updated_board, state, _ = evalate_and_display_board(original, board, frame, current_player, best_move)
        
        # a negative frame indicates that the game is over and serves to flash the gridlines
        if state != GAME_ACTIVE and frame >= 0:
            frame = -1
            
        if frame < -20:
            break
        
        # increment the frame in either direction, frame mod 2 is used to flash move or gamestate indicators
        frame += 1 if frame >= 0 else -1
        
        if updated_board is None:
            continue
        
        # detect if a change has occurred to the board, but require a given number of votes before accepting the move
        votes, candidate_board = evaluate_potential_board_update(board, updated_board, candidate_board, votes)
        
        if votes < CONSECUTIVE_VOTES_REQUIRED_FOR_MOVE:
            continue
            
        # compare the current board state with the updated state to determine which cell was played and by whom
        cell, last_player = get_last_move(board, candidate_board)                    

        if last_player != current_player:
            print("The most recent player played out of turn")
            # game is over
            frame = -1
            
        board[cell] = last_player
        current_player = -current_player

        # get the next move if it's the AI's turn
        if current_player == AI_PLAYER:
            best_move = engine.find_best_move(board, current_player)
        
    if cv2.waitKey(0):  
        cv2.destroyAllWindows()

def evalate_and_display_board(img, board, frame, current_player, best_move):
    # first, determine the current game state (draw/win/active) as well as whose turn it is, 
    # and if it's a win, where is the three-in-a-row located
    state, turn, win_type, win_info = engine.get_state(board)
    
    # now look at the current webcam frame, run graphical processing on it, rotate it to orthogonality,
    # and get the resulting board state
    copy, grayscale, decorated, updated_board = vision.interpret(img, board)

    # create an image showing the virtual state of the gameboard
    if state != GAME_ACTIVE or frame < 0:
        # if the game is won, draw the line showing the three in a row , else if the game is a draw, flash the gridlines
        virtual = vision.get_virtual_gameboard(grayscale, board, frame, turn, None, state, win_type, win_info)        
    elif current_player == AI_PLAYER and frame % 2 == 0:
        # if it's the AI's turn, flash the move it wants to make, until the user fills in the cell
        virtual = vision.get_virtual_gameboard(grayscale, board, frame, current_player, best_move)
    else:
        # else just show the board, it's the player's turn to move 
        virtual = vision.get_virtual_gameboard(grayscale, board, frame)
        
    # build a 2x2 array of different views of the gameboard
    left = np.concatenate((copy, decorated), axis=0)
    right = np.concatenate((grayscale, virtual), axis=0)
    final = np.concatenate((left, right), axis=1)
    
    cv2.imshow(f'Tic tac toe', final) 
    cv2.waitKey(100)
    
    return updated_board, state, turn

def evaluate_potential_board_update(board, last_board, candidate_board, votes):
    
    if last_board is None or last_board == board:
         return 0, candidate_board
     
    # we have our first candidate_board 
    if candidate_board is None:
        candidate_board = last_board
        votes = 1
    else:
        if last_board == candidate_board:
            # we have an additional consecutive vote for this board
            votes += 1
        else:
            # whoops, the candidate board missed a vote, start from scratch
            votes = 0
            candidate_board = None
        
    return votes, candidate_board

def get_last_move(board, candidate_board):
    
    for i, cell in enumerate(board):
        
        if cell != candidate_board[i]:
            return i, candidate_board[i]
    
    raise Exception("Failed to find a difference between the current board and the candidate board")

if __name__ == "__main__":
    play()    
