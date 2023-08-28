# Adapted from https://stackoverflow.com/questions/72739112/mini-max-not-giving-optimal-move-tic-tac-toe

from math import inf as infinity
from constants import *

def rate_state(state):
    '''
    this def is returning
    GAME_X_WINS if X wins
    GAME_Y_WINS if O wins 
    GAME_ACTIVE if nothing 
    GAME_DRAW if draw
    '''
    ter = terminate(state)
    
    if ter != False:
        if state[ter[0]] == X:
            return GAME_X_WINS
        elif state[ter[0]] == O:
            return GAME_O_WINS
        
    is_draw = True
    
    for ws in state:
        if ws == EMPTY:
            is_draw = False
    
    return GAME_DRAW if is_draw == True else GAME_ACTIVE
            
def terminate(state):
    '''
    this def is returning
    position of same X or O in a line
    or False if board full not wins 
    '''
    win_pos = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
    
    for ws in win_pos:
        if state[ws[0]] != EMPTY and state[ws[0]] == state[ws[1]] and state[ws[0]] == state[ws[2]]:            
            return [ws[0], ws[1], ws[2]]
        
    return False

def get_best_move(board, player):
    best_score = -1000
    best_move = 0
    isMax = player == O
    
    for i in range(len(board)):
        if board[i] != EMPTY:
            continue
        
        board[i] = X

        move_score = deep(board, isMax)

        if move_score > best_score:
            best_score = move_score
            best_move = i

        board[i] = O

        move_score = -deep(board, isMax)

        if move_score > best_score:
            best_score = move_score
            best_move = i

        board[i] = EMPTY

    return best_move

def deep(state, isMax):
    state_score = rate_state(state)
    
    if state_score == GAME_X_WINS:
        return state_score
    elif state_score == GAME_O_WINS:
        return state_score
    
    if terminate(state) == False:
        return 0
    
    if isMax:
        score = -1000
        for itr in range(len(state)):
            if state[itr] == EMPTY:
                state[itr] = X
                score = max(score, deep(state, False))
                state[itr] = EMPTY
        return score
    
    else:
        score = 1000
        for itr in range(len(state)):
            if state[itr] == EMPTY:
                state[itr] = O
                score = min(score, deep(state, True))
                state[itr] = EMPTY
        return score
