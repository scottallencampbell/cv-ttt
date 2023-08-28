# Adapted from https://github.com/agrawal-rohit/tic-tac-toe-bot/blob/master/HumanvsAI_Minimax.py

from math import inf as infinity
from constants import *

players = [X,O]

def play_move(state, player, block_num):
    if state[int((block_num)/3)][(block_num)%3] == EMPTY:
        state[int((block_num)/3)][(block_num)%3] = player
    else:
        raise Exception(f"Failed to play cell {block_num}, the cell is not empty")
        
def copy_game_state(state):
    #new_state = [[EMPTY] * 3] * 3    #  engine doesn't work with this .... but why???
    new_state = [[EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY], [EMPTY, EMPTY, EMPTY]]
    for i in range(3):
        for j in range(3):
            new_state[i][j] = state[i][j]
    return new_state
    
def check_current_state(game_state):
    # Check horizontals
    if (game_state[0][0] == game_state[0][1] and game_state[0][1] == game_state[0][2] and game_state[0][0] != EMPTY):
        return game_state[0][0], GAME_DONE
    if (game_state[1][0] == game_state[1][1] and game_state[1][1] == game_state[1][2] and game_state[1][0] != EMPTY):
        return game_state[1][0], GAME_DONE
    if (game_state[2][0] == game_state[2][1] and game_state[2][1] == game_state[2][2] and game_state[2][0] != EMPTY):
        return game_state[2][0], GAME_DONE
    
    # Check verticals
    if (game_state[0][0] == game_state[1][0] and game_state[1][0] == game_state[2][0] and game_state[0][0] != EMPTY):
        return game_state[0][0], GAME_DONE
    if (game_state[0][1] == game_state[1][1] and game_state[1][1] == game_state[2][1] and game_state[0][1] != EMPTY):
        return game_state[0][1], GAME_DONE
    if (game_state[0][2] == game_state[1][2] and game_state[1][2] == game_state[2][2] and game_state[0][2] != EMPTY):
        return game_state[0][2], GAME_DONE
    
    # Check diagonals
    if (game_state[0][0] == game_state[1][1] and game_state[1][1] == game_state[2][2] and game_state[0][0] != EMPTY):
        return game_state[1][1], GAME_DONE
    if (game_state[2][0] == game_state[1][1] and game_state[1][1] == game_state[0][2] and game_state[2][0] != EMPTY):
        return game_state[1][1], GAME_DONE
    
    # Check if draw
    draw_flag = 0
    for i in range(3):
        for j in range(3):
            if game_state[i][j] == EMPTY:
                draw_flag = 1
    if draw_flag == 0:
        return None, GAME_DRAW
    
    return None, GAME_ACTIVE
    
def get_best_move(state, player):
    '''
    Minimax Algorithm
    '''
    winner_loser , done = check_current_state(state)
    if done == GAME_DONE and winner_loser == O: # If AI won
        return (1,0)
    elif done == GAME_DONE and winner_loser == X: # If Human won
        return (-1,0)
    elif done == GAME_DRAW:    # Draw condition
        return (0,0)
        
    moves = []
    empty_cells = []
    for i in range(3):
        for j in range(3):
            if state[i][j] == EMPTY:
                empty_cells.append(i*3 + j)
    
    for empty_cell in empty_cells:
        move = {}
        move[KEY_INDEX] = empty_cell
        new_state = copy_game_state(state)
        play_move(new_state, player, empty_cell)
        
        if player == O:    # If AI
            result,_ = get_best_move(new_state, X)    # make more depth tree for human
            move[KEY_SCORE] = result
        else:
            result,_ = get_best_move(new_state, O)    # make more depth tree for AI
            move[KEY_SCORE] = result
        
        moves.append(move)

    # Find best move
    best_move = None
    if player == O:   # If AI player
        best = -infinity
        for move in moves:            
            if move[KEY_SCORE] > best:
                best = move[KEY_SCORE]
                best_move = move[KEY_INDEX]
    else:
        best = infinity
        for move in moves:
            if move[KEY_SCORE] < best:
                best = move[KEY_SCORE]
                best_move = move[KEY_INDEX]
                
    return (best, best_move)