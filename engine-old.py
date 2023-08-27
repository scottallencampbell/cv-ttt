# adapted from https://www.geeksforgeeks.org/finding-optimal-move-in-tic-tac-toe-using-minmax-algorithm-in-game-theory/
X = 'X'
O = 'O'
EMPTY = ' '
player, opponent = X, O 

def evaluate(b):
  
    # Checking for Rows for X or O victory.
    for row in range(0, 3):
      
        if b[row][0] == b[row][1] and b[row][1] == b[row][2]:
          
            if b[row][0] == X:
                return 10
            elif b[row][0] == O:
                return -10
 
    # Checking for Columns for X or O victory.
    for col in range(0, 3):
      
        if b[0][col] == b[1][col] and b[1][col] == b[2][col]:
          
            if b[0][col]==X:
                return 10
            elif b[0][col] == O:
                return -10
 
    # Checking for Diagonals for X or O victory.
    if b[0][0] == b[1][1] and b[1][1] == b[2][2]:
      
        if b[0][0] == X:
            return 10
        elif b[0][0] == O:
            return -10
      
    if b[0][2] == b[1][1] and b[1][1] == b[2][0]:
      
        if b[0][2] == X:
            return 10
        elif b[0][2] == O:
            return -10
      
    # Else if none of them have won then return 0
    return 0
  
  
def isMovesLeft(board): 
  
    for i in range(3):
        for j in range(3):
            if (board[i][j] == EMPTY):
                return True 
    return False

# This is the minmax function. It considers all 
# the possible ways the game can go and returns 
# the value of the board 
def minmax(board, depth, isMax): 
    score = evaluate(board)
  
    print("score", score)
    # If Maximizer has won the game return his/her 
    # evaluated score 
    if (score == 10): 
        return score
  
    # If Minimizer has won the game return his/her 
    # evaluated score 
    if (score == -10):
        return score
  
    # If there are no more moves and no winner then 
    # it is a tie 
    if (isMovesLeft(board) == False):
        return 0
  
    # If this maximizer's move 
    if (isMax):     
        best = -1000 
        print(best)
        # Traverse all cells 
        for i in range(3):         
            for j in range(3):
               
                # Check if cell is empty 
                if (board[i][j]==EMPTY):
                  
                    # Make the move 
                    board[i][j] = player 
  
                    # Call minmax recursively and choose 
                    # the maximum value 
                    best = max(best, minmax(board, depth + 1, not isMax))
                    # Undo the move 
                    board[i][j] = EMPTY
        return best
  
    # If this minimizer's move 
    else:
        best = 1000 
  
        # Traverse all cells 
        for i in range(3):         
            for j in range(3):
               
                # Check if cell is empty 
                if (board[i][j] == EMPTY):
                  
                    # Make the move 
                    board[i][j] = opponent 
  
                    # Call minmax recursively and choose 
                    # the minimum value 
                    best = min(best, minmax(board, depth + 1, not isMax))
  
                    # Undo the move 
                    board[i][j] = EMPTY
        return best
  
# This will return the best possible move for the player 
def findBestMove(board): 
    bestVal = -1000 
    bestMove = (-1, -1) 
  
    # Traverse all cells, evaluate minmax function for 
    # all empty cells. And return the cell with optimal 
    # value. 
    for i in range(3):     
        for j in range(3):
          
            # Check if cell is empty 
            if (board[i][j] == EMPTY): 
              
                # Make the move 
                board[i][j] = player
  
                # compute evaluation function for this 
                # move. 
                moveVal = minmax(board, 0, False) 
  
                # Undo the move 
                board[i][j] = EMPTY 
  
                # If the value of the current move is 
                # more than the best value, then update 
                # best/ 
                if (moveVal > bestVal):                
                    bestMove = (i, j)
                    bestVal = moveVal
  
    print("The value of the best Move is:", bestVal)
    print()
    return bestMove

# Driver code
if __name__ == "__main__":
  
    board = [[X, O, X],
             [O, X, EMPTY],
             [O, X, O]]
    print(board)
    #value = evaluate(board)
    #print("The value of this board is", value)
    
    
    bestMove = findBestMove(board) 
    print(bestMove)
    #print("The Optimal Move is:") 
    #print("ROW:", bestMove[0], " COL:", bestMove[1])
    