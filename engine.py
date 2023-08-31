# Adapted from https://www.geeksforgeeks.org/finding-optimal-move-in-tic-tac-toe-using-minimax-algorithm-in-game-theory/

# This function returns true if there are moves
# remaining on the board. It returns false if
# there are no moves left to play.
from constants import *

def are_moves_left(board):

	for i in range(3):
		for j in range(3):
			if (board[i][j] == EMPTY):
				return True
	return False

# This is the evaluation function as discussed
# in the previous article ( http://goo.gl/sJgv68 )

def get_state(flat_board):
	board = [flat_board[0:3], flat_board[3:6], flat_board[6:9]]
	ev, win_type, win_info = evaluate(board, X)
	xs = 0
	os = 0
 
	for item in flat_board:
		if item == X:
			xs += 1
		elif item == O:
			os += 1
   
	turn = O if xs > os else X
 
	if ev == 10:
		state = GAME_X_WON
		turn = X
	elif ev == -10:
		state = GAME_O_WON
		turn = O
	elif are_moves_left(board):
		state = GAME_ACTIVE
	else:
		state = GAME_DRAWN
 
	return state, turn, win_type, win_info

def evaluate(b, player):
	opponent = -player
	# Checking for Rows for X or O victory.
	for row in range(3):	
		if (b[row][0] == b[row][1] and b[row][1] == b[row][2]):		
			if (b[row][0] == player):
				return 10, "row", row
			elif (b[row][0] == opponent):
				return -10, "row", row

	# Checking for Columns for X or O victory.
	for col in range(3):
	
		if (b[0][col] == b[1][col] and b[1][col] == b[2][col]):
		
			if (b[0][col] == player):
				return 10, "col", col
			elif (b[0][col] == opponent):
				return -10, "col", col

	# Checking for Diagonals for X or O victory.
	if (b[0][0] == b[1][1] and b[1][1] == b[2][2]):
	
		if (b[0][0] == player):
			return 10, "diag", -1
		elif (b[0][0] == opponent):
			return -10, "diag", -1

	if (b[0][2] == b[1][1] and b[1][1] == b[2][0]):
	
		if (b[0][2] == player):
			return 10, "diag", 1
		elif (b[0][2] == opponent):
			return -10, "diag", 1

	# Else if none of them have won then return 0
	return 0, None, 0

# This is the minimax function. It considers all
# the possible ways the game can go and returns
# the value of the board
def minimax(board, player, depth, isMax):
	opponent = -player
	score, _, _ = evaluate(board, player)

	# If Maximizer has won the game return his/her
	# evaluated score
	if (score == 10):
		return score-depth

	# If Minimizer has won the game return his/her
	# evaluated score
	if (score == -10):
		return score+depth

	# If there are no more moves and no winner then
	# it is a tie
	if (are_moves_left(board) == False):
		return 0

	# If this maximizer's move
	if (isMax):	
		best = -1000

		# Traverse all cells
		for i in range(3):		
			for j in range(3):
			
				# Check if cell is empty
				if (board[i][j]==EMPTY):
				
					# Make the move
					board[i][j] = player

					# Call minimax recursively and choose
					# the maximum value
					best = max(best, minimax(board, player, depth + 1, not isMax) )

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

					# Call minimax recursively and choose
					# the minimum value
					best = min(best, minimax(board, player, depth + 1, not isMax))

					# Undo the move
					board[i][j] = EMPTY
		return best

# This will return the best possible move for the player
def find_best_move(flat_board, player):
	board = [flat_board[0:3], flat_board[3:6], flat_board[6:9]]
 
	bestVal = -1000
	bestMove = (-1, -1)

	# Traverse all cells, evaluate minimax function for
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
				moveVal = minimax(board, player, 0, False)

				# Undo the move
				board[i][j] = EMPTY

				# If the value of the current move is
				# more than the best value, then update
				# best/
				if (moveVal > bestVal):				
					bestMove = (i, j)
					bestVal = moveVal

	return bestMove[0] * 3 + bestMove[1]

