import sys
import signal


def coord_to_idx(coord):
	"""Converts a grid coordinate to it's equivalent zero-based numerical index. E.g. A7 becomes 6."""
	rows = 'ABCDEFGHI'
	return (9 * rows.index(coord[0])) + (int(coord[1]) - 1)


def idx_to_coord(idx):
	"""Converts a grid index to it's equivalent alphanumeric coordinate. E.g. 6 becomes A7."""
	rows = 'ABCDEFGHI'
	col = (idx + 1) % 9
	col = 9 if col == 0 else col
	return rows[int(idx / 9)] + str(col)


def cross(vector_a, vector_b):
	"""Cross product of two vectors A and B, concatenating strings."""
	return [a+b for a in vector_a for b in vector_b]


def sudoku_elements():
	"""

	Generates a list of all co-ordinates for a 9x9 Sudoku board (labelled A1 tto I9), as well as a dictionary with all
	of the peer co-ordinates for every square on the board. Diagrams illustrating the grid labelling as well as the
	peer definitions are shown below.

	Sudoku Grid

	 A1 A2 A3 | A4 A5 A6| A7 A8 A9
	 B1 B2 B3 | B4 B5 B6| B7 B8 B9
	 C1 C2 C3 | C4 C5 C6| C7 C8 C9
	 ---------+---------+---------
	 D1 D2 D3 | D4 D5 D6| D7 D8 D9
	 E1 E2 E3 | E4 E5 E6| E7 E8 E9
	 F1 F2 F3 | F4 F5 F6| F7 F8 F9
	 ---------+---------+---------
	 G1 G2 G3 | G4 G5 G6| G7 G8 G9
	 H1 H2 H3 | H4 H5 H6| H7 H8 H9
	 I1 I2 I3 | I4 I5 I6| I7 I8 I9

	Units and peers for C2
	3 units for C2, each with 9 items => 27 indices.
	Peers are a unique set of other positions, except for itself.
	Both groups are vital to obtaining the solution

	     A2   |         |                    |         |            A1 A2 A3|         |
	     B2   |         |                    |         |            B1 B2 B3|         |
	     C2   |         |            C1 C2 C3| C4 C5 C6| C7 C8 C9   C1 C2 C3|         |
	 ---------+---------+---------  ---------+---------+---------  ---------+---------+---------
	     D2   |         |                    |         |                    |         |
	     E2   |         |                    |         |                    |         |
	     F2   |         |                    |         |                    |         |
	 ---------+---------+---------  ---------+---------+---------  ---------+---------+---------
	     G2   |         |                    |         |                    |         |
	     H2   |         |                    |         |                    |         |
	     I2   |         |                    |         |                    |         |

	Returns:
		tuple: First element is a list of all the possible coordinates on a sudoku board (81 elements from A1 to I9).
		Second element is a dictionary with all of the possible peers.
		Third element is a dictionary of all units for each position on the grid.
		Fourth element is a list of all units on the grid, useful for validation.
	"""
	all_rows = 'ABCDEFGHI'
	all_cols = '123456789'
	coords = cross(all_rows, all_cols)  # Flat list of all possible squares

	# Flat list of all possible units
	all_units = [cross(row, all_cols) for row in all_rows] + \
				[cross(all_rows, col) for col in all_cols] + \
				[cross(rid, cid) for rid in ['ABC', 'DEF', 'GHI'] for cid in ['123', '456', '789']]  # Squares

	# Indexed dictionary of units for each square (list of lists)
	# Each position will have three units, each a list of 9: row, column and square
	units = {pos: [unit for unit in all_units if pos in unit] for pos in coords}

	# Indexed dictionary of peers for each square (set)
	# Peers are the unique set of possible positions except itself
	peers = {pos: set(sum(units[pos], [])) - {pos} for pos in coords}

	return coords, peers, units, all_units


def display_sudoku_grid(grid, coords=False):
	"""
	Displays a 9x9 soduku grid in a nicely formatted way.

	Args:
		grid (str|dict|list): A string representing a Sudoku grid. Valid characters are digits from 1-9 and empty squares are
			specified by 0 or . only. Any other characters are ignored. A `ValueError` will be raised if the input does
			not specify exactly 81 valid grid positions.
			Can accept a dictionary where each key is the position on the board from A1 to I9.
			Can accept a list of strings or integers with empty squares represented by 0.
		coords (bool): Optionally prints the coordinate labels.
	Returns:
		str: Formatted depiction of a 9x9 soduku grid.
	"""
	if grid is None or grid is False:
		return None

	all_rows = 'ABCDEFGHI'
	all_cols = '123456789'
	null_chars = '0.'

	if type(grid) == str:
		grid = parse_sudoku_puzzle(grid)
	elif type(grid) == list:
		grid = parse_sudoku_puzzle(''.join([str(el) for el in grid]))

	width = max([3, max([len(grid[pos]) for pos in grid]) + 1])

	display = ''
	if coords:
		display += '   ' + ''.join([all_cols[i].center(width) for i in range(3)]) + '|'
		display += ''.join([all_cols[i].center(width) for i in range(3, 6)]) + '|'
		display += ''.join([all_cols[i].center(width) for i in range(6, 9)]) + '\n   '
		display += '--' + ''.join(['-' for x in range(width * 9)]) + '\n'

	row_counter = 0
	col_counter = 0
	for row in all_rows:
		row_counter += 1
		for col in all_cols:
			col_counter += 1
			if grid[row + col] in null_chars:
				grid[row + col] = '.'

			display += ('%s' % grid[row + col]).center(width)
			if col_counter % 3 == 0 and col_counter % 9 != 0:
				display += '|'
			if col_counter % 9 == 0:
				display += '\n'
		if row_counter % 3 == 0 and row_counter != 9:
			if coords:
				display += '  |'
			display += '+'.join([''.join(['-' for x in range(width * 3)]) for y in range(3)]) + '\n'
	return display


def parse_sudoku_puzzle(puzzle):
	"""
	Parses the input Sudoku puzzle and returns any non-empty grid values. Raises a ValueError if the input is bad.

	Args:
		puzzle (str|dict): A string representing a Sudoku grid. Valid characters are digits from 1-9 and empty squares are
			specified by 0 or . only. Any other characters are ignored. A `ValueError` will be raised if the input does
			not specify exactly 81 valid grid positions. Alternatively, can pass a dictionary where each key is a board
			position from A1 to I9.

	Returns:
		dict: Dictionary with each key representing a position on the board between A1 and I9 and each value a digit
			between 1 and 9, represented as a `str`. Empty cells are described as `.`.
	"""

	digits = '123456789'
	nulls = '.0'

	if type(puzzle) == str:
		# Serialise the input into a string, let the position define the grid location and .0 can be empty positions
		# Ignore any characters that aren't digit input or nulls
		flat_puzzle = ['.' if char in nulls else char for char in puzzle if char in digits + nulls]

		# flat_puzzle = [char for char in in_puzzle if char in digits or char in '0.']
		if len(flat_puzzle) != 81:
			raise ValueError('Input puzzle has %s grid positions specified, must be 81. Specify a position using any '
							 'digit from 1-9 and 0 or . for empty positions.' % len(flat_puzzle))

		coords, peers, units, all_units = sudoku_elements()
		return dict(zip(coords, flat_puzzle))
	elif type(puzzle) == dict:  # Set all non-digit values to '.'
		for k in puzzle:
			if type(puzzle[k]) == int:
				puzzle[k] = str(puzzle[k])
			if puzzle[k] not in digits:
				puzzle[k] = '.'
		return puzzle
	else:
		raise TypeError('Input puzzle is neither a dict nor a str, cannot parse.')


def validate_sudoku(puzzle):
	"""
	Checks if a completed Sudoku puzzle has a valid solution.

	Args:
		puzzle (dict): A dictionary where each key is the position on the board from A1 to I9.

	Returns:
		bool: Indicator if the Sudoku puzzle is correctly solved.
	"""
	if puzzle is False or puzzle is None:
		return False

	coords, peers, units, all_units = sudoku_elements()
	full = [str(x) for x in range(1, 10)]  # Full set, 1-9 as strings

	# Checks if all units contain a full set
	return all([sorted([puzzle[cell] for cell in unit]) == full for unit in all_units])


def validate_input(puzzle, get_contradictions=False):
	"""Checks if the input puzzle has any contradictions in any given units."""
	if puzzle is False:
		return False

	puzzle = parse_sudoku_puzzle(puzzle)
	coords, peers, units, all_units = sudoku_elements()

	def count_num(vals):
		count = {}
		for v in vals:
			count[v] = count.get(v, 0) + 1
		return count

	# Checks if there are any units that have more than one non-zero digit present
	contradictions = []
	for i, unit in enumerate(all_units):
		# If requested, create a list of all the contradicting cell locations and their value.
		if get_contradictions:
			# Get all the cells that have contradictory values within a unit
			bad_digits = [k for k, v in (count_num([puzzle[pos] for pos in unit])).items() if k != '.' and v > 1]
			bad_digits = [cnt for cnt in bad_digits if len(cnt) > 0]
			if len(bad_digits) > 0:
				# Build a list of dictionaries of each contradictory set
				for digit in bad_digits:
					contradictions.append({k: v for k, v in puzzle.items() if k in unit and v == digit})
		else:
			if len([k for k, v in (count_num([puzzle[pos] for pos in unit])).items() if k != '.' and v > 1]) > 0:
				return False

	if get_contradictions and len(contradictions) > 0:
		return contradictions

	return True


def solve(puzzle, timeout=1):
	"""
	Handler for solving puzzles with a timeout wrapper, useful for preventing runaway loops when the input puzzle is
	invalid.
	"""
	def timeout_handler(signum, frame):
		raise TimeoutError('Puzzle unsolvable, stuck in decision loop.')

	if not sys.platform.startswith('linux'):
		return solve_puzzle(puzzle)

	# Set an alarm to the timeout
	signal.signal(signal.SIGALRM, timeout_handler)
	signal.alarm(timeout)

	try:
		solution = solve_puzzle(puzzle)
		signal.alarm(0)  # Disables the alarm so it doesn't affect other processes
		return solution
	except TimeoutError:
		return False


def solve_puzzle(puzzle):
	"""
	Solves a Sudoku puzzle.

	Args:
		puzzle (str|dict): A string representing a Sudoku grid. Valid characters are digits from 1-9 and empty squares are
			specified by 0 or . only. Any other characters are ignored. A `ValueError` will be raised if the input does
			not specify exactly 81 valid grid positions. Alternatively accepts a dictionary where each key is the position
			on the board from A1 to I9.

	Returns:
		dict: Solved Sudoku grid represented by a dicionary where each key is a grid position from A1 to I9.
	"""

	if validate_input(puzzle) is not True:
		print('Board input is invalid and has contradictions, will not attempt to solve.')
		return False

	coords, peers, units, all_units = sudoku_elements()
	digits = '123456789'  # Strings are immutable, so they are easier to use here than lists

	input_grid = parse_sudoku_puzzle(puzzle)  # Parse puzzle
	input_grid = {k: v for k, v in input_grid.items() if v != '.'}  # Filter out empty keys
	output_grid = {cell: digits for cell in coords}  # To start with, assume all digits 1-9 are possible

	def set_value(values, pos, val):
		"""
		Eliminate all the other values except the entered val from the input position.
		The elimination function will propagate to peers and will do checks based on unit
		Return values, except return False if a contradiction is detected.
		"""
		remaining_values = values[pos].replace(val, '')
		answers = []
		for v in remaining_values:
			answers.append(eliminate(values, pos, v))

		if all(answers):
			return values
		else:
			return None

	def eliminate(values, pos, val):
		"""
		Eliminate val from values[pos] and propogate the elimination when possible.
		Based on two rules:

		* Any values we know immediately remove the possibility from existing in any peer.
		* When there is only possible location left in a unit, it must have the remaining value.
		"""

		if val not in values[pos]:
			return values  # Already eliminated this value

		values[pos] = values[pos].replace(val, '')  # Remove value from the list

		if len(values[pos]) == 0:
			return None  # Contradiction - can't remove all the possibilities
		elif len(values[pos]) == 1:
			new_val = values[pos]  # New candidate for elimination from all peers

			# Loop over peers and eliminate this value from all of them
			for peer in peers[pos]:
				values = eliminate(values, peer, new_val)
				if values is None:  # Exit as soon as a contradiction is found
					return None

		# Check for the number of remaining places the eliminated value can occupy in each unit
		for unit in units[pos]:
			possible_places = [cell for cell in unit if val in values[cell]]
			if len(possible_places) == 0:  # Contradiction - can't have no possible locations left
				return None
			# If there is only only possible location for the eliminated digit, confirm that position
			elif len(possible_places) == 1 and len(values[possible_places[0]]) > 1:
				if not set_value(values, possible_places[0], val):
					return None  # Exit if the outcome is a contradiction

		return values

	# First pass, should never raise a contradiction
	# Will complete easy sudokus at this point
	for position, value in input_grid.items():
		set_value(output_grid, position, value)

	if validate_sudoku(output_grid):  # Finish if we're done
		return output_grid

	def guess_solution(values, depth=0):
		if values is None:
			return None  # Already failed

		if all(len(v) == 1 for k, v in values.items()):
			return values  # Solved the puzzle, can end

		# Gets the cell with the shortest length, i.e. the fewest options to try.
		# This gives the highest probability for propagating a solution correctly.
		# If there are two options, there is a 0.5 chance it is correct. If there are 5, only 0.2
		possible_values = [(len(v), k) for k, v in values.items() if len(v) > 1]
		if len(possible_values) == 0:  # Contradiction, invalid solution
			return None
		n, pos = (min([(len(v), k) for k, v in values.items() if len(v) > 1]))

		# Sort possible values for the position by the number of positions possible in peers.
		# Further increases the likelihood of making the write choice.
		# Adds ~0.01s to difficult puzzles but guarantees a fast solution for even the toughest of puzzles.
		def num_peer_possibilities(poss_val):
			return len([(cell, v) for cell, v in output_grid.items() if cell in peers[pos] and len(v) > 1 and poss_val in v])

		possible_values = ''.join(sorted(output_grid[pos], key=num_peer_possibilities))

		# Attempt all choices from our minimum choice positions.
		# It is important to run all possibilities, otherwise we hit a dead end.
		# We break as soon as it succeeds, so only one solution is found.

		# Update to the above - have now seen runaway loops that are caused by trying to solve invalid puzzles. This has
		# been mitigated by wrapping this function in a timeout handler.
		for val in possible_values:
			solution = guess_solution(set_value(values.copy(), pos, val), depth + 1)
			if solution is not None:  # Complete as soon as a valid solution is found
				return solution

	return guess_solution(output_grid)


def load_grids(f_name):
	"""Generator loading grids from a file with a format that has the grid after a title line starting with 'Grid'."""
	new_grid = ''
	with open(f_name, 'r') as f:
		for line in f:
			if line.startswith('Grid'):
				if new_grid != '':
					yield new_grid
					new_grid = ''
				continue
			new_grid += line
	yield new_grid


