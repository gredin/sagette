import math
import random
from timeit import default_timer as timer

import lexpy
import svgwrite
import bitarray

# EMPTY = 0
BLOCKED = 1
DOWN = 2
DOWN_RIGHT = 3
RIGHT = 4
RIGHT_DOWN = 5
DOWN_AND_RIGHT = 6
DOWN_AND_RIGHT_DOWN = 7
DOWN_RIGHT_AND_RIGHT = 8
DOWN_RIGHT_AND_RIGHT_DOWN = 9
ARROWS = [
    DOWN,
    DOWN_RIGHT,
    RIGHT,
    RIGHT_DOWN,
    DOWN_AND_RIGHT,
    DOWN_AND_RIGHT_DOWN,
    DOWN_RIGHT_AND_RIGHT,
    DOWN_RIGHT_AND_RIGHT_DOWN,
]

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
letter_to_index = {letter: index for index, letter in enumerate(letters)}

"""
état de la grille
[[], []...]
- A-Z
- bas, bas droite
  droite, droite bas
  bas ET droite
  bas ET droite bas
  bas droite ET droite
  bas droite ET droite bas

mots
- coord de la 1ère lettre, horizontal/vertical
- coord de l'indice

indices
- coord de la 2ème du mot

indices-mots
- coord de l'indice

n x m
i, j
min length

dico => index
- par nb de lettres
- par (lettre, position)
- n-grams

wave function collapse "n x n"
=> raisonner en termes de n-grams (n ≥ min length)
26 x 26 x 26 = 17576

4567.34³
"""


def initialize(n, m):
    grid = []
    for i in range(n + 1):
        grid.append([None] * (m + 1))

    grid_letter_bitarray = []
    for i in range(n + 1):
        line = []

        for j in range(m + 1):
            # instantiate each bitarray because they seem to be pointers
            bitarr = bitarray.bitarray(len(letters))
            bitarr.setall(0)

            line.append(bitarr)

        grid_letter_bitarray.append(line)

    clue_candidates = set()
    clue_candidates_grid = []  # [i][j] => [bool for RIGHT, bool for DOWN]
    for i in range(n + 1):
        clue_candidates_grid.append([[False, False]] * (m + 1))

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0:
                if j >= 2 and j % 2 == 0:
                    grid[i][j] = DOWN
                    clue_candidates.add((i, j))
                    clue_candidates_grid[i][j] = [False, True]
                    continue
                else:
                    grid[i][j] = BLOCKED
                    continue

            if j == 0:
                if i >= 2 and i % 2 == 0:
                    grid[i][j] = RIGHT
                    clue_candidates.add((i, j))
                    clue_candidates_grid[i][j] = [True, False]
                    continue
                else:
                    grid[i][j] = BLOCKED
                    continue

            if i == 1 and j == 1:
                grid[i][j] = BLOCKED
                continue

            if i == 1 and j % 2 == 1:
                grid[i][j] = BLOCKED
                clue_candidates.add((i, j))
                clue_candidates_grid[i][j] = [False, True]
                continue

            if j == 1 and i % 2 == 1:
                grid[i][j] = BLOCKED
                clue_candidates.add((i, j))
                clue_candidates_grid[i][j] = [True, False]
                continue

            grid[i][j] = None
            grid_letter_bitarray[i][j].setall(1)

    return grid, clue_candidates, clue_candidates_grid, grid_letter_bitarray


def draw_grid(filename, grid, clue_candidates, clue_candidates_grid, n, m):
    margin_x = 0
    margin_y = 0
    drawing = svgwrite.Drawing(
        f"{filename}.svg",
        profile="full",
        size=(800, 800),
        viewBox=f"0 0 {m + 1 + 2*margin_x} {n + 1 + 2*margin_y}",
    )

    drawing.add(
        drawing.rect((1, 1), (m + 1 + 2 * margin_x, n + 1 + 2 * margin_y), fill="white")
    )

    stroke_width = 0.1

    for i in range(1, n + 1):
        drawing.add(
            drawing.line(
                (margin_x + 1, margin_y + i),
                (margin_x + m + 1, margin_y + i),
                stroke_width=stroke_width,
                stroke=svgwrite.rgb(0, 0, 0, "%"),
            )
        )

    for j in range(1, m + 1):
        drawing.add(
            drawing.line(
                (margin_x + j, margin_y + 1),
                (margin_x + j, margin_y + n + 1),
                stroke_width=stroke_width,
                stroke=svgwrite.rgb(0, 0, 0, "%"),
            )
        )

    # grid[2][2] = "A"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            value = grid[i][j]

            if value is None:
                continue

            if value == BLOCKED or value == DOWN or value == RIGHT:
                x = margin_x + j
                y = margin_y + i
                width, height = 1, 1
                drawing.add(
                    drawing.rect(
                        insert=(x + stroke_width / 2, y + stroke_width / 2),
                        size=(width - stroke_width, height - stroke_width),
                        fill="grey",
                    )
                )

                continue

            size = 1
            x = margin_x + j + 0.5  # TODO?
            y = margin_y + i + 0.85  # TODO?
            drawing.add(
                drawing.text(
                    value,
                    insert=(x, y),
                    dominant_baseline="middle",
                    text_anchor="middle",
                    font_size=f"{size}",
                    fill="black",
                )
            )

            """
            if value == DOWN_RIGHT:
                drawing.add(
                    drawing.line(
                        (margin_x + i + 0.5, margin_y + 1 - 0.1),
                        (margin_x + i + 0.5, margin_y + 1 + 0.2),
                        stroke_width=0.05,
                        stroke="blue",
                    )
                )

                drawing.add(
                    drawing.line(
                        (margin_x + i + 0.5, margin_y + 1 + 0.2),
                        (margin_x + i + 0.7, margin_y + 1 + 0.2),
                        stroke_width=0.05,
                        stroke="blue",
                    )
                )
            """

    for i, j in clue_candidates:
        x = margin_x + j
        y = margin_y + i
        width, height = 1, 1
        drawing.add(
            drawing.rect(
                insert=(x + stroke_width / 2, y + stroke_width / 2),
                size=(width - stroke_width, height - stroke_width),
                fill="pink",
            )
        )

    # drawing.add(drawing.text('Test', insert=(0, 0.2), fill='red'))

    drawing.save()


def add_word(
    grid,
    clue_candidates,
    clue_candidates_grid,
    grid_letter_bitarray,
    n,
    m,
    min_len,
    word,
    clue_line,
    clue_column,
    clue_orientation,
):
    grid[clue_line][clue_column] = clue_orientation
    grid_letter_bitarray[clue_line][clue_column].setall(0)

    word_len = len(word)

    if clue_orientation == RIGHT:
        j = clue_column + 1
        for letter in word:
            grid[clue_line][j] = letter

            grid_letter_bitarray[clue_line][j].setall(0)
            letter_index = letter_to_index[letter]
            grid_letter_bitarray[clue_line][j][letter_index] = 1

            j += 1

        clue_candidates_grid[clue_line][clue_column][0] = False

        next_clue_column = clue_column + word_len + 1 + min_len
        if next_clue_column <= m:
            clue_candidates.add((clue_line, next_clue_column))
            clue_candidates_grid[clue_line][next_clue_column][0] = True

            grid_letter_bitarray[clue_line][next_clue_column].setall(0)
    elif clue_orientation == DOWN:
        i = clue_line + 1
        for letter in word:
            grid[i][clue_column] = letter

            grid_letter_bitarray[i][clue_column].setall(0)
            letter_index = letter_to_index[letter]
            grid_letter_bitarray[i][clue_column][letter_index] = 1

            i += 1

        clue_candidates_grid[clue_line][clue_column][1] = False

        next_clue_line = clue_line + word_len + 1 + min_len
        if next_clue_line <= n:
            clue_candidates.add((next_clue_line, clue_column))
            clue_candidates_grid[next_clue_line][clue_column][1] = True

            grid_letter_bitarray[next_clue_line][clue_column].setall(0)

    if clue_candidates_grid[clue_line][clue_column] == [False, False]:
        clue_candidates.remove((clue_line, clue_column))

    return grid, clue_candidates, clue_candidates_grid, grid_letter_bitarray


if __name__ == "__main__":
    n, m = 10, 10
    min_len = 3

    grid, clue_candidates, clue_candidates_grid, grid_letter_bitarray = initialize(n, m)

    lengths = set()
    words = []
    with open("dev_lexicon.txt") as f:  # open("officiel-du-scrabble-8.txt") as f:
        for line in f:
            w = line.strip()

            length = len(w)

            if length < min_len:
                continue

            lengths.add(length)
            words.append(w)

    bitarrays = []

    bitarray_size = len(words)
    for _ in range(100):
        a = bitarray.bitarray(bitarray_size)
        a.setall(0)
        for i in random.sample(range(bitarray_size), round(bitarray_size / 10)):
            a[i] = 1
        bitarrays.append(a)

    start = timer()
    for _ in range(100):
        a1 = random.choice(bitarrays)
        a2 = random.choice(bitarrays)

        intersection_count = (a1 & a2).count()

        print(intersection_count)

    print(timer() - start)
    #exit()

    length_to_words = {length: set() for length in lengths}
    for w in words:
        length_to_words[len(w)].add(w)

    position_to_words = {}
    for w in words:
        for i, letter in enumerate(w):
            position_to_words.setdefault((i + 1, letter), set()).add(w)
    for (position, letter), words in position_to_words.items():
        position_to_words[(position, letter)] = sorted(words, key=len, reverse=True)

    dawg = lexpy.DAWG()
    dawg.add_all(words)

    print(dawg.search("C?L?R"))
    print(len(dawg.search("C?L?R")))

    print(len(length_to_words[10]))

    word_candidate_len = m
    number_of_candidates = 50
    word_candidates = random.sample(
        list(length_to_words[word_candidate_len]), number_of_candidates
    )

    best_candidate = None
    best_score = 0
    for w in word_candidates:
        # score = math.inf
        score = 1
        for i in range(1, word_candidate_len, 2):
            # score = min(score, len(position_to_words.get((2, w[i]), [])))
            score *= len(position_to_words.get((2, w[i]), []))
        if score > best_score:
            best_score = score
            best_candidate = w

    print(best_candidate, best_score)

    print("----- ", grid_letter_bitarray[4][1])

    draw_grid("grid_1", grid, clue_candidates, clue_candidates_grid, n, m)

    grid, clue_candidates, clue_candidates_grid, grid_letter_bitarray = add_word(
        grid,
        clue_candidates,
        clue_candidates_grid,
        grid_letter_bitarray,
        n,
        m,
        min_len,
        best_candidate,
        2,
        0,
        RIGHT,
    )

    print("----- ", grid_letter_bitarray[4][1])

    draw_grid("grid_2", grid, clue_candidates, clue_candidates_grid, n, m)

    word_candidate_len = n
    number_of_candidates = 10

    # print(dawg.search(f"?{grid[2][2]}" + (n-3)*"?" + "R"))
    # print(f"?{grid[2][2]}" + (n-3)*"?" + "R")
    # print(position_to_words[(2, grid[2][2])])
    # exit()

    word_candidates = position_to_words[(2, grid[2][2])][:number_of_candidates]

    best_candidate = None
    best_score = 0
    for w in word_candidates:
        # score = math.inf
        score = 1
        for i in range(1, len(w), 2):
            # score = min(score, len(position_to_words.get((2, w[i]), [])))
            score *= len(position_to_words.get((2, w[i]), []))

        if score > best_score:
            best_score = score
            best_candidate = w

    print(best_candidate, best_score)

    print("----- ", grid_letter_bitarray[4][1])

    grid, clue_candidates, clue_candidates_grid, grid_letter_bitarray = add_word(
        grid,
        clue_candidates,
        clue_candidates_grid,
        grid_letter_bitarray,
        n,
        m,
        min_len,
        best_candidate,
        0,
        2,
        DOWN,
    )

    print("----- ", grid_letter_bitarray[4][1])

    draw_grid("grid_3", grid, clue_candidates, clue_candidates_grid, n, m)

    # 2^5 = 32
    # 4 bytes
    # 0b1111111111111111111111111

    for (clue_i, clue_j) in clue_candidates:
        is_right, is_down = clue_candidates_grid[clue_i][clue_j]

        if is_right:
            letter_constraints = []
            j = clue_j + 1
            while j <= m:
                # TODO more efficient check that this is a clue
                print(j, clue_candidates_grid[clue_i][j])
                if clue_candidates_grid[clue_i][j][0] or clue_candidates_grid[clue_i][j][1]:
                    break

                letter_constraints.append(grid_letter_bitarray[clue_i][j])

                print(clue_i, j, grid_letter_bitarray[clue_i][j])

                j += 1

            print(clue_i, clue_j, "right", letter_constraints)

            exit()

        if is_down:
            letter_constraints = []
            i = clue_i + 1
            while i <= n:
                # TODO more efficient check that this is a clue
                if clue_candidates_grid[i][clue_j][0] or clue_candidates_grid[i][clue_j][1]:
                    break

                letter_constraints.append(grid_letter_bitarray[i][clue_j])

                i += 1

            print(clue_i, clue_j, "down", letter_constraints)

            exit()

    exit()

    n, m = 10, 10
    min_word_length = 3

    grid = initialize(n, m)

    print(grid)

    draw_grid(grid)
