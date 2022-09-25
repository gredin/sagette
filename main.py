import math
import random
import sys
import time
from timeit import default_timer as timer

import bitarray
import cairosvg
import lexpy
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import svgwrite

plt.ion()

# EMPTY = 0
BLOCKED = 0
CLUE = 1
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

letters_list = [l for l in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
letters = set(letters_list)
letter_to_index = {letter: index for index, letter in enumerate(letters_list)}

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


class Solver:
    image_width = 800
    image_height = 800
    dpi = 96  # TODO

    # TODO delete
    image_i = 0

    def __init__(self, n, m, min_len, custom_clues, lexicon):
        self.n = n
        self.m = m
        self.min_len = min_len
        self.image = (
            None  # np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        )

        (
            filtered_lexicon,
            length_to_words,
            position_to_words,
            dawg,
        ) = self.init_from_lexicon(lexicon)
        self.lexicon = filtered_lexicon
        self.length_to_words = length_to_words
        self.position_to_words = position_to_words
        self.dawg = dawg

        grid = []
        for i in range(n + 1):
            grid.append([None] * (m + 1))

        # default clues
        for i in range(n + 1):
            for j in range(m + 1):
                if i == 0:
                    if j >= 2 and j % 2 == 0:
                        grid[i][j] = CLUE
                        # clue_candidates.add((i, j))
                        # clue_candidates_grid[i][j] = [False, True]
                        continue
                    else:
                        grid[i][j] = BLOCKED
                        continue

                if j == 0:
                    if i >= 2 and i % 2 == 0:
                        grid[i][j] = CLUE
                        # clue_candidates.add((i, j))
                        # clue_candidates_grid[i][j] = [True, False]
                        continue
                    else:
                        grid[i][j] = BLOCKED
                        continue

                if i == 1 and j == 1:
                    grid[i][j] = BLOCKED
                    continue

                if i == 1 and j % 2 == 1:
                    grid[i][j] = CLUE
                    # clue_candidates.add((i, j))
                    # clue_candidates_grid[i][j] = [False, True]
                    continue

                if j == 1 and i % 2 == 1:
                    grid[i][j] = CLUE
                    # clue_candidates.add((i, j))
                    # clue_candidates_grid[i][j] = [True, False]
                    continue

                # grid[i][j] = None
                # grid_letter_bitarray[i][j].setall(1)

        # custom clues

        for i, j in custom_clues:
            grid[i][j] = CLUE

        # clue candidates

        clue_candidates = set()
        # clue_candidates_grid = []  # [i][j] => [bool for RIGHT, bool for DOWN]
        # for i in range(n + 1):
        #    clue_candidates_grid.append([[False, False]] * (m + 1))

        for i_clue in range(n + 1):
            for j_clue in range(m + 1):
                if grid[i_clue][j_clue] != CLUE:
                    continue

                word_len = 0
                j = j_clue + 1
                while j <= m:
                    if grid[i_clue][j] is not None:
                        break

                    word_len += 1
                    j += 1
                if word_len >= 2:
                    if word_len < min_len:
                        raise ValueError(
                            f"there is a word of length {word_len} for clue ({i_clue}, {j_clue}) but the minimum length is {min_len}"
                        )

                    clue_candidates.add((i_clue, j_clue, RIGHT, word_len))

                word_len = 0
                i = i_clue + 1
                while i <= n:
                    if grid[i][j_clue] is not None:
                        break

                    word_len += 1
                    i += 1
                if word_len >= 2:
                    if word_len < min_len:
                        raise ValueError(
                            f"there is a word of length {word_len} for clue ({i_clue}, {j_clue}) but the minimum length is {min_len}"
                        )

                    clue_candidates.add((i_clue, j_clue, DOWN, word_len))

        """
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
        """

        cell_to_clues = {}
        for i_clue, j_clue, orientation, word_len in clue_candidates:
            if orientation == RIGHT:
                for j in range(j_clue + 1, j_clue + 1 + word_len):
                    cell_to_clues.setdefault((i_clue, j), set()).add(
                        (i_clue, j_clue, orientation, word_len)
                    )

            if orientation == DOWN:
                for i in range(i_clue + 1, i_clue + 1 + word_len):
                    cell_to_clues.setdefault((i, j_clue), set()).add(
                        (i_clue, j_clue, orientation, word_len)
                    )

        clue_to_clue = {}
        for i in range(n + 1):
            for j in range(m + 1):
                clues = cell_to_clues.get((i, j), set())

                for i_clue, j_clue, orientation, word_len in clues:
                    for i_clue2, j_clue2, orientation2, word_len2 in clues:
                        if i_clue == i_clue2 and j_clue == j_clue2:
                            continue

                        clue_to_clue.setdefault(
                            (i_clue, j_clue, orientation, word_len), set()
                        ).add((i_clue2, j_clue2, orientation2, word_len2))

        grid_letter_bitarray = []
        for i in range(n + 1):
            line = []

            for j in range(m + 1):
                # instantiate each bitarray because they seem to be pointers
                bitarr = bitarray.bitarray(len(letters))
                bitarr.setall(0)

                line.append(bitarr)

            grid_letter_bitarray.append(line)

        for i in range(n + 1):
            for j in range(m + 1):
                if grid[i][j] is None:
                    grid_letter_bitarray[i][j].setall(1)

        self.initial_grid = grid
        self.initial_clue_candidates = clue_candidates
        self.initial_grid_letter_bitarray = grid_letter_bitarray
        self.clue_to_clue = clue_to_clue

    def init_from_lexicon(self, lexicon):
        filtered_lexicon = []

        lengths = set()
        for word in lexicon:
            length = len(word)
            if length < min_len:
                continue

            filtered_lexicon.append(word)
            lengths.add(length)

        """
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

            #print(intersection_count)

        #print(timer() - start)
        #exit()
        """

        length_to_words = {length: set() for length in lengths}
        for w in filtered_lexicon:
            length_to_words[len(w)].add(w)

        position_to_words = {}
        for w in filtered_lexicon:
            for i, letter in enumerate(w):
                position_to_words.setdefault((i + 1, letter), set()).add(w)

        for (position, letter), words in position_to_words.items():
            position_to_words[(position, letter)] = sorted(words, key=len, reverse=True)

        dawg = lexpy.DAWG()
        dawg.add_all(filtered_lexicon)

        return filtered_lexicon, length_to_words, position_to_words, dawg

    def add_word(
        self,
        grid,
        clue_candidates,
        grid_letter_bitarray,
        word,
        clue_line,
        clue_column,
        clue_orientation,
    ):
        # grid[clue_line][clue_column] = clue_orientation
        # grid_letter_bitarray[clue_line][clue_column].setall(0)
        # word_len = len(word)

        if clue_orientation == RIGHT:
            j = clue_column + 1
            for letter in word:
                grid[clue_line][j] = letter

                grid_letter_bitarray[clue_line][j].setall(0)
                letter_index = letter_to_index[letter]
                grid_letter_bitarray[clue_line][j][letter_index] = 1

                j += 1
        elif clue_orientation == DOWN:
            i = clue_line + 1
            for letter in word:
                grid[i][clue_column] = letter

                grid_letter_bitarray[i][clue_column].setall(0)
                letter_index = letter_to_index[letter]
                grid_letter_bitarray[i][clue_column][letter_index] = 1

                i += 1

        word_len = len(word)
        print("remove", (clue_line, clue_column, clue_orientation, word_len))
        clue_candidates.remove((clue_line, clue_column, clue_orientation, word_len))

        grid_letter_bitarray = self.recompute_grid_letter_bitarray(
            grid, clue_candidates, grid_letter_bitarray
        )

        for clue_line2, clue_column2, clue_orientation2, word_len2 in self.clue_to_clue[
            (clue_line, clue_column, clue_orientation, word_len)
        ]:
            if (
                clue_line2,
                clue_column2,
                clue_orientation2,
                word_len2,
            ) not in clue_candidates:
                continue

            if clue_orientation2 == RIGHT:
                is_complete = True
                for j in range(clue_column2 + 1, clue_column2 + 1 + word_len2):
                    if grid_letter_bitarray[clue_line2][j].count() > 1:
                        is_complete = False
                        break
                if is_complete:
                    clue_candidates.remove(
                        (clue_line2, clue_column2, clue_orientation2, word_len2)
                    )

            elif clue_orientation2 == DOWN:
                is_complete = True
                for i in range(clue_line2 + 1, clue_line2 + 1 + word_len2):
                    if grid_letter_bitarray[i][clue_column2].count() > 1:
                        is_complete = False
                        break
                if is_complete:
                    clue_candidates.remove(
                        (clue_line2, clue_column2, clue_orientation2, word_len2)
                    )

        return grid, clue_candidates, grid_letter_bitarray

    def recompute_grid_letter_bitarray(
        self, grid, clue_candidates, grid_letter_bitarray
    ):
        for i_clue, j_clue, orientation, word_len in clue_candidates:
            if orientation == RIGHT:
                compatible_words = self.dawg.search("." * word_len)

                bitarrs = []
                for _ in range(word_len):
                    bitarr = bitarray.bitarray(len(letters))
                    bitarr.setall(0)
                    bitarrs.append(bitarr)

                for w in compatible_words:
                    for k in range(word_len):
                        letter = w[k]
                        letter_index = letter_to_index[letter]

                        bitarrs[k][letter_index] = 1

                for k in range(word_len):
                    grid_letter_bitarray[i_clue][j_clue + 1 + k] &= bitarrs[k]

            if orientation == DOWN:
                compatible_words = self.dawg.search("." * word_len)

                bitarrs = []
                for _ in range(word_len):
                    bitarr = bitarray.bitarray(len(letters))
                    bitarr.setall(0)
                    bitarrs.append(bitarr)

                for w in compatible_words:
                    for k in range(word_len):
                        letter = w[k]
                        letter_index = letter_to_index[letter]

                        bitarrs[k][letter_index] = 1

                for k in range(word_len):
                    grid_letter_bitarray[i_clue + 1 + k][j_clue] &= bitarrs[k]

        return grid_letter_bitarray

    def choose_clue_candidate(self, grid, clue_candidates, grid_letter_bitarray):
        return list(clue_candidates)[0]

    def choose_compatible_words(
        self,
        grid,
        clue_candidates,
        grid_letter_bitarray,
        clue_i,
        clue_j,
        clue_orientation,
        word_len,
    ):
        if clue_orientation == RIGHT:
            search_pattern = ""
            for k in range(word_len):
                grid_value = grid[clue_i][clue_j + 1 + k]

                if grid_value in letters:
                    search_pattern += grid_value
                else:
                    search_pattern += "."

            search_results = self.dawg.search(search_pattern)

            compatible_words = []
            for search_result in search_results:
                is_compatible = False
                for k in range(word_len):
                    letter = search_result[k]
                    letter_index = letter_to_index[letter]

                    if grid_letter_bitarray[clue_i][clue_j + 1 + k][letter_index] == 1:
                        is_compatible = True
                        break

                if is_compatible:
                    compatible_words.append(search_result)

            return compatible_words

        if clue_orientation == DOWN:
            search_pattern = ""
            for k in range(word_len):
                grid_value = grid[clue_i + 1 + k][clue_j]

                if grid_value in letters:
                    search_pattern += grid_value
                else:
                    search_pattern += "."

            search_results = self.dawg.search(search_pattern)

            compatible_words = []
            for search_result in search_results:
                is_compatible = False
                for k in range(word_len):
                    letter = search_result[k]
                    letter_index = letter_to_index[letter]

                    if grid_letter_bitarray[clue_i + 1 + k][clue_j][letter_index] == 1:
                        is_compatible = True
                        break

                if is_compatible:
                    compatible_words.append(search_result)

            return compatible_words

    def backtracking(self, grid, clue_candidates, grid_letter_bitarray):
        """
        si plus de candidat => c'est bon

        sinon
            on choisit un candidat
            on choisit une liste de mots compatibles
            si aucun mot possible => échec
            pour chaque mot possible, on lance la fonction récursive
        """

        print("backtracking")
        # self.image = self.generate_image(grid, clue_candidates, grid_letter_bitarray)
        self.generate_image(grid, clue_candidates, grid_letter_bitarray)

        if len(clue_candidates) == 0:
            return grid

        clue_i, clue_j, clue_orientation, word_len = self.choose_clue_candidate(
            grid, clue_candidates, grid_letter_bitarray
        )

        compatible_words = self.choose_compatible_words(
            grid,
            clue_candidates,
            grid_letter_bitarray,
            clue_i,
            clue_j,
            clue_orientation,
            word_len,
        )

        for word in compatible_words:
            new_grid, new_clue_candidates, new_grid_letter_bitarray = self.add_word(
                grid,
                clue_candidates,
                grid_letter_bitarray,
                word,
                clue_i,
                clue_j,
                clue_orientation,
            )

            result = self.backtracking(
                new_grid, new_clue_candidates, new_grid_letter_bitarray
            )

            if result is not None:
                return result

        return None

    def solve(self):
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # im = ax.imshow(self.image)

        # self.fig = fig
        """

        def update_image(i):
            print("update image")

            im.set_array(self.image)
            # time.sleep(.5)
            # plt.pause(0.5)

        anim = animation.FuncAnimation(fig, update_image, interval=1000)  # TODO interval
        """
        # plt.show(block=False)

        return self.backtracking(
            self.initial_grid,
            self.initial_clue_candidates,
            self.initial_grid_letter_bitarray,
        )

    def generate_image(self, grid, clue_candidates, grid_letter_bitarray):
        svg_string = self.generate_svg(grid, clue_candidates, grid_letter_bitarray)

        # reference: https://github.com/Kozea/CairoSVG/issues/57
        tree = cairosvg.surface.Tree(bytestring=svg_string)
        surface = cairosvg.surface.PNGSurface(tree, None, self.dpi).cairo

        im = np.frombuffer(surface.get_data(), np.uint8)
        h, w = surface.get_height(), surface.get_width()
        im.shape = (h, w, 4)  # for RGBA

        rgb_image = im[:, :, :3]

        im = Image.fromarray(rgb_image)
        im.save(f"{self.image_i}.png")
        self.image_i += 1
        # time.sleep(1)

        return rgb_image

    def generate_svg(self, grid, clue_candidates, grid_letter_bitarray):
        # TODO move (constants or parameters)
        margin_x = 1
        margin_y = 1
        stroke_width = 0.1

        drawing = svgwrite.Drawing(
            profile="full",
            size=(self.image_width, self.image_height),
            viewBox=f"0 0 {m + 1 + 2 * margin_x} {n + 1 + 2 * margin_y}",
        )

        drawing.add(
            drawing.rect(
                (0, 0), (m + 1 + 2 * margin_x, n + 1 + 2 * margin_y), fill="black"
            )
        )

        drawing.add(drawing.rect((1 + margin_x, 1 + margin_y), (m, n), fill="white"))

        for i in range(1, n + 1 + 1):
            drawing.add(
                drawing.line(
                    (margin_x + 1, margin_y + i),
                    (margin_x + m + 1, margin_y + i),
                    stroke_width=stroke_width,
                    stroke=svgwrite.rgb(0, 0, 0, "%"),
                )
            )

        for j in range(1, m + 1 + 1):
            drawing.add(
                drawing.line(
                    (margin_x + j, margin_y + 1),
                    (margin_x + j, margin_y + n + 1),
                    stroke_width=stroke_width,
                    stroke=svgwrite.rgb(0, 0, 0, "%"),
                )
            )

        for i in range(0, n + 1):
            for j in range(0, m + 1):
                value = grid[i][j]

                if value is None:
                    compatible_letter_count = grid_letter_bitarray[i][j].count()

                    x = margin_x + j
                    y = margin_y + i
                    width, height = 1, 1
                    interpolation_factor = compatible_letter_count / len(letters)
                    r1, g1, b1 = (255, 255, 255)
                    r0, g0, b0 = (255, 0, 0)
                    r, g, b = (
                        interpolation_factor * r1 + (1 - interpolation_factor) * r0,
                        interpolation_factor * g1 + (1 - interpolation_factor) * g0,
                        interpolation_factor * b1 + (1 - interpolation_factor) * b0,
                    )
                    drawing.add(
                        drawing.rect(
                            insert=(x + stroke_width / 2, y + stroke_width / 2),
                            size=(width - stroke_width, height - stroke_width),
                            fill=f"rgb({round(r)}, {round(g)}, {round(b)})",
                        )
                    )

                    size = 0.18
                    x = margin_x + j + 0.20  # TODO?
                    y = margin_y + i + 0.25  # TODO?
                    drawing.add(
                        drawing.text(
                            compatible_letter_count,
                            insert=(x, y),
                            dominant_baseline="middle",
                            text_anchor="middle",
                            font_size=f"{size}",
                            fill="black",
                        )
                    )

                    continue

                if value == BLOCKED:  # or value == DOWN or value == RIGHT:
                    x = margin_x + j
                    y = margin_y + i
                    width, height = 1, 1
                    drawing.add(
                        drawing.rect(
                            insert=(x + stroke_width / 2, y + stroke_width / 2),
                            size=(width - stroke_width, height - stroke_width),
                            fill="black",
                        )
                    )

                    continue

                if value == CLUE:
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

                size = 0.8
                x = margin_x + j + 0.5  # TODO?
                y = margin_y + i + 0.5  # TODO?
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

        return drawing.tostring()


if __name__ == "__main__":
    # reference example: https://www.fortissimots.com/wp-content/uploads/fleches_fortissimots_51_dumont.pdf
    n, m = 14, 10
    min_len = 2
    custom_clues = [
        (4, 6),
        (5, 8),
        (6, 3),
        (7, 8),
        (8, 7),
        (8, 9),
        (9, 6),
        (10, 3),
        (12, 3),
        (12, 6),
        (13, 5),
    ]

    lexicon = []
    with open("dev_lexicon.txt") as f:
        # with open("officiel-du-scrabble-8.txt") as f:
        lexicon = [line.strip() for line in f]

    solver = Solver(n, m, min_len, custom_clues, lexicon)

    solution = solver.solve()

    """

    grid, clue_candidates, grid_letter_bitarray, clue_to_clue = initialize(
        n, m, min_len, clues
    )
    draw_grid("grid_0", grid, clue_candidates, grid_letter_bitarray, n, m)

    print(clue_candidates)

    # first clue/word



    draw_grid("grid_1", grid, clue_candidates, grid_letter_bitarray, n, m)
    exit()

    # print(dawg.search("S...RIS."))
    # print(len(dawg.search("C?L?R")))

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

    grid, clue_candidates, grid_letter_bitarray = add_word(
        grid,
        clue_candidates,
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

    draw_grid("grid_2", grid, clue_candidates, grid_letter_bitarray, n, m)

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

    grid, clue_candidates, grid_letter_bitarray = add_word(
        grid,
        clue_candidates,
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

    draw_grid("grid_3", grid, clue_candidates, grid_letter_bitarray, n, m)

    # 2^5 = 32
    # 4 bytes
    # 0b1111111111111111111111111
    
    """

    """
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
    """

    """
    for j in range(3, m + 1):
        print(f"--- {j}")
        print(grid[2][j])
        vertical_results = dawg.search(f".{grid[2][j]}.*")
        print(len(vertical_results))

        vertical_compatible_letters = set([w[2] for w in vertical_results])

        print(grid[3][2])
        horizontal_results = dawg.search(f"{grid[3][2]}..*")
        print(len(horizontal_results))

        horizontal_compatible_letters = set([w[2] for w in horizontal_results])

        compatible_letters = vertical_compatible_letters.intersection(
            horizontal_compatible_letters
        )

        print(len([w for w in vertical_results if w[2] in compatible_letters]))
        print(len([w for w in horizontal_results if w[2] in compatible_letters]))
    """

    # TODO à lire
    """
    longueur possible pour mot horizontale :
    - 1 : NON car config interdite
    - 2 : NON car < min_len
    - 4, 6, 8 : NON car impliquerait de ne pas respecter min_len verticalement
    - 3, 5, 7, 9 : OUI
    
    lettre representée par 32 bits
    type c uint32
    """

    """
    n, m = 10, 10
    min_word_length = 3

    grid = initialize(n, m)

    print(grid)

    draw_grid(grid)
    """
