#!/usr/bin/env python3

import time
import math
import random
import numpy as np
# import cv2
from PIL import Image
from PIL import ImageDraw


from rgbmatrix import RGBMatrix, RGBMatrixOptions

colors = {
    "b": (1.0, 0.0, 0.0),  # Blue
    "g": (0.0, 1.0, 0.0),  # Green
    "r": (0.0, 0.0, 1.0),  # Red
    "lb": (1.0, 0.5, 0.5),  # Light blue
    "lg": (0.5, 1.0, 0.5),  # Light green
    "lr": (0.5, 0.5, 1.0),  # Light red (pink lol)
    "wh": (1.0, 1.0, 1.0),  # White
    "gr": (0.5, 0.5, 0.5),  # Gray
    "bl": (0.0, 0.0, 0.0),  # Black
}


def smooth_osc(x):
    return ((16 * x - 32) * x + 16) * x * x


def smooth_step(x):
    return (-2 * x + 3) * x * x


#### Screen functions


def gradient_char(value):
    np.clip(value, 0, 1)
    print(value)
    chars = [" ", "~", ":", "*", "%", "@", "@"]
    return chars[int(value * 6)]


def print_screen(screen, method="rgbmatrix", wait_for_key=True, matrix=None):
    if method == "rgbmatrix":
        image = Image.new("RGB", (192, 128))
        # TODO: Put the pixels into image
        matrix.Clear()
        matrix.SetImage(image)
    elif method == "ascii":
        for row in reversed(screen):
            for val in row:
                print(gradient_char(val) * 2, end="")
            print()
    elif method == "cv2_window":
        sc = screen
        # sc = np.array(sc)
        sc = np.flip(sc, 0)
        sc = np.clip(sc, 0, 1)
        sc = np.multiply(sc, 255)
        sc = sc.astype("uint8")
        cv2.imshow("window", sc)
        # Decodes the system character by masking part of the hex code,
        # then using chr()
        if wait_for_key:
            ret_char = chr(cv2.waitKey() & 0xFF)
        else:
            ret_char = chr(cv2.waitKey(25) & 0xFF)
        if ret_char in ("q", "e"):
            cv2.destroyAllWindows()
        return ret_char
    else:
        print(f"Unknown method '{method}'")


# Just fiddling for performance
class Timer:
    def __init__(self):
        self.start_time = time.time()

    def print_time(self, message=""):
        print(message + str(time.time() - self.start_time))


#
class Shader:
    """Main class where I put all the visual generating stuff in.
    An instance of it is one specific pattern.
    shape - a string that gives information on which pattern to run
      It can contain multiple words
    param - a general-use variable to pass in a value for one of the shapes
    lifespan - For particle shapes, it's how long each particle will exist
      For whole shapes, it's how long a cycle is
    spawn_inter - Time between each spawn of a particle
      (accounts for delay between calls)
    gen_color - A function that gives what color should be generated
      based on a seed and an age. The seed is used to differentiate between
      separate particles (if you want randomness), and age is to have color
      based on the time in a lifespan (value from 0-1).
    """

    def __init__(
        self, shape="point", param=None, lifespan=1, spawn_inter=0.1, gen_color=None
    ):
        random.seed()
        self.shape = shape
        self.param = param
        self.lifespan = lifespan
        self.spawn_inter = spawn_inter
        # Default gen_color function
        if not gen_color:

            def gen_color(seed, age):
                return [(-2 * abs(age - 0.5) + 1) for i in range(3)]

        self.gen_color = gen_color

        self.clear_sides()
        self.points = []
        self.start_time = time.time()
        self.prev_time = time.time()
        self.curr_time = 0
        self.delta_time = 0

        ### For going over sides
        # Which face to go to
        # [0, 1, 2, 3] => (N, E, S, W)
        # (0, 1, 2, 3, 4, 5) => (left, front, right, back, top, bot)
        self.links = (
            (4, 1, 5, 3),
            (4, 2, 5, 0),
            (4, 3, 5, 1),
            (4, 0, 5, 2),
            (3, 2, 1, 0),
            (1, 2, 3, 0),
        )

        # How to rotate to get there
        # [0, 1, 2, 3] => (N, E, S, W)
        # (0, 1, 2, 3) => (0, 90, 180, 270) clockwise
        self.rots = (
            (3, 0, 1, 0),
            (0, 0, 0, 0),
            (1, 0, 3, 0),
            (2, 0, 2, 0),
            (2, 3, 0, 1),
            (0, 1, 2, 3),
        )

    def clear_sides(self):
        # Screen panels
        #   Side order: (left, front, right, back, top, bot)
        self.sides = [np.zeros((64, 64, 3)) for _ in range(6)]

    # Get a cardinal direction (in 0-3 rotation form) based on x/y +/- direction
    def get_card(self, x_drc, y_drc):
        if x_drc < 0:
            return 3
        if x_drc > 0:
            return 1
        if y_drc < 0:
            return 2
        else:
            return 0

    # Convert the side, rotation, x, and y to the corrisponding values of
    #   another side (when the x and y coords are out of the current side's bounds)
    # This does not work when the coords are out of bounds in both dimensions.
    #   (only 0 <= x < 64 or 0 <= y < 64, not both)
    def get_side_info(self, side, rot, x, y):
        x_drc = math.floor(x / 64)
        y_drc = math.floor(y / 64)
        card = self.get_card(x_drc, y_drc)
        card = (card + rot) % 4
        rot = self.rots[side][card]
        side = self.links[side][card]
        x = x % 64
        y = y % 64
        a, b = self.get_coord(x, y, rot)
        return (side, rot, a, b)

    # Get the new coordinates after a rotation
    def get_coord(self, x, y, rot):
        if rot == 0:
            return (x, y)
        elif rot == 1:
            return ((-y - 1) % 64, x)
        elif rot == 2:
            return ((-x - 1) % 64, (-y - 1) % 64)
        elif rot == 3:
            return (y, (-x - 1) % 64)

    # Adds particles to the particle list based on the spawn interval
    def add_particles(self):
        # If no particles were spawned in the last {self.spawn_inter} seconds
        if all(
            [
                (self.curr_time - spawn_time) > self.spawn_inter
                for _, _, _, _, spawn_time in self.points
            ]
        ):
            # Account for frame delay
            spawn_cnt = math.ceil(self.delta_time / self.spawn_inter)
            for _ in range(spawn_cnt):
                random.seed()
                seed = random.randint(0, 1024)
                side = random.randrange(6)
                x = random.random() * 64
                y = random.random() * 64
                spawn_time = self.curr_time
                self.points.append([seed, side, x, y, spawn_time])

    # Remove particles that are too old
    def remove_particles(self):
        for i, point in enumerate(list(self.points)):
            spawn_time = point[4]
            age = (self.curr_time - spawn_time) / self.lifespan
            if age > 1:
                del self.points[i]

    # These are shaders that are based around spawn points
    def render_particles(self):
        # Particle ideas:
        # Particles with physics (with platforms)
        # Shooting stars that go sideways
        # Text that wraps around
        # Wobbly stuff
        # Noise that stays in place (when move cube, lights follow)
        # Sphereify

        for point in self.points:
            seed, side, x, y, spawn_time = point
            rot = 0
            xi = int(x)
            yi = int(y)
            age = (self.curr_time - spawn_time) / self.lifespan
            # Skip particles that are now too old
            #   (to account for delay while rendering)
            if age > 1:
                continue

            # Only edits the value at that point
            if "point" in self.shape:
                self.sides[side][yi][xi] += self.gen_color(seed, age)

            # Makes a radiating circle from that specific point
            #   (that continues onto another side)
            if "rain" in self.shape:
                size = 4
                if self.param:
                    size = self.param
                for dy in range(-size, size + 1):
                    for dx in range(-size, size + 1):
                        dis = math.sqrt(dx ** 2 + dy ** 2)
                        # First calculate if it's a pixel on the ring
                        val = -1 * abs(dis - size * age) + 1
                        if val <= 0:
                            continue
                        a = xi + dx
                        b = yi + dy
                        if 0 <= a < 64 and 0 <= b < 64:
                            nside, nrot, na, nb = (side, rot, a, b)
                        elif 0 <= a < 64 or 0 <= b < 64:
                            nside, nrot, na, nb = self.get_side_info(side, rot, a, b)
                        else:
                            continue
                        ci = self.gen_color(seed, age)
                        co = [val * (1 - age) * ci[i] for i in range(3)]
                        self.sides[nside][nb][na] += co

            # Makes streaks of light
            if "streak" in self.shape:
                random.seed(seed)
                x_mult = random.choice((1, -1))
                y_mult = random.choice((1, -1))
                length = 10
                if self.param:
                    length = self.param
                start = int(age * 32)
                for pix in range(length):
                    si = start + pix
                    a = xi + x_mult * si
                    b = yi + y_mult * si
                    if 0 <= a < 64 and 0 <= b < 64:
                        nside, nrot, na, nb = (side, rot, a, b)
                    elif 0 <= a < 64 or 0 <= b < 64:
                        nside, nrot, na, nb = self.get_side_info(side, rot, a, b)
                    else:
                        continue
                    ci = self.gen_color(seed, age)
                    co = [(pix / length) * ci[i] for i in range(3)]
                    self.sides[nside][nb][na] += co

    # These are shaders that color all pixels
    def render_whole(self):
        if "wave" in self.shape:
            age = (self.curr_time % self.lifespan) / self.lifespan
            for side in range(6):
                if side in (0, 3):
                    rot = 0
                elif side == 4:
                    rot = 2
                else:
                    rot = 3
                nage = age
                if side in (2, 3, 5):
                    nage = (age + 0.5) % 1
                for a in range(64):
                    for b in range(64):
                        nnage = (nage + (a + b) / 128) % 1
                        na, nb = self.get_coord(a, b, rot)
                        self.sides[side][na][nb] = self.gen_color(0, nnage)

        # Accidental version of 'wave' that doesn't reset a variable after each loop
        #   (and creates crazy patterns)
        if "crazy" in self.shape:
            age = (self.curr_time % self.lifespan) / self.lifespan
            for side in range(6):
                if side in (0, 3):
                    rot = 0
                elif side == 4:
                    rot = 2
                else:
                    rot = 3
                if side in (2, 3, 5):
                    age = (age + 0.5) % 1
                for a in range(64):
                    for b in range(64):
                        age = (age + (a + b) / 128) % 1
                        na, nb = self.get_coord(a, b, rot)
                        # nside, nrot, na, nb = self.get_side_info(side, rot, a, b)
                        self.sides[side][na][nb] = self.gen_color(0, age)

    # Main render function that colors the screens
    def render(self, screen):
        self.curr_time = time.time() - self.start_time
        self.delta_time = time.time() - self.prev_time
        self.prev_time = time.time()

        self.clear_sides()
        if "particle" in self.shape:
            self.add_particles()
            self.remove_particles()
            self.render_particles()
        if "whole" in self.shape:
            self.render_whole()

        # This is rendering the 64x64 screens onto a single window
        # Cube skin pattern like:
        #       ##
        #     ########
        #       ##
        screen[64 * 1 : 64 * 2, 64 * 0 : 64 * 1] = self.sides[0]
        screen[64 * 1 : 64 * 2, 64 * 1 : 64 * 2] = self.sides[1]
        screen[64 * 1 : 64 * 2, 64 * 2 : 64 * 3] = self.sides[2]
        screen[64 * 1 : 64 * 2, 64 * 3 : 64 * 4] = self.sides[3]
        screen[64 * 2 : 64 * 3, 64 * 1 : 64 * 2] = self.sides[4]
        screen[64 * 0 : 64 * 1, 64 * 1 : 64 * 2] = self.sides[5]


def emersons_favorites(name="green_stars"):
    global colors
    if name == "green_stars":

        def gen_color(seed, age):
            random.seed(seed)
            c1 = colors["lg"]
            variance = 0.2 + random.random() * 0.8
            co = [c1[i] * (-2 * abs(age - 0.5) + 1) * variance for i in range(3)]
            return co

        return Shader(
            shape="particle point", lifespan=8, spawn_inter=0.03, gen_color=gen_color
        )
    elif name == "rainbow_sparkles":

        def gen_color(seed, age):
            random.seed(seed)
            t = time.time()
            period = 4
            rainbow = [
                (
                    -2
                    * abs(
                        (
                            (t + i * period / 3 + random.random() * 0.15 * period)
                            % period
                            / period
                        )
                        - 0.5
                    )
                    + 1
                )
                for i in range(3)
            ]
            co = [rainbow[i] * (-2 * abs(age - 0.5) + 1) for i in range(3)]
            return co

        return Shader(
            shape="particle point", lifespan=1, spawn_inter=0.005, gen_color=gen_color
        )
    elif name == "rain_drops":

        def gen_color(seed, age):
            random.seed(seed)
            c1 = colors["wh"]
            variance = 0.5 + random.random() * 0.5
            co = [c1[i] * (-2 * abs(age - 0.5) + 1) * variance for i in range(3)]
            return co

        return Shader(
            shape="particle rain",
            param=4,
            lifespan=3,
            spawn_inter=0.08,
            gen_color=gen_color,
        )
    elif name == "rainbow_wave":

        def gen_color(seed, age):
            co = [(-2 * abs(((age + i / 3) % 1) - 0.5) + 1) for i in range(3)]
            return co

        return Shader(shape="whole wave", lifespan=8, gen_color=gen_color)
    elif name == "crazy":

        def gen_color(seed, age):
            co = [(-2 * abs(((age + i / 3) % 1) - 0.5) + 1) for i in range(3)]
            return co

        return Shader(shape="whole crazy", lifespan=5, gen_color=gen_color)
    elif name == "green_streaks":

        def gen_color(seed, age):
            c1 = colors["g"]
            co = [c1[i] * (-2 * abs(age - 0.5) + 1) for i in range(3)]
            return co

        return Shader(
            shape="particle streak",
            param=10,
            lifespan=5,
            spawn_inter=0.1,
            gen_color=gen_color,
        )
        # return Shader(shape = 'particle streak', param = 10, lifespan = 1, spawn_inter = 0.01, gen_color = gen_color)
    elif name == "night_light":

        def gen_color(seed, age):
            global warmness
            global intensity
            c1 = [0.8 - warmness * 0.2, 0.8, 0.8 + warmness * 0.2]
            # co = [c1[i] * (0.8 + abs(age - 0.5) * 0.4) * intensity for i in range(3)]
            co = [c1[i] * (0.8 + smooth_osc(age) * 0.2) * intensity for i in range(3)]
            return co

        return Shader(shape="whole wave", lifespan=60, gen_color=gen_color)
    else:
        print(f"No such favorite '{name}'")
        exit()


def main():
    # Configuration for the matrix
    options = RGBMatrixOptions()
    options.rows = 64
    options.cols = 128
    options.chain_length = 1
    options.parallel = 3
    options.hardware_mapping = 'regular'  # If you have an Adafruit HAT: 'adafruit-hat'

    matrix = RGBMatrix(options = options)


    shader = emersons_favorites("night_light")

    # Make a custom color based on age (0-1)
    #   I have a lot of 'preset' ones as comments in there too that you can try out.
    def gen_color(seed, age):
        global warmness
        global intensity
        c1 = [0.8 - warmness * 0.2, 0.8, 0.8 + warmness * 0.2]
        # co = [c1[i] * (0.8 + abs(age - 0.5) * 0.4) * intensity for i in range(3)]
        co = [c1[i] * (0.8 + smooth_osc(age) * 0.2) * intensity for i in range(3)]
        return co

    # Loop to display screen (press q to close it)
    ret_char = None
    last_time = time.time()
    while ret_char != "q":
        # Here you can set global variables
        #   for some shaders to be controlled 'remotely'
        global warmness
        global intensity
        # warmness = abs((time.time() % 5) / 5 * 4 - 2) - 1
        warmness = 0.5
        intensity = 0.4

        screen = np.full((64 * 3, 64 * 4, 3), 0.1)
        shader.render(screen)
        # screen = np.kron(screen, np.ones((2,2,1)))
        ret_char = print_screen(screen, wait_for_key=False, matrix=matrix)

        print("\r" + str(time.time() - last_time), end="")
        last_time = time.time()


if __name__ == "__main__":
    main()
