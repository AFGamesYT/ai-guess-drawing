from os import makedirs
from os.path import join
import os.path
import os

import shutil

import numpy as np

import pygame

from timer import Timer

from PIL import Image, ImageOps

from random import choice

import train_model

import tensorflow as tf

pygame.init()

WIDTH = 1250
HEIGHT = 1000

FPS = 120

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI guesses the drawing")

clock = pygame.time.Clock()
run = True
fullscreen = True

drawings_list = [
    "pizza", "tree", "house", "sun", "ball", "shoe", "cupcake", "fish",
    "ice cream", "cloud", "banana", "glasses", "key", "apple", "door", "bridge",
    "ladder", "cookie", "turtle", "drum", "ghost", "candle"
]
selected_theme = None

DRAWINGS_DIR = "drawings/"

for drawing in drawings_list:
    makedirs(join(DRAWINGS_DIR, drawing), exist_ok=True)


def convert_list(brightness_values: list | np.ndarray, name: str, to_folder: str, rotate: int = 0, mirror: bool = False):
    side_length = int(np.sqrt(len(brightness_values)))
    image_array = brightness_values[:side_length ** 2].reshape((side_length, side_length))

    image_array = (image_array * 255).astype(np.uint8)

    image = Image.fromarray(image_array, mode='L')

    image = image.rotate(rotate)
    if mirror:
        image = ImageOps.mirror(image)

    if to_folder.endswith("/"):
        image.save(f"{to_folder}{name}.png")
    else:
        image.save(f"{to_folder}/{name}.png")


def handle_keyinputs():
    global fullscreen, screen, run
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_q:
            run = False


class Button:
    def __init__(self, scr: pygame.Surface, rect: pygame.Rect | tuple[int, int, int, int], color: tuple[int, int, int],
                 btn_text: str, btntext_color: tuple[int, int, int], on_click,
                 border_thickness: int = 5, border_color: tuple[int, int, int] = (255, 255, 255)):

        self.screen = scr
        self.rect = pygame.Rect(rect)
        self.color = color
        self.text = btn_text
        self.text_color = btntext_color
        self.on_click = on_click

        self.active = True
        self.show = True

        self.border_thickness = border_thickness
        self.border_color = border_color

    def handle(self):
        if self.show:
            pygame.draw.rect(self.screen, self.border_color,
                             (self.rect[0] - self.border_thickness, self.rect[1] - self.border_thickness,
                              self.rect[2] + (self.border_thickness * 2), self.rect[3] + (self.border_thickness * 2)))
            pygame.draw.rect(self.screen, self.color, self.rect)

            btn_font = pygame.font.Font("files/Raleway-VariableFont_wght.ttf", self.rect[3] - (self.rect[3] // 3))

            textobj = btn_font.render(self.text, True, self.text_color)
            textobj_rect = textobj.get_rect()

            textobj_rect.center = (self.rect[0] + self.rect[2] // 2, self.rect[1] + self.rect[3] // 2)

            screen.blit(textobj, textobj_rect)

            mouse_pos = pygame.mouse.get_pos()
            if self.rect.colliderect((mouse_pos[0], mouse_pos[1], 1, 1)) and self.active:
                for e in pygame.event.get():
                    if e.type == pygame.MOUSEBUTTONDOWN:
                        self.on_click()

menu = 0
# 0 - main menu
# 1 - before round
# 2 - round
# 3 - after round
# 4 - training mode

show_train_text = True


font = pygame.font.Font("files/Raleway-VariableFont_wght.ttf", 95)
smaller_font = pygame.font.Font("files/Raleway-VariableFont_wght.ttf", 75)


def start_game():
    global menu, selected_theme, drew, show_train_text
    menu = 1
    show_train_text = True
    train_model.train()
    show_train_text = False
    selected_theme = choice(drawings_list)

guessed = False
seconds_passed = 0

MODEL_PATH = "models/best_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

def on_ai_guess(result):
    global guessed
    predicted_class, confidence = result
    if predicted_class == selected_theme and confidence >= 0.3:
        guessed = True
    elif predicted_class == selected_theme and confidence < 0.3:
        print("The AI technically guessed it, but it isn't confident enough.")

    print(f"AI guessed: {predicted_class}. Confidence: {confidence:.2f}")

def start_drawing():
    global menu, seconds_passed

    round_length = 20

    seconds_passed = 0
    menu = 2
    TMP_DIR = "temp"
    IMG_DIR = os.path.join(TMP_DIR, "guess_image.png")
    # CLASS_DIR = os.path.join("drawings", str(selected_theme))

    def on_timer():
        global menu, seconds_passed
        seconds_passed += 1
        if os.path.isfile(IMG_DIR):
            os.remove(IMG_DIR)

        np_drew = np.array(drew)
        convert_list(np_drew, "guess_image", TMP_DIR, -90, True)

        # train_model.guess(IMG_DIR, model)

        train_model.guess_async(IMG_DIR, model, on_ai_guess)

        if not guessed and not seconds_passed >= round_length:
            guess_timer.start()
        else:
            if guessed and not seconds_passed >= round_length:
                print("The AI has guessed correctly!")
            if not guessed and seconds_passed >= round_length:
                print("You ran out of time. The AI didn't guess your drawing.")

            menu = 3

    guess_timer = Timer(1, on_timer)
    guess_timer.start()


def reset_drawing():
    global drew, grid_size
    drew = []
    for a in range(grid_size):
        for b in range(grid_size):
            drew.append(0.0)


start_game_btn = Button(
    scr=screen,
    rect=((WIDTH // 2) - 200, (HEIGHT // 2) - 50, 400, 100),
    color=(29, 158, 0),
    btn_text="Start Game!",
    btntext_color=(255, 255, 255),
    on_click=start_game,
    border_color=(143, 255, 117)
)

def train_mode():
    global menu, selected_theme
    menu = 4
    selected_theme = input("Select a theme: ")
    if selected_theme not in drawings_list:
        menu = 0
    else:
        TMP_DIR = "temp"
        IMG_DIR = os.path.join(TMP_DIR, "guess_image.png")

        def on_timer():
            global menu
            if os.path.isfile(IMG_DIR):
                os.remove(IMG_DIR)

            np_drew = np.array(drew)
            convert_list(np_drew, "guess_image", TMP_DIR, -90, True)

            if menu == 4:
                guess_timer.start()
            else:
                return

        guess_timer = Timer(1, on_timer)
        guess_timer.start()

training_mode_btn = Button(
    scr=screen,
    rect=((WIDTH // 2) - 200, (HEIGHT // 2) + 100, 400, 100),
    color=(140, 39, 130),
    btn_text="Train The AI",
    btntext_color=(255, 255, 255),
    on_click=train_mode,
    border_color=(54, 15, 50)
)

start_drawing_btn = Button(
    scr=screen,
    rect=((WIDTH // 2) - 225, (HEIGHT // 2) - 50, 450, 100),
    color=(29, 158, 0),
    btn_text="Start Drawing!",
    btntext_color=(255, 255, 255),
    on_click=start_drawing,
    border_color=(143, 255, 117)
)

reset_drawing_btn = Button(
    scr=screen,
    rect=(1025, 60, 200, 40),
    color=(255, 0, 0),
    btn_text="Erase Drawing",
    btntext_color=(0, 0, 0),
    on_click=reset_drawing,
    border_color=(255, 156, 156)
)

def go_back_to_main_menu():
    global menu, drew, selected_theme, guessed
    drew = []
    for x in range(grid_size):
        for y in range(grid_size):
            drew.append(0.0)
    selected_theme = choice(drawings_list)
    menu = 0
    guessed = False

def retry():
    global menu, drew, selected_theme, show_train_text, guessed
    drew = []
    for x in range(grid_size):
        for y in range(grid_size):
            drew.append(0.0)
    selected_theme = choice(drawings_list)
    guessed = False
    menu = 1
    show_train_text = True
    train_model.train()
    show_train_text = False


to_main_menu_btn = Button(
    scr=screen,
    rect=(600, (HEIGHT // 2) + 150, 500, 100),
    color=(135, 0, 113),
    btn_text="To Main Menu",
    btntext_color=(255, 255, 255),
    on_click=go_back_to_main_menu,
    border_color=(84, 0, 79)
)

retry_btn = Button(
    scr=screen,
    rect=(200, (HEIGHT // 2) + 150, 200, 100),
    color=(133, 145, 26),
    btn_text="Retry",
    btntext_color=(255, 255, 255),
    on_click=retry,
    border_color=(75, 82, 14)
)

def btn_train_ai():
    global train_ai_btn
    TMP_DIR = "temp"
    IMG_DIR = os.path.join(TMP_DIR, "guess_image.png")
    CLASS_DIR = os.path.join("drawings", str(selected_theme))

    file_count = len([f for f in os.listdir(CLASS_DIR) if os.path.isfile(os.path.join(CLASS_DIR, f))])

    dest_dir = os.path.join(CLASS_DIR, str(file_count + 1) + ".png")
    try:
        shutil.move(IMG_DIR, dest_dir)
    except PermissionError:
        print("idk why but ok")
    print("Trained the AI!")
    train_ai_btn.show = False

train_ai_btn = Button(
    scr=screen,
    rect=((WIDTH // 2)-385, (HEIGHT // 2)+300, 770, 100),
    color=(0, 22, 168),
    btn_text="Train the AI on this image.",
    btntext_color=(255, 255, 255),
    on_click=btn_train_ai,
    border_color=(0, 8, 64)
)

def finish_drawing():
    global menu
    TMP_DIR = "temp"
    IMG_DIR = os.path.join(TMP_DIR, "guess_image.png")
    CLASS_DIR = os.path.join("drawings", str(selected_theme))

    file_count = len([f for f in os.listdir(CLASS_DIR) if os.path.isfile(os.path.join(CLASS_DIR, f))])

    dest_dir = os.path.join(CLASS_DIR, str(file_count + 1) + ".png")
    try:
        shutil.move(IMG_DIR, dest_dir)
    except PermissionError:
        print("idk why but ok")
    print("Trained the AI!")
    reset_drawing()

    menu = 0

finish_drawing_btn = Button(
    scr=screen,
    rect=(1025, 900, 200, 40),
    color=(0, 22, 168),
    btn_text="Finish drawing",
    btntext_color=(255, 255, 255),
    on_click=finish_drawing,
    border_color=(0, 8, 64)
)

buttons = [start_game_btn, start_drawing_btn, reset_drawing_btn, to_main_menu_btn, retry_btn, train_ai_btn, training_mode_btn, finish_drawing_btn]

grid_size = 28
SPACE_AVALIABLE = 1000

is_drawing = False

drew = []
for x in range(grid_size):
    for y in range(grid_size):
        drew.append(0.0)

def handle_drawing():
    global grid_size, SPACE_AVALIABLE, is_drawing, drew

    def cptl(pos: tuple[int, int]):
        return pos[1] + (pos[0] * grid_size)

    if pygame.mouse.get_pressed()[0]:
        mouse_pos = pygame.mouse.get_pos()

        if mouse_pos[0] < SPACE_AVALIABLE and mouse_pos[1] < SPACE_AVALIABLE:
            mouse_grid_pos = (
                int(mouse_pos[0] // (SPACE_AVALIABLE / grid_size)),
                int(mouse_pos[1] // (SPACE_AVALIABLE / grid_size)))

            # setting brightness values #
            try:
                # center
                pos = (mouse_grid_pos[0], mouse_grid_pos[1])
                if pos[0] < 0 or pos[1] < 0:
                    raise IndexError

                drew[cptl(pos)] = 1
            except IndexError:
                print("Out of range, not drawing.")

            try:
                # left of center
                pos = (mouse_grid_pos[0] - 1, mouse_grid_pos[1])
                if pos[0] < 0 or pos[1] < 0:
                    raise IndexError

                if 0.6 > drew[cptl(pos)]:
                    drew[cptl(pos)] = 0.6
            except IndexError:
                print("Out of range, not drawing.")

            try:
                # right of center
                pos = (mouse_grid_pos[0] + 1, mouse_grid_pos[1])
                if pos[0] < 0 or pos[1] < 0:
                    raise IndexError

                if 0.6 > drew[cptl(pos)]:
                    drew[cptl(pos)] = 0.6
            except IndexError:
                print("Out of range, not drawing.")

            try:
                # above center
                pos = (mouse_grid_pos[0], mouse_grid_pos[1] - 1)
                if pos[0] < 0 or pos[1] < 0:
                    raise IndexError

                if 0.6 > drew[cptl(pos)]:
                    drew[cptl(pos)] = 0.6
            except IndexError:
                print("Out of range, not drawing.")

            try:
                # below center
                pos = (mouse_grid_pos[0], mouse_grid_pos[1] + 1)
                if pos[0] < 0 or pos[1] < 0:
                    raise IndexError

                if 0.6 > drew[cptl(pos)]:
                    drew[cptl(pos)] = 0.6
            except IndexError:
                print("Out of range, not drawing.")

            try:
                # top left
                pos = (mouse_grid_pos[0] - 1, mouse_grid_pos[1] - 1)
                if pos[0] < 0 or pos[1] < 0:
                    raise IndexError

                if 0.2 > drew[cptl(pos)]:
                    drew[cptl(pos)] = 0.2
            except IndexError:
                print("Out of range, not drawing.")

            try:
                # top right
                pos = (mouse_grid_pos[0] + 1, mouse_grid_pos[1] - 1)
                if pos[0] < 0 or pos[1] < 0:
                    raise IndexError

                if 0.2 > drew[cptl(pos)]:
                    drew[cptl(pos)] = 0.2
            except IndexError:
                print("Out of range, not drawing.")

            try:
                # bottom left
                pos = (mouse_grid_pos[0] - 1, mouse_grid_pos[1] + 1)
                if pos[0] < 0 or pos[1] < 0:
                    raise IndexError

                if 0.2 > drew[cptl(pos)]:
                    drew[cptl(pos)] = 0.2
            except IndexError:
                print("Out of range, not drawing.")

            try:
                # bottom right
                pos = (mouse_grid_pos[0] + 1, mouse_grid_pos[1] + 1)
                if pos[0] < 0 or pos[1] < 0:
                    raise IndexError

                if 0.2 > drew[cptl(pos)]:
                    drew[cptl(pos)] = 0.2
            except IndexError:
                print("Out of range, not drawing.")

    for index, sq in enumerate(drew):
        color = int(sq * 255)
        if color == 0:
            color = 20
        pygame.draw.rect(screen,
                         (color, color, color),
                         (int(index / grid_size) * (SPACE_AVALIABLE / grid_size),
                          (index % grid_size) * (SPACE_AVALIABLE / grid_size),
                          (SPACE_AVALIABLE / grid_size), (SPACE_AVALIABLE / grid_size))
                         )


while run:
    screen.fill((32, 32, 32))

    if menu == 2 or menu == 4:
        handle_drawing()

    for event in pygame.event.get():
        handle_keyinputs()
        if event.type == pygame.QUIT:
            run = False

    for button in buttons:
        button.handle()

    if menu == 0:  # main menu
        start_game_btn.show = True
        training_mode_btn.show = True
        start_drawing_btn.show = False
        reset_drawing_btn.show = False
        to_main_menu_btn.show = False
        retry_btn.show = False
        train_ai_btn.show = False
        finish_drawing_btn.show = False

        text = font.render('Main menu', True, (255, 255, 255))
        textRect = text.get_rect()
        textRect.center = (WIDTH // 2, HEIGHT // 2 - 400)

        screen.blit(text, textRect)

    elif menu == 1:  # before round
        start_game_btn.show = False
        training_mode_btn.show = False
        start_drawing_btn.show = True
        reset_drawing_btn.show = False
        to_main_menu_btn.show = False
        retry_btn.show = False
        train_ai_btn.show = False
        finish_drawing_btn.show = False

        text = font.render(f"The AI is training. Please wait.", True, (255, 255, 255))

        textRect = text.get_rect()
        textRect.center = (WIDTH // 2, HEIGHT // 2 - 200)

        if show_train_text:
            screen.blit(text, textRect)

        text = font.render(f"""You have to draw: {selected_theme}""", True, (255, 255, 255))

        textRect = text.get_rect()
        textRect.center = (WIDTH // 2, HEIGHT // 2 - 200)

        screen.blit(text, textRect)

    elif menu == 2:  # round
        text = smaller_font.render(f"Draw: {selected_theme}. Time left: {20-seconds_passed} seconds.", True, (255, 255, 255))

        textRect = text.get_rect()
        textRect.center = (WIDTH // 2 - 30, HEIGHT // 2 - 430)

        screen.blit(text, textRect)

        start_game_btn.show = False
        training_mode_btn.show = False
        start_drawing_btn.show = False
        reset_drawing_btn.show = True
        to_main_menu_btn.show = False
        retry_btn.show = False
        train_ai_btn.show = False
        finish_drawing_btn.show = False


    elif menu == 3:  # after round
        start_game_btn.show = False
        training_mode_btn.show = False
        start_drawing_btn.show = False
        reset_drawing_btn.show = False
        to_main_menu_btn.show = True
        retry_btn.show = True
        train_ai_btn.show = True
        finish_drawing_btn.show = False

        if guessed:
            text = smaller_font.render(f"The AI guessed your drawing!", True, (255, 255, 255))

            textRect = text.get_rect()
            textRect.center = (WIDTH // 2, HEIGHT // 2 - 100)

            screen.blit(text, textRect)
        else:
            text = smaller_font.render(f"The AI failed to guess your drawing!", True, (255, 255, 255))

            textRect = text.get_rect()
            textRect.center = (WIDTH // 2, HEIGHT // 2 - 100)

            screen.blit(text, textRect)

    elif menu == 4:  # drawing mode
        start_game_btn.show = False
        training_mode_btn.show = False
        start_drawing_btn.show = False
        reset_drawing_btn.show = True
        to_main_menu_btn.show = False
        retry_btn.show = False
        train_ai_btn.show = False
        finish_drawing_btn.show = True

    pygame.display.update()
    clock.tick(FPS)

pygame.quit()
