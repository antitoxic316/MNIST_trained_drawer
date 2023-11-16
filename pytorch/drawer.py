import torch
import pygame
import sys
from CNN import CNNmodel
import numpy as np
import time
import torchvision
import multiprocessing
import concurrent.futures
import matplotlib.pyplot as plt


# model initialisation
device = "cpu"

model = CNNmodel(in_shape=1, hidden_shape=32, out_shape=10).to(device)
model.load_state_dict(torch.load(f="CNN.pth"))



# creating graphic interface with pygame
pygame.init()

SCREEN_W = 800
SCREEN_H = 600

RESULT_SCREEN_X = (600, SCREEN_W)
RESULT_SCREEN_Y = (0, SCREEN_H)

WHITE = (255,255,255)
BLACK = (0,0,0)

NUMS_PADDING = 40
MAIN_FONT_SIZE = 50
INSTRUCTIONS_FONT_SIZE = 22
main_font = pygame.font.Font(None, MAIN_FONT_SIZE)
instrustions_font = pygame.font.Font(None, INSTRUCTIONS_FONT_SIZE)

draw_thickness = 25

screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
result_screen = pygame.Rect(RESULT_SCREEN_X[0], RESULT_SCREEN_Y[0], RESULT_SCREEN_X[1], RESULT_SCREEN_Y[1])
pygame.draw.rect(screen, WHITE, result_screen, 0)

NUMBER_COORDS = [(RESULT_SCREEN_X[0] + NUMS_PADDING, y) \
                 for y in range(RESULT_SCREEN_Y[0] + NUMS_PADDING, RESULT_SCREEN_Y[1], NUMS_PADDING)]
if len(NUMBER_COORDS) > 10: NUMBER_COORDS = NUMBER_COORDS[:10]


for i, coords in enumerate(NUMBER_COORDS):
    num = main_font.render(str(i), True, BLACK)
    chance = main_font.render("0.00", True, BLACK)
    screen.blit(num, coords)
    screen.blit(chance, (coords[0]+NUMS_PADDING, coords[1]))

instructions_coords = NUMBER_COORDS[-1]
instructions = [
    instrustions_font.render("mouse left to draw", True, BLACK),
    instrustions_font.render("mouse right to erase", True, BLACK),
    instrustions_font.render("key down to evaluate", True, BLACK)
]
for i in range(0, len(instructions)):
    screen.blit(instructions[i], (instructions_coords[0], instructions_coords[1]+NUMS_PADDING*(i+1)))



# functions for program work
def blit_new_chance(chance: float, num: int):
    coords = NUMBER_COORDS[num]
    font_chance = main_font.render(str(chance), True, BLACK)
    screen.fill(WHITE, ((coords[0]+NUMS_PADDING, coords[1]), (100, 30)))
    screen.blit(font_chance, (coords[0]+NUMS_PADDING, coords[1]))

def get_draw_screen():
    return [screen.get_at((x, y))[0] for x in range(0, 600) for y in range(0, 600)]

def transform_img(img):
    img = torch.from_numpy(np.array(img))
    img = torch.reshape(img, (600, 600)).unsqueeze(dim=0)
    img = torchvision.transforms.Resize((28, 28))(img)
    img = torch.rot90(img.squeeze(dim=0), 1)
    img = torch.flip(img, (0,)).unsqueeze(dim=0)
    img = img.unsqueeze(dim=0) # add batch dimension
    img = img.float()
    img = img.to(device)
    img /= 255
    return img

def predict(model, draw_screen):
    model.eval()
    y_logits = model(draw_screen)
    return y_logits.squeeze(dim=0).numpy(force=True)

def predict_and_update_chances(model):
    draw_screen = get_draw_screen()
    draw_screen = transform_img(draw_screen)

    pred_classes = predict(model, draw_screen)
    for num, chance in enumerate(pred_classes):
        blit_new_chance(chance=float(f'{chance:.2f}'), num=num)



drawing= True
while True:

    events = pygame.event.get()

    pos = pygame.mouse.get_pos()

    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if pos[0] not in range(RESULT_SCREEN_X[0] - int(draw_thickness), RESULT_SCREEN_X[1]):    
            if pygame.mouse.get_pressed()[0]:
                screen.fill(WHITE, (pos, (draw_thickness, draw_thickness)))
            if pygame.mouse.get_pressed()[2]:
                screen.fill(BLACK, (pos, (draw_thickness, draw_thickness)))
        if event.type == pygame.KEYDOWN:
            drawing = False

    if not drawing:
        predict_and_update_chances(model)
        drawing = True

    pygame.display.flip() 