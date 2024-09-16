# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 21:45:09 2024

@author: lcuev
"""
import pygame
from numpy import tanh


class Hopfield:
    def __init__(self, length=16**2):
        self.length = length
        self.area = self._flatten_(length)
        self.cells = [0 for c in range(length)]
        self.weights = [0 for c in range(self.area)]
        self.dict = self._get_dict_(self.area)

    def _get_dict_(self, length):
        dict = {}
        for i in range(length):
            dict[i] = self._unflatten_(i)
        return dict

    def _unflatten_(self, index):
        f = int(0.5 * (8 * index + 1) ** 0.5 + 0.5)
        return f, index - f * (f - 1) // 2

    def _flatten_(self, left, right=0):
        return left * (left - 1) // 2 + right

    def _activate_(self, input):
        for i in range(self.length):
            input[i] = tanh(input[i])
        return input

    def memorize(self, output):
        for key, value in self.dict.items():
            left, right = value
            self.weights[key] += output[left] * output[right]

    def recall(self, input, rate=2**-7):
        for key, value in self.dict.items():
            left, right = value
            input[left] += self.weights[key] * input[right] * rate
            input[right] += self.weights[key] * input[left] * rate

        return self._activate_(input)


class GUI:
    bg_col = (0, 0, 0)
    recall_col = (255, 0, 0)
    pen_col = (255, 255, 0)
    erase_col = (255, 0, 255)

    def __init__(self, length=16, cell_size=32):
        pygame.init()
        self.width = length * cell_size
        self.surface = pygame.display.set_mode(
            (self.width, self.width))
        self.font = pygame.font.Font('CascadiaMono.ttf', 32)
        self.length = length
        self.area = length ** 2
        self.cell_size = cell_size
        self.cells = [-1 for a in range(self.area)]
        self.pen_down = False
        self.erase = False
        self.recall_mode = False
        self.recall_time = 0
        self.running = True
        self.last_change = None
        self.hopfield = Hopfield(self.area)
        pygame.display.set_caption('Hopfield')

    def __del__(self):
        pygame.quit()

    def _flatten_(self, left, right):
        return left // self.cell_size * self.length + right // self.cell_size

    def _unflatten_(self, index):
        return index // self.length * self.cell_size, index % self.length * self.cell_size

    def _handle_events_(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.pen_down = True
            if event.type == pygame.MOUSEBUTTONUP:
                self.pen_down = False
                self.last_change = None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    self.hopfield.memorize(self.cells)
                if event.key == pygame.K_r:
                    self.recall_mode = not self.recall_mode
                if event.key == pygame.K_c:
                    self._clear_cells_()
                if event.key == pygame.K_e:
                    self.erase = not self.erase

    def _clear_cells_(self):
        self.cells = [-1 for a in range(self.area)]

    def _update_(self):
        if self.recall_mode:
            self.cells = self.hopfield.recall(self.cells)
        else:
            if self.pen_down:
                left, right = pygame.mouse.get_pos()
                index = self._flatten_(right, left)
                if self.last_change != index:
                    self.cells[index] = -1 if self.erase else 1
                    self.last_change = index

    def _display_(self):
        self.surface.fill(self.bg_col)
        for index in range(self.area):
            left, right = self._unflatten_(index)
            color = 255 * (self.cells[index] + 1) // 2
            pygame.draw.rect(self.surface, (color, color, color),
                             (right, left, self.cell_size, self.cell_size))

        if self.recall_mode:
            mode = self.font.render('recall', True, self.recall_col)
        else:
            if self.erase:
                mode = self.font.render('eraser', True, self.erase_col)
            else:
                mode = self.font.render('pen', True, self.pen_col)

        mode_rect = mode.get_rect()
        mode_rect.topleft = (5, 0)
        self.surface.blit(mode, mode_rect)
        pygame.display.update()

    def loop(self):
        self._handle_events_()
        self._update_()
        self._display_()
        return self.running


if __name__ == '__main__':
    gui = GUI()
    while gui.loop():
        pass
    del gui
