import os
import math
import pickle

import pygame
import neat

MAX_FRAMES = 1000
WIDTH, HEIGHT = 1920, 1080
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car Racing NEAT")

TRACK_IMG = os.path.join('assets', 'track1.png') # Track 1
# TRACK_IMG = os.path.join('assets', 'track2.png') # Track 2
CAR_IMG = os.path.join('assets', 'bluecar.png')

CAR_X, CAR_Y = 60, 30 # Track 1
# CAR_X, CAR_Y = 80, 40 # Track 2
POS_X, POS_Y, ANGLE = 960, 835, 180 # Track 1
# POS_X, POS_Y, ANGLE = 960, 140, 0 # Track 2
TURN_RATE = 5
MAX_SPEED = 10
MIN_SPEED = 2
ACCEL_RATE = 0.5

COLLISION_COLOR = pygame.Color(0, 0, 0)

class Car:
    def __init__(self, track_mask: pygame.mask.Mask):
        self.car_sprite = pygame.image.load(CAR_IMG).convert_alpha()
        self.car_sprite = pygame.transform.smoothscale(self.car_sprite, (CAR_X, CAR_Y))
        self.vehicle = self.car_sprite

        self.track_mask = track_mask
        self.mask = pygame.mask.from_surface(self.vehicle)

        self.x = POS_X + CAR_X / 2
        self.y = POS_Y + CAR_Y / 2
        self.rect = self.vehicle.get_rect(center=(self.x, self.y))
        self.angle = ANGLE
        self.speed = MIN_SPEED
        self.alive = True

        self.distance = 0

    def drive(self):
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y -= math.sin(rad) * self.speed
        self.rect.center = (self.x, self.y)
        self.distance += self.speed

    def rotate(self):
        self.vehicle = pygame.transform.rotate(self.car_sprite, self.angle)
        self.rect = self.vehicle.get_rect(center=self.rect.center)
        self.mask = pygame.mask.from_surface(self.vehicle)

    def collision(self):
        offset_x = int(self.rect.left)
        offset_y = int(self.rect.top)
        overlap_pos = self.track_mask.overlap(self.mask, (offset_x, offset_y))
        if overlap_pos:
            self.alive = False

    def update(self):
        self.drive()
        self.rotate()
        self.collision()

    def get_data(self):
        data = []
        angles = [-60, -30, 0, 30, 60]
        max_dist = 200
        step = 5

        cx, cy = self.rect.center
        for ray in angles:
            length = 0
            ang = math.radians(self.angle + ray)
            x, y = cx, cy

            while not SCREEN.get_at((x, y)) == COLLISION_COLOR and length < max_dist:
                length += step
                x = int(cx + math.cos(ang) * length)
                y = int(cy - math.sin(ang) * length)

            pygame.draw.line(SCREEN, (255, 255, 255), self.rect.center, (x, y))
            pygame.draw.circle(SCREEN, (0, 255, 0), (x, y), 3)

            data.append(length)
        
        return data


def eval_genomes(genomes, config):
    pygame.init()
    clock = pygame.time.Clock()

    track = pygame.image.load(TRACK_IMG).convert()
    track = pygame.transform.smoothscale(track, (1920, 1080))
    track_mask = pygame.mask.from_threshold(track, COLLISION_COLOR, (1, 1, 1))

    nets, cars, ge = [], [], []
    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        cars.append(Car(track_mask))
        genome.fitness = 0
        ge.append(genome)

    run = True
    frame = 0
    while run and len(cars) > 0 and frame < MAX_FRAMES:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        SCREEN.blit(track, (0, 0))
        for i, car in enumerate(cars):
            if not car.alive:
                nets.pop(i)
                cars.pop(i)
                ge.pop(i)
                continue

            inputs = car.get_data()
            outputs = nets[i].activate(inputs)
            steer = outputs[0] * 2 - 1
            car.angle += steer * TURN_RATE
            throttle = (outputs[1] * 2 - 1) * ACCEL_RATE
            update_speed = max(MIN_SPEED, min(car.speed + throttle, MAX_SPEED))
            car.speed = update_speed
            car.update()
            ge[i].fitness = car.distance

            SCREEN.blit(car.vehicle, car.rect)
        
        pygame.display.flip()
        clock.tick(60)
        frame += 1


def run(config_path: str):
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    winner = pop.run(eval_genomes, 500)  # Run for 500 generations
    print('\nBest genome:\n{!s}'.format(winner))
    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.ini')
    run(config_path)