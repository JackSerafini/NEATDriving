import os
import pickle
import math

import pygame
import neat

# Import your Car class and constants from the main script
# (Assuming you saved it as 'train.py' next to this file)
from main import Car, TRACK_IMG, COLLISION_COLOR, WIDTH, HEIGHT, TURN_RATE

MAX_FRAMES = 1000

def main(config_path, winner_path):
    # Load NEAT config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Load the champion genome
    with open(winner_path, 'rb') as f:
        genome = pickle.load(f)

    # Build its neural net
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Test Winner")
    clock = pygame.time.Clock()

    # Load track and create mask
    track = pygame.image.load(TRACK_IMG).convert()
    track = pygame.transform.smoothscale(track, (WIDTH, HEIGHT))
    track_mask = pygame.mask.from_threshold(track, COLLISION_COLOR, (1,1,1))

    # Instantiate a single Car with the winner net
    car = Car(track_mask)

    running = True
    while running:
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                running = False

        screen.blit(track, (0,0))

        # Update car and check collision
        if not car.alive:
            break

        inputs = car.get_data()
        outputs = net.activate(inputs)
        steer = outputs[0] * 2 - 1
        car.angle += steer * TURN_RATE
        car.speed += outputs[1]
        car.update()
        screen.blit(car.vehicle, car.rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    here = os.path.dirname(__file__)
    cfg  = os.path.join(here, 'config.ini')
    winp = os.path.join(here, 'winner.pkl')
    main(cfg, winp)