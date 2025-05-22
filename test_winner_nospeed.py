import os
import pickle

import pygame
import neat

from main_nospeed import Car, TRACK_IMG, COLLISION_COLOR, WIDTH, HEIGHT, TURN_RATE

def main(config_path: str, winner_path: str):
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path
    )

    with open(winner_path, 'rb') as f:
        genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Test Winner")
    clock = pygame.time.Clock()

    track = pygame.image.load(TRACK_IMG).convert()
    track = pygame.transform.smoothscale(track, (WIDTH, HEIGHT))
    track_mask = pygame.mask.from_threshold(track, COLLISION_COLOR, (1,1,1))

    car = Car(track_mask)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.blit(track, (0,0))
        if not car.alive:
            break

        inputs = car.get_data()
        outputs = net.activate(inputs)
        steer = outputs[0] * 2 - 1
        car.angle += steer * TURN_RATE
        car.update()
        screen.blit(car.vehicle, car.rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path  = os.path.join(local_dir, 'config_nospeed.ini')
    winner_path = os.path.join(local_dir, 'winner_nospeed.pkl')
    main(config_path, winner_path)