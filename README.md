# NEATDriving

NEATDriving is a Python project that employs the NEAT algorithm to train virtual cars to navigate a 2D track autonomously. Utilizing Pygame for rendering and NEAT-Python for neuroevolution, the simulation evolves neural networks that control the steering and speed of cars based on sensor inputs.

### Visualization Demo

Below is a visual overview of the training process (just the first three generations) and results (the best genome after 500 generations):

<p align="center">
<img src="media/train.gif" alt="NEAT Training" width="400" height="300"> <img src="media/track1.gif" alt="NEAT Training" width="400" height="300">
</p>

The real test of performance comes when evaluating the best-performing genome on an entirely new, unseen track. Here’s how well the model generalizes:

<p align="center">
<img src="media/track2.gif" alt="NEAT Training" width="800" height="500">
</p>

### Features

- Neuroevolution with NEAT: Evolves neural networks to control car behavior without predefined rules
- Sensor-Based Inputs: Cars use simulated distance sensors at multiple angles to perceive the environment
- Dynamic Speed Control: Neural networks output both steering and throttle values, allowing cars to accelerate and decelerate
- Collision Detection: Cars detect collisions with track boundaries using pixel-perfect masking
- Fitness Evaluation: Cars are rewarded based on the distance traveled without collisions
- Multiple Configurations: Includes different configurations for experimentation, such as `config.ini` and `config_nospeed.ini`

### Requirements

To install the packages, use:
```bash
pip install pygame neat-python
```

*Note*: the suggested use is through a virtual environment, using python3.11 which is the last version of python supported by NEAT-Python (https://neat-python.readthedocs.io/en/latest/):
```bash
python3.11 -m venv myenv
myenv/bin/activate # Mac & Linux
myenv/Scripts/activate # Windows
```

### Usage

1. **Prepare Assets:**
Ensure the assets directory contains the necessary track and car images:
- `track1.png`
- `track2.png`
- `bluecar.png`

2. **Run the Simulation:**
To start training the neural networks:
```bash
python main.py
```

*Note*: for a version without speed control, use `main_nospeed.py`.

3. **View Results:**
After training, the best-performing genome is saved as `winner.pkl` (which can be already found in the repo). To visualize its performance use:
```bash
python test_winner.py
# Or run python test_winner_nospeed.py
```

To test the best-performing genome on `track2` and evaluate its performance on a new and unseen track, uncomment the lines of code at the beginning of either `main.py` or `main_nospeed.py` to test it on the other track.

### Configuration

- `config.ini`: Standard NEAT configuration allowing both steering and speed control.
- `config_nospeed.ini`: Configuration where speed is fixed, and only steering is evolved.

Adjust parameters such as population size, number of generations, and mutation rates within these files to experiment with different evolutionary strategies.
