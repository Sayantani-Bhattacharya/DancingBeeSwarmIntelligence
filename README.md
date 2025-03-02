# The Dancing Bee Swarm Intelligence
<p align="center">
  <img src="/images/bee.png" alt="Alt text" width="250"/>
</p>

## Objective and Motivation: 

The project aims to simulate and learn the waggle dance communication method used by bees for efficient
foraging. The waggle dance encodes directional and distance information to food sources, helping the colony optimize energy
expenditure and resource gathering. This problem is interesting to me because it models emergent swarm intelligence, which is
applicable in real-world scenarios such as multi-robot exploration (e.g., drones, extra-terrestrial rovers or underwater robots operating
in environments with limited direct communication). By implementing multi-agent reinforcement learning (MARL), we could explore
how artificial agents can develop a communication protocol without explicit direct messaging, learning from movement-based cues
alone.


## Installation

1. Create a Python virtual environment:
```bash
python3 -m venv venv 
source venv/bin/activate
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Troubleshooting: If you face numpy issues, reinstall and try again.
```bash
pip install --force-reinstall numpy
```

## Project Structure
- [`multi_model_eval.py`](multi_model_eval.py): Evaluation script for trained RL models
- [`multi_model_train.py`](multi_model_train.py): Training script for multi-agent shepherding models
- [`envEmulator.py`](envEmulator.py): Custom gymnasium environment for sheep herding simulation
- [`robotEmulator.py`](robotEmulator.py): Dynamics definition for the differential drive robots used in the simulation. 
