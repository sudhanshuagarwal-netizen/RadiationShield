# RadiationShield
Radiation-Resilient Neural Networks for Space-Based Data Centers

Building AI models that can survive cosmic radiation-induced bit flips — without relying on expensive radiation-hardened hardware.

## Why This Matters
Space radiation causes random bit flips in memory and compute. Neural networks are extremely sensitive to these errors. This project simulates the problem and develops software techniques to make models robust.

## Project Goals
- Train a strong baseline CNN on FashionMNIST (~92-93% accuracy)
- Simulate realistic radiation effects
- Test multiple robustness techniques (noise injection, redundancy, quantization, etc.)

## Tech Stack
- Python
- PyTorch + torchvision
- Matplotlib, Seaborn, Pandas

## Milestones
- [ ] Milestone 0: Project Setup + Baseline Model
- [ ] Milestone 1: Radiation Simulator
...

## How to Run
```bash
source .venv/bin/activate
pip install -r requirements.txt