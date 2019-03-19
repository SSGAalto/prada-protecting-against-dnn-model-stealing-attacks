# PRADA: Protecting Against DNN Model Stealing Attacks

This repo contains code that allows you to easily integrate the model stealing defense introduced in _**PRADA: Protecting Against DNN Model Stealing Attacks**_ paper and presented at _**EuroS&P 2019**_. It consists of a) a self-contained defense agent b) a small wrapper that allows you to query the model (through the defense agent). [Link to the arxiv version.](https://arxiv.org/abs/1805.02628v4)

## Requirements

- `Python3`
- `pytorch`
- `torchvision`
- `numpy`
- `scipy`
- `matplotlib`
- `flask`
- `requests`

## Usage

- Interactive querying mode: `python main.py`.

- Provide path to the importable `pytorch` model.

- Simple post client included for the interactive mode: `python client.py server_url image_file` by default model is served at `http://localhost:8080/predict`.

Hence an example query: `py client.py http://localhost:8080/predict cat.ppm`

- Code contains additional comments for running the experiment with your model and data
