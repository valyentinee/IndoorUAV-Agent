#!/bin/bash

gnome-terminal -- bash -c "conda activate uv && python online_eval/vla_eval/model_runner.py; exec bash"

sleep 12

gnome-terminal -- bash -c "conda activate habitat && python online_eval/vla_eval/sim_runner.py; exec bash"
# if the sim_runner.py run into problem, running this command before might be helpful: Xvfb :99 -screen 0 1024x768x24 & export DISPLAY=:99

sleep 1

gnome-terminal -- bash -c "conda activate habitat && python online_eval/vla_eval/vla_controller.py; exec bash"
