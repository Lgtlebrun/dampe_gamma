# HPC usage by LL

This is Léonard Lebrun's personal space for Unige's Baobab HPC facility usage.

## Launch interactive session
To access a compute node interactively, source .bashrc if needed then run command :
`launch_interactive`

This will log you in a cpu node. Maybe later we shall add an option for gpu nodes.

## Launch jupyter server

To launch a jupyter server **from the compute node**, execute the `start_jupyter.sh` script.
You can then run notebooks under `~/notebooks` from e.g. vscode by choosing the appropriate kernel.

Ctrl + I in terminal to open copilot