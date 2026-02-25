# HPC usage by LL

This is Léonard Lebrun's personal space for Unige's Baobab HPC facility usage.

## Launch interactive session
To access a compute node interactively :
- Add this line to your .bashrc :
```alias launch_interactive='salloc --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=32G --time=06:00:00 --partition=shared-cpu'```
- ```source .bashrc``` 
- Run command :
```launch_interactive```

This will log you in a cpu node. You might adapt the launch_interactive command to ask for gpu nodes.

## Launch jupyter server

To launch a jupyter server **from the compute node**, execute the `start_jupyter.sh` script.
You can then run notebooks under `~/notebooks` from e.g. vscode by choosing the appropriate kernel.

## Get data

This code was created to analyse data 

