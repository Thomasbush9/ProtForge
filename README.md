# ProtForge



The repository offers the possibility to generate data for molecular biology project efficiently on the Kempner cluster. 


## Installation 

To start a new project follow the next steps: 

1. Clone this repository on your home: 

```{bash}
git clone https://

```

2. Install the tools needed, by running: 

```{bash}
bash download_tools.sh /path/to/dir tool1 tool2 
```

Currently the tools supported are: 

3. Upload and format your data: use `bash_scripts/generate_data.sh --data <table.tsv|.csv> --original <reference.fasta>` (table must have column `aaMutations`; optional `--subsample N`). Set `input.fasta_dir` in config to the printed path.



4. Write your own config file: 

Inside the ```config.yaml` you can modify the config that will be used to generate the features. 

The main parameters that you can modify are: 




5. Generate the features 

After you have modified the configuration file, you can launch the features generation by using the ```run.sh` script. It will automatically launch all the slurm jobs and checkers to generate the features. 



## Usage 




## Performance

##TODOs

- add scripts V
- add option for only one step V
- add python functions to convert files and generate dataset from csv of mutations or allow the user to upload the files as they want. V
- currently it works with msa + boltz
- solve issue with the ESM and ES, envs etc. make it constant 
- Solve issue with the standard installation of ESM-> find the changes that you made and save to apply them to the standard model 
- finish documentation + do trial with cache already existent 

