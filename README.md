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


5. Generate the features 


## Usage 




## Performance

##TODOs

- add scripts V
- add option for only one step V
- add python functions to convert files and generate dataset from csv of mutations or allow the user to upload the files as they want. 
- finish documentation + do trial with cache already existent 

