This GitHub repo contains the material used in the MDM2 data science project of group 4 to predict the popularity of songs.

The files present are a mixture of datasets, Python scripts used to train models and get results and MATLAB scripts to preprocess the data.

The following is a list of the relevant files to understanding the process undertaken in achieving our results and their descriptions.

1. spotify_dataset.xlsx

   Data set containing the original data pulled from Spotify's API.

2. preprocessed.csv

   Data set containing the preprocessed data set from all decades used to train models.

3. recentpreprocessed.csv

   Data set containing the preprocessed data set from the two most recent decades while omitting the 'decades' variable.

5. Preprocessingfile

   MATLAB file used to preprocess file (1.) and output file (2.).

5. Recent_decades.m

   MATLAB file used to preprocess file (1.) and output file (3.).

6. main2.py

   Python script used to build, validate and test models and to produce figrues. This is an earlier version of file (7.) and will plot some of the figures used for validation.

7. main_results.py

   This is similar to file (6.) but uses the final models with tuned hyperparameters to produce results and feature importance plots. There are many  commented sections which contain redundant code but all processes used to finish the project after preprocessing are documented in this script.

   
