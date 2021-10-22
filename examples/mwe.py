#native imports
import pandas as pd

#package imports
from .submit import FluentBatchSubmission, FluentPBS

"""
Author: Michael Lanahan
Date: 08.06.2021
Last Edit: 09.08.2021

This is a so-called minimal working example showing how this library is meant to be used

Before you begin working through this example ensure the following:

1. you can see the example_inputs.csv file in the same directory as this directory
2. you have the file ICM-11.cas.gz on the cluster
"""

#The first step is to read in the data file of boundary conditions that we want to 
#create a batch submission for. The most important part of this data file is ensuring the columns
#are in the correct format which is 
#name:input:type 
#name - the name of the boundary condition in the fluent file - if this does not match up this will definitly throw an error
#       during fluent execution
#input - the input into the boundary condition i.e. temperature,pressure, with documentation on the allowable inputs
#       provided : NEED TO PROVIDE DOCUMENTATION
#type - the type of boundary condition, in the same format as the fluent tui interface, docuementation provided: NEED TO PROVIDE DOCUMENTATION
#
#The data SHOULD be in the correct format and allow you to run the case, if not contact me at mlanahan3@gatech.edu and I will try and help you debug 

parameter_table = pd.read_csv('example_inputs.csv',index_col= 0)
#The next step is to format our pbs structure for pbs submitting. The two required arguments here
#are the name of the input files to fluent, in this case "fluent_input" - this really doesn't matter 
#and I couldn't find any convention so I am just rolling with this for now - and the working folder we plan to run the
#batch file from.
#All of the keyword arguments specify requested parameters for EACH fluent run
#WALLTIME and MEMORY are REQUIRED keyword arguemnts that specify the walltime expected for EACH run in seconds
#while memoery specifies the requested memory (per node) in GB. N_NODES specifies the requested number of nodes
#and N_PROCESSESORS specifies the requested numbe of processsors - both N_NODES and N_PROCESSORS default to 1
#if they are not specified.
#email will tell pace which email to send updates regarding the status of the job too. If you submit a ton of jobs 
#and don't want your inbox to be full I would suggest not providing this
pbs = FluentPBS('fluent_input',
                WALLTIME = 1000,
                MEMORY = 8,
                N_NODES = 1,
                N_PROCESSORS = 12,
                email = 'mlanahan3@gatech.edu')

#the next step is to format the submission from the parameter_table. This is done through the FluentBatchSubmission class 
#method "from_frame" which takes a case name - in our cause ICM-11.cas.gz, a data frame object defininng the boundary condition parameters,
#and the pbs object we just created. The class turns the dataframe it into a list of FluentSubmission objects which can be submitted on 
#PACE @ GT easily
submission = FluentBatchSubmission.from_frame('ICM-11.cas.gz','ke-standard',parameter_table,pbs)

#we now have the oppurtunity to make the submission via the "batch_submit method". The only required argument here is the folder
#we would like to submit batch jobs from, Make sure this is the same as the folder we are changing too
submission.bash_submit('test-case')

#You will notice that a file batch.sh has been created, along with a folder "test-case". In the folder 
#there are three subfolders, corresponding to the rows of the supplied parameter table. Each of the 
#subfolders contains a fluent input file with the same name as created by the pbs script, and a fluent.pbs file
#which contains instructions for pace. The batch file file submit each of these jobs to the que once you have navigated
#to the cluster and invoked the command 
#>> ./batch.sh
#from the same directory as the directory that contains both ICM-11.cas.gz AND the subfolders that were just created ""
# Structure of the folder should look like this (-> indicates a subdirectory):
# test-case
# -> ICM-11.cas.gz
# -> 0
# -> -> fluent_input
# -> -> fluent.pbs
# -> 1
# -> -> fluent_input
# -> -> fluent.pbs
# -> 1
# -> -> fluent_input
# -> -> fluent.pbs