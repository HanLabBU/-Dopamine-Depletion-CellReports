This repository includs all the code needed to generate the figures for the menuscript:
"Dopamine Depletion Selectively Disrupts Interactions Between Striatal Neuron
Subtypes and LFP Oscillations" at Cell Reports.

This repo contain 2 yml conda enviroment files - 
6OHDA.yml used on our windows machines and 
env6OHDA.yml that was used on our linux machines. 

Step 1) process the data
Step 2) Run "repopulate-onsetAndPeriods-file" notebook
Step 3) Run "Final-Submission-python" for all figures that were generated using Python. 
Step 4) All pair plots were generated using R ggplot package. 
	The R shiney apps (stats.R and moreStats.R) save the plots for pre-post L-Dopa
	And Pre-post Amphetamine automatically when you check their statistics. 
	For figure 6 and figure 6S use file freq_mvmt_plot.R with the desiered 
	frequency and mvmt bouts. 