# C2C Power Quality Analysis

This repository contains a Python script for analysing large-scale power quality (PQ) monitoring data captured during the [C2C project](http://www.enwl.co.uk/c2c), led by Electricity North West Limited. The script has been used to generate results in the [final Strathclyde report](http://strathprints.strath.ac.uk/54345/1/Blair_Booth_2014_Analysis_of_the_technical_performance_of_C2C_operation.pdf).

The data required to run the analysis is available here: [https://strathcloud.sharefile.eu/share?cmd=d&id=s54bf049e5e142b0a#/view/s54bf049e5e142b0a?_k=jxa57e]. After cloning the repository, put these PyTables (.h5) files in the `data` directory.

The function `validate_freq_sync()` performs time synchronisation of the PQ data using correlation of measured frequency trends.

The pre-generated outputs from the script are provided in the `plots`, `circuit_plots`, and 'csv_data' directories.