# C2C Power Quality Analysis

This repository contains a Python script for analysing large-scale power quality (PQ) monitoring data captured during the [C2C project](http://www.enwl.co.uk/c2c), led by Electricity North West Limited. The script has been used to generate results in the [final Strathclyde report](http://strathprints.strath.ac.uk/54345/8/Blair_Booth_2014_Analysis_of_the_technical_performance_of_C2C_operation.pdf).

The data required to run the analysis is available here: https://strathcloud.sharefile.eu/share?cmd=d&id=s54bf049e5e142b0a#/view/s54bf049e5e142b0a?_k=jxa57e. After cloning the repository, put these PyTables (.h5) files in the `data` directory.

The function [`validate_freq_sync()`](https://github.com/stevenblair/c2c_pq_analysis/blob/master/pq_analysis.py#L485) performs time synchronisation of the PQ data using correlation of measured frequency trends. The device used to demonstrate this technique is the PQube 2. As described on the vendor's specification (see http://www.powersensorsltd.com/PQube.php#specs), the frequency is measured on a cycle-by-cycle basis using zero-crossing detection. The voltage on phase A (or phase B) is used to measure frequency, and it is sampled at 256 samples per cycle. This data is aggregated to provide 1-minute and 5-minute average, minimum, and maximum values.

The pre-generated outputs from the script are provided in the `plots`, `circuit_plots`, and `csv_data` directories.
