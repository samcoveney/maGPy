# TODO

## general
* check how the nugget should be used for prediction vs estimation with the posterior
* how to ensure all setup is done correctly? Especialy important if we re-setup Data to setup the GP and Basis again, it seems
* better formatting for printing fitting and sensitivity - too messy to follow at the moment (possibly full results to a log file, less results on screen)

## history matching
* have an alternative mincount for imp plots to remove isolated blobs?
* change how NROY plot works so that it isn't just a density plot? Would need to stay imp...

## sensitivity
* Since results can be pickled, need to have a routine to reprint results (rather than recalc)
* We also have calculated the mean effect... could have option to plot this
* In Sensitivity, the indexes are all local rather than global. Probably fine, but might be nice to have a print of the global indices instead

## examples
* Need a History Matching example

