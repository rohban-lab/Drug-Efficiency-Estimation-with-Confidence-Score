A luigi pipeline to analysis RxRx19a dataset with cellprofiler

Before running make sure that data paths data_paths.py are assigned correctly also set number of plates in each experiment in config.py EXPERIMENT_PLATES dictionary. 

To run a experiment:
> python experiment.py --expr 'VERO-1'

To see dependency tree:
> PYTHONPATH=../:. luigi-deps-tree --module tasks.profile_aggregator PlateProfileAggregator --expr 'HRCE-1' --plate 1

To run each plate manualy:
Calculate illumination function
> PYTHONPATH=. python tasks/illumination_function_calculator.py --expr 'HRCE-1' --plate 1

Calculate cellprofiler features
> PYTHONPATH=. python tasks/feature_calculator.py --expr 'HRCE-1' --plate 1

Aggregate profile
> PYTHONPATH=. python tasks/profile_aggregator.py --expr 'HRCE-1' --plate 1

To access the Luigi scheduler interface:
> luigid --background --port 8082 --logdir .log --state-path .pickle

and then open http://localhost:8082/
