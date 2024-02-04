# Avoiding post-processing with event-based detection in biomedical signals 

Code repository accompanying our publication on event-based modeling.


The repo is split among our synthetic data experiments and the real-world data experiments, since both rely on separate dependencies (synthetic uses PyTorch, real-world Tensorflow). The supplementary material can be found in `supp.pdf`.

## Synthetic data
The files `event_train_and_evaluation.py` and `epoch_train_and_evaluate.py` run the event-based and epoch-based training runs respectively. The epoch-based post-processing can be set using command line arguments.


## Real-world data
EEG artefact training runs can be started with `run_event_artefact.py` and `run_epoch_artefact.py`. One can choose how many signal recordings each run trains on.

EEG seizures training runs can be started with `run_event_seizure.py` and `run_epoch_seizure.py`. One can choose how many signal recordings each run trains on.
