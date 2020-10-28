import os
import errno
import optuna
import gcn_config as cfg

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
        
class EarlyStopping(optuna.exceptions.OptunaError):
    pass

def callback(study,trial):
    since_best = trial.number - study.best_trial.number
    if since_best == cfg.study_patience:
        raise EarlyStopping