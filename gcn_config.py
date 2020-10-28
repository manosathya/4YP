#!/usr/bin/python
adj_setup = [["audio", 200, 40, "B", 0],
              ["visual",200,60, "B", 0],
              ["mm-audio",500,40, "B", 0],
              ["mm-visual",500,60, "B", 0]]
#0:mode 1:K 2:R 3:Threshold Type 4:Threshhold

num_epochs = 300

save_train = "n"
save_test = "n"
save_study = "y"

patience = 30

num_trials = 500
study_patience = 200

