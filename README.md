Training Command-
python -m app.auto_trainer rm_m7_r3

python -m app.auto_trainer rm_m4_r1

python -m app.auto_trainer test

Testing Command-
python -m app.auto_tester rm_m4_r1

Experiments To Try Out:
hange criterion to something else : any suitable loss in pytorch
Optim : try with Ada
Try with adaptive learning rate
Try with a different Snr
