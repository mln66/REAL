# REAL


[//]: # (## Describe)

[//]: # (This is the official project page of the paper "Rectifying Adversarial Sample with Low Entropy Prior for Test-Time Defense" )


## Program Running
train
1. python main_rec_ours.py -t False -d mnist -at None -aux rec
2. python cifar10_rec_ours.py -t False -d cifar10 -at None -aux rec 

test
1. python main_rec_ours.py -t True -d mnist -aux rec -shot True -at FGSM -pd 0.1 -ps 0.1 -pni 3
2. python cifar10_rec_ours.py -t True -d cifar10 -aux rec -shot True -at FGSM -pd 4 -ps 4 -pni 3