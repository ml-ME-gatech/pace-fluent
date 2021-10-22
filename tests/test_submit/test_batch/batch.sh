echo "executing job located in folder: 0"
cd 0
qsub fluent.pbs
cd ..
echo "executing job located in folder: 1"
cd 1
qsub fluent.pbs
cd ..
