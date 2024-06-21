bsub -W 10 -n 128 -R "span[ptile=16]"  -q cpuII -o ScLETD.out -e ScLETD.err  mpijob.intelmpi ./bin/ScLETD 64
