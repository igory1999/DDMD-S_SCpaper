export PYTHONPATH=../Outlier_search/:../CVAE_exps/:../misc/:$PYTHONPATH

python adios2pandasR4b.py ../aggregate/aggregator0.bp 40 > 0.out 2>0.err
python adios2pandasR4b.py ../aggregate/aggregator1.bp 40 > 1.out 2>1.err
python adios2pandasR4b.py ../aggregate/aggregator2.bp 40 > 2.out 2>2.err
python adios2pandasR4b.py ../aggregate/aggregator3.bp 40 > 3.out 2>3.err
python adios2pandasR4b.py ../aggregate/aggregator4.bp 40 > 4.out 2>4.err
python adios2pandasR4b.py ../aggregate/aggregator5.bp 40 > 5.out 2>5.err
python adios2pandasR4b.py ../aggregate/aggregator6.bp 40 > 6.out 2>6.err
python adios2pandasR4b.py ../aggregate/aggregator7.bp 40 > 7.out 2>7.err
python adios2pandasR4b.py ../aggregate/aggregator8.bp 40 > 8.out 2>8.err
python adios2pandasR4b.py ../aggregate/aggregator9.bp 40 > 9.out 2>9.err
