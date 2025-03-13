# Optical Networking Gym: an open-source toolkit for resource assignment problems in optical networks

## Citing the work

You can cite using the following bibtex:

```bibtex
@ARTICLE{Natalino_2024_gym,
  author={Natalino, Carlos and Magalhaes, Talles and Arpanaei, Farhad and Lobato, Fabricio R. L. and Costa, Joao C. W. A. and Hernandez, Jose Alberto and Monti, Paolo},
  journal={Journal of Optical Communications and Networking}, 
  title={{Optical Networking Gym}: an open-source toolkit for resource assignment problems in optical networks}, 
  year={2024},
  volume={16},
  number={12},
  pages={G40-G51},
  doi={10.1364/JOCN.532850},
}
```

## Reproducing the results

### Launch power optimization

To reproduce the results from Sec. 4.A of the paper, use the following command.
Make sure to fine tune the number of threads (the `th` parameter) to a reasonable value depending on the computer you are running it in.

```bash
python examples/JOCN_Benchmark_2024/graph_launch_power.py -t nobel-eu.xml -e 1000 -s 1000 -l 210 -th 1
```

### Benchmarking QoT-Aware Dynamic RMSA Algorithms

```bash
python examples/JOCN_Benchmark_2024/graph_load.py -t nobel-eu.xml -e 1000 -s 1000 -th 1
```

### Impact of margin

```bash
python examples/JOCN_Benchmark_2024/graph_margin.py -t nobel-eu.xml -e 1000 -s 1000 -l 210 -th 1
```