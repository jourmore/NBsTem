# NBsTem local

- Official repository for NBsTem (NBsTem_Tm & NBsTem_Q), deep learning models for nanobody thermostability prediction, as described in [NBsTem paper](http://www.nbscal.online/).
- You can also access [NBsTem Webserver](http://www.nbscal.online/) for nanobody thermostability prediction online.

## 1.Setup

Clone this repository and install the package locally:
```bash
$ git clone git@github.com:jourmore/NBsTem.git
$ cd NBsTem_local
$ pip install -r requirements.txt
```

## 2.Usage

```bash
python app.py -i in.fasta
python app.py -t QVQLVESGGGSVQAGGSLRLSCAASGYTVSTYCMGWFRQAPGKEREGVATILGGSTYYGDSVKGRFTISQDNAKNTVYLQMNSLKPEDTAIYYCAGSTVASTGWCSRLRPYDYHYRGQGTQVTVSS
```

```bash
*usage: python app.py [-h] [-i I] [-o O] [-t T] [-seed SEED] [-device DEVICE]

optional arguments:
  -h, --help      show this help message and exit
  -i I            Input path with fasta format. [Such as: ./in.fasta]
  -o O            Output file name when input is fasta format. [Default: "Output-NBsTem-[Year]-[Month]-[Day].csv"
  -t T            Input one sequecne with text format. [Default:
                  QVQLVESGGGSVQAGGSLRLSCAASGYTVSTYCMGWFRQAPGKEREGVATILGGSTYYGDSVKGRFTISQDNAKNTVYLQMNSLKPEDTAIYYCAGSTVASTGWCSRLRPYDYHYRGQGTQVTVSS]
  -seed SEED      Random seed for torch, numpy, os. [Default: 42]
  -device DEVICE  Device: cpu, cuda. [Default: auto]
```

### Example

- Example (Using default parameters and example sequences):

```bash
python app.py
```

- Terminal output message:

```bash
******************************************************************
**                                                              **
**  NBsTem v.2025 Thermostability prediction for Nanobody/VHH.  **
**                                                              **
**                  http://www.nbscal.online/                   **
**                    maojun@stu.scu.edu.cn                     **
******************************************************************

== 1.Use seed: 42
== 2.Device: cuda
== 3.Loading antibody language model: AntiBERTy
== 5.Begin to predict: Tm, Qclass, Specie and Chain
** Calculating Specie and Chain [Fast]
** Calculating Tm:: 100%|██████████████████████████████████████
** Calculating Qclass:: 100%|██████████████████████████████████
== 6.Finish ! The results are shown below or you can check file [Tm1.csv]

           ID         Tm Qclass Specie                                           Sequence
1        seq1  53.900002      3  Camel  GGGGLVQAGGSLRLSCAASGRTFYNYAMGWFRQAPGKEREFVAAIS...
2        seq2  56.130001      3  Camel  GGGGLVQAGGSLRLSCAASGRTLYNYAMGWFRQAPGKEREFVAAIS...
3        seq3  55.759998      3  Camel  GGGGLVQAGGSLRLSCAASGPTFYNYAMGWFRQAPGKEREFVAAIS...
4        seq4  55.709999      3  Camel  GGGLVQAGGSLRLSCAASGPTFYNYAMGWFRQAPGKEREFVAAISW...
5        seq5  75.879997      4  Camel  RSQFVESGGGLVQPGGSLRLSCTASGFSLKYWAVGWFRQAPGKERE...
...       ...        ...    ...    ...                                                ...
4996  seq4996  71.300003      3  Camel  EVQLVESGGDLVQPGGSLRLSCAASGSIFSINDMGWFRQAPGKQRE...
4997  seq4997  53.419998      3  Camel  QVQLQESGGGLVQAGGSLRLSCAASGRTFSSHAMAWFRQGPGEERQ...
4998  seq4998  72.000000      1  Camel  EVQLQESGGGLVQAGGSLRLSCAASGRTFSIYTIGWFRQAPGKERE...
4999  seq4999  67.970001      4  Camel  QVQLQESGGGSVQDGGSLTLSCAASSSYVFNNLNMGWFRQAPGKEC...
5000  seq5000  62.330002      3  Camel  QVKLEESGGGSAQTGGSLRLTCAASGRTSRSYGMGWFRQAPGKERE...

[5000 rows x 5 columns]
```

### 3.About models

- A general framework, consisting of two core components: (1) Sequence Encoding Module: Leveraging the advanced antibody language model AntiBERTy to generate sequence representations. (2) Downstream Training Module: Implementing MS-ResLSTM – a novel fusion architecture integrating a multi-scale residual network (MS-ResNet) with bidirectional long short-term memory (Bi-LSTM) as the computational unit.

- **NBsTem_Tm**: The melting temperature (Tm) from experiments (nanoDSF, DSF, DSC and CD, etc.).

- **NBsTem_Q**: A new theoretical indicator proposed by us is derived from Q-values of molecular dynamics (MD) trajectories (Qclass).

## Citing this work

```bibtex
@article{...,
    title = {NBsTem: ...},
    author = {Jourmore...},
    journal = {...},
    year= {2024}
}
```
