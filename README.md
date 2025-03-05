# NBsTem Standalone
<p align="center">
    <img align="center" src="https://github.com/jourmore/NBsTem/blob/master/GA.png" width="500" alt="logo"/>
</p>

- This is the official repository of NBsTem_Tm & NBsTem_Q, two deep learning models designed for thermostability prediction of nanobodies (VHH).
- You can also access [NBsTem Webserver](http://www.nbscal.online/) for thermostability prediction online.

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
** Calculating Tm:: 100%|█████████████████████████████████████████████████| 83/83 [00:03<00:00, 22.40it/s]
** Calculating Qclass:: 100%|█████████████████████████████████████████████| 83/83 [00:02<00:00, 33.12it/s]
== 6.Finish ! The results are shown below or you can check file [Tm83.csv]

                    ID         Tm Qclass Specie                                           Sequence
1                 4W70  73.279999      4  Camel  EVQLVESGGGLVQAGDSLRLSATASGRTFSRAVMGWFRQAPGKERE...
2                 5SV3  69.769997      4  Camel  EVQLVESGGGLVQAGDSLRLSCTASGRTLGDYGVAWFRQAPGKERE...
3                  Nb4  63.790001      4  Camel  QVQLVESGGGSVQAGGSLRLSCAASGLDIHSYCMTWFRQAPGKERE...
4                  Nb5  68.080002      3  Camel  QVQLVESGGGSVQAGGSLRLSCAASGSAISNLYMAWFRQAPGKERE...
5                  Nb6  80.320000      3  Camel  HVQLVESGGGSVQAGGSLRLSCEISLYIYSSYCMGWFRQAPGKERE...
..                 ...        ...    ...    ...                                                ...
79  NB-AGT-2-L22A-I72V  67.870003      3  Camel  QVQLVESGGGLVQAGGSLRASCAASGRTFSSYAMGWFRQAPGKERE...
80  NB-AGT-2-L22A-I72A  69.099998      3  Camel  QVQLVESGGGLVQAGGSLRASCAASGRTFSSYAMGWFRQAPGKERE...
81            NB-extra  74.070000      4  Human  EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIGWVRRAPGKGEE...
82      NB-extra-CA-CV  71.000000      4  Human  EVQLVESGGGLVQPGGSLRLSAAASGFNIKDTYIGWVRRAPGKGEE...
83      NB-extra-CA-CA  71.019997      4  Human  EVQLVESGGGLVQPGGSLRLSAAASGFNIKDTYIGWVRRAPGKGEE...

[83 rows x 5 columns]
```

### 3.About models

- **NBsTem_Tm**: A model for predicting the melting temperature (Tm) from experiments (nanoDSF, DSF, DSC and CD, etc.).

- **NBsTem_Q**: A model for predicting a new theoretical indicator (Qclass) proposed by us, which is derived from molecular dynamics simulation.

## Citing this work

```bibtex
@article{...,
    title = {NBsTem: Complementary dual models inferred from experimental and theoretical indicators to realize reliable prediction for nanobody thermostability},
    author = {Jourmore, ..., Xuemei-Pu},
    journal = {Under submission},
    year= {2025}
}
```
