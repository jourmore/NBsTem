# NBsTem local
- You can also access [NBsTem Webserver](http://www.nbscal.online/) for more.

- Official repository for NBsTem (NBsTem_Tm & NBsTem_Q), deep learning models for nanobody thermostability prediction, as described in [NBsTem Webserver](http://www.nbscal.online/) and [NBsTem paper](http://www.nbscal.online/).

- Since the nanobody Tm prediction task, NBsTem Webserver deployed the [AntiBERTy+CNN] model.
- Therefore, we provide the source code of running the NBsTem_Tm[ProtT5+CNN] model here.

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
**  NBsTem v.2024 Thermostability prediction for Nanobody/VHH.  **
**                                                              **
**                  http://www.nbscal.online/                   **
**                    maojun@stu.scu.edu.cn                     **
******************************************************************

*./Rostlab/prot_t5_xl_uniref50 exists, and it will be automatically loaded.

== 1.Use seed: 42
== 2.Device: cuda
== 3.Loading antibody language model: AntiBERTy + MS-ResLSTM
== 4.Loading protein language model: ProtT5_XL_UniRef50 + CNN
== 5.Begin to predict: Tm, Qclass, Specie and Chain
** Calculating Specie and Chain [Fast]
** Calculating Tm:: 100%|█████████████████████| 1/1 [00:01<00:00,  1.48s/it]
** Calculating Qclass:: 100%|█████████████████| 1/1 [00:00<00:00,  2.54it/s]
== 6.Finish ! The results are shown below or you can check file [NBsTem-2024-12-13.csv]

         ID     Tm Qclass Specie                                           Sequence
1  Nanobody  67.32      4  Camel  QVQLVESGGGSVQAGGSLRLSCAASGYTVSTYCMGWFRQAPGKERE...
2  Nanobody  67.32      4  Camel  QVQLVESGGGSVQAGGSLRLSCAASGYTVSTYCMGWFRQAPGKERE...
...
```

### 3.About models

- **NBsTem_Tm**: To use [ProtT5_XL_UniRef50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50) to generate sequence embeddings, and CNN deep learning framework to training model.

- **NBsTem_Q**: To use [AntiBERTy](https://github.com/jeffreyruffolo/AntiBERTy) to generate sequence embeddings, and MS-ResLSTM deep learning framework to training model.

## Citing this work

```bibtex
@article{...,
    title = {NBsTem: ...},
    author = {Jourmore...},
    journal = {...},
    year= {2024}
}
```