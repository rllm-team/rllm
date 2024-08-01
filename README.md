# rLLM (**[Documentation](https:/)**|**[Paper](https://arxiv.org/abs/2407.20157)**)

**rLLM** (relationLLM) is an easy-to-use Pytorch library for Relational Table Learning (RTL) with LLMs, by performing two key functions:
1. Breaks down state-of-the-art GNNs, LLMs, and TNNs as standardized modules.
2. Facilitates novel model building in a "combine, align, and co-train" way using these modules.  


<p align="center">
  <img width="400" height="310" src="https://zhengwang100.github.io/img/rllm/rllm_overview.png">
</p>

### How to Try:
Let's run a RTL-type method [BRIDGE](./examples/bridge) as an example:

```bash
# cd ./examples
# set parameters if necessary
python bridge.py
```


### Highlight Features: 
- **LLM-friendly:** Modular interface designed for LLM-oriented applications, integrating smoothly with LangChain and Hugging Face transformers.
- **One-Fit-All Potential:**  Processes various graphs (like Social/Citation/E-commerce Networks) by treating them as multiple tables linked by foreigner keys. 
- **Novel Datasets:**  Introduces two new relational table datasets (rllm.datasets) useful for model design. Includes standard tasks like classification and regression, with examples and baseline results.
- **Community Support:**  Maintained by students and teachers from Shanghai Jiao Tong University and Tsinghua University. Supports the SJTU undergraduate course "Content Understanding (NIS4301)" and the graduate course "Social Network Analysis (NIS8023)".


## Citation
```
@article{rllm2024,
      title={rLLM: Relational Table Learning with LLMs}, 
      author={Weichen Li and Xiaotong Huang and Jianwu Zheng and Zheng Wang and Chaokun Wang and Li Pan and Jianhua Li},
      year={2024},
      eprint={2407.20157},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2407.20157}, 
}
```
