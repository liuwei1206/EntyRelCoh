# Joint Modeling of Entities and Discourse Relations for Coherence Assessment
Code for the EMNLP2025 paper "Joint Modeling of Entities and Discourse Relations for Coherence Assessment".

If any questions, please contact the email: willie1206@163.com

## 1 Requirement
For the **fusion** part, please refer to the requirements in [RelCoh](https://github.com/liuwei1206/RelCoh).

For the **prompt** part, please refer to the requirements of [LLaMAFactory](https://github.com/hiyouga/LLaMA-Factory) (but not the latest version).

## 2 Run
To run experiments, you should:
1. Preprocess the data;
2. Put it under the fold "fusion/data/dataset/" or "prompt/data/".
3. Call the script.

## 3 Citation
```
@inproceedings{liu-strube-2025-joint,
    title = "Joint Modeling of Entities and Discourse Relations for Coherence Assessment",
    author = "Liu, Wei  and
      Strube, Michael",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1113/",
    doi = "10.18653/v1/2025.emnlp-main.1113",
    pages = "21910--21926",
    ISBN = "979-8-89176-332-6",
    abstract = "In linguistics, coherence can be achieved by different means, such as by maintaining reference to the same set of entities across sentences and by establishing discourse relations between them. However, most existing work on coherence modeling focuses exclusively on either entity features or discourse relation features, with little attention given to combining the two. In this study, we explore two methods for jointly modeling entities and discourse relations for coherence assessment. Experiments on three benchmark datasets show that integrating both types of features significantly enhances the performance of coherence models, highlighting the benefits of modeling both simultaneously for coherence evaluation."
}
```
