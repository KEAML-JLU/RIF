---
sort: 3
---

# Annotation transferring

RIF Transfer ensure coherence in annotation of SDI results on target slices with the annotated reference slices. we adopted a transfer learning strategy to address that. To account for batch effects, We deconstruct the gene expression on each slice into two components: gene-specific expression captured by MLP, and spot-specific expression captured by GNN. The sum of these two components constitutes the reconstructed gene expression.

{% include list.liquid %}
