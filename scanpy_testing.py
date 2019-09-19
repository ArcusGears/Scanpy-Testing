import numpy as np
import pandas as pd
import scanpy as sc
import loompy
import pathlib as pa
import os
#import anndata2ri


#CLUSTERING TUTORIAL

#preprocessing and setting up
sc.settings.verbosity = 3
sc.logging.print_versions()
results_file = pa.Path('./write/Combined.h5ad')  #file that will store analysis
i = 0
while os.path.exists('./write/figures%s.png' % i):
	i += 1
figure_file = open('figures%s.png' % i, 'w')

sc.settings.set_figure_params(dpi = 80)

adata = sc.read_loom('PC9Combined.loom', var_names = 'gene_symbols', sparse = True, cleanup = False, X_name = 'spliced', obs_names = 'CellID')

adata.var_names_make_unique()  #unnecessary if using 'gene_ids'

#returns an AnnData object with variable gene_ids
adata

#show genes that yeild highest fraction of counts in each single cells across all cells
sc.pl.highest_expr_genes(adata, n_top = 20)

#basic filtering to remove genes that are detected in less than 3 cells
sc.pp.filter_cells(adata, min_genes = 200)
sc.pp.filter_genes(adata, min_cells = 3)

#computing the fraction of mitochondrial genes and additional measures
mito_genes = adata.var_names.str.startswith('MT-')
#for each cell compute fraction of counts in mito genes vs all genes
#the .A1 is only necessary as X is sparse (to transform to a dense array after summing)
adata.obs['percent_mito'] = np.sum (
	adata[:, mito_genes].X, axis = 1).A1 / np.sum(adata.X, axis = 1).A1
#add total counts per cell as observations/annotations to adata
adata.obs['n_counts'] = adata.X.sum(axis = 1).A1

#make a violin plot
sc.pl.violin(adata, ['n_genes', 'n_counts', 'percent_mito'], jitter = 0.4, multi_panel = True)

#removing cells that have too many mitochondrial genes expressed or too many total counts
sc.pl.scatter(adata, x = 'n_counts', y = 'percent_mito')
sc.pl.scatter(adata, x = 'n_counts', y = 'n_genes')

adata

#filtering
adata = adata[adata.obs['n_genes'] < 2500, :]
adata = adata[adata.obs['percent_mito'] < 0.05, :]

#normalize cell data
sc.pp.normalize_per_cell(adata, counts_per_cell_after = 1e4)

#logarithmize the data
sc.pp.log1p(adata)

#setting raw attribute of adata object to loarithmized raw gene expressions.  Freezes the state of the adata object.  
adata.raw = adata

#identifying highly variable genes
sc.pp.highly_variable_genes(adata, min_mean = 0.0125, max_mean = 3, min_disp = 0.5)

#plots
sc.pl.highly_variable_genes(adata)


#filters for highly variable genes
adata = adata[:, adata.var['highly_variable']]

#regresses out effects of total counts per cell and % of migochondrial genes expressed.  Scales data to unit variance
sc.pp.regress_out(adata, ['n_counts', 'percent_mito'])

#scale each gene to unit variance.  Clip values to 10
sc.pp.scale(adata, max_value = 10)


#Principal Component Analysis

sc.tl.pca(adata, svd_solver = 'arpack')

#scatter plot in PCA coordinates
sc.pl.pca(adata, color = 'CST3')

#Contribution of single PCs to total variance
#Consider how many PCs we should consider in order to computer the neighborhood relations of cells
sc.pl.pca_variance_ratio(adata, log = True)

#save result
adata.write(results_file)

adata


#Neighborhood Graph

#computing
#consider changing to default valies or according to the PCs
sc.pp.neighbors(adata, n_neighbors = 10, n_pcs = 40)

#Embedding neighborhood graph

#Note:  If disconnected clusters and other connectivity violations appear consider running the commented out code below
#tl.paga(adata)
#pl.paga(adata, plot = False) #remove if you want to see coarse-grained graph
#tl.umap(adata, init_pos = 'paga')

sc.tl.umap(adata)
sc.pl.umap(adata, color = ['CST3', 'NKG7', 'PPBP'])

#plot not using the .raw
sc.pl.umap(adata, color = ['SCT3', 'NKG7', 'PPBP'], use_raw = False)


#clustering the graph
#uses Louvain graph-clustered method
sc.tl.louvain(adata)

#plot
sc.pl.umap(adata, color = ['louvain', 'CST3', 'NKG7'])

#save
adata.write(results_file)


#Finding Marker Genes

#Ranking for the highly differential genes in each cluster
#Uses the .raw attribute
sc.tl.rank_genes_groups(adata, 'louvain', method = 't-test')
sc.pl.rank_genes_groups(adata, n_genes = 25, sharey = False)

#reduce verbosity
sc.settings.verbosity = 2

#wilcox test version
sc.tl.rank_genes_groups(adata, 'louvain', method = 'wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes = 25, sharey = False)

#save result
adata.write(results_file)

#logistic regression ranking
sc.tl.rank_genes_groups(adata, 'louvain', method = 'logreg')
sc.pl.rank_genes_groups(adata, n_genes = 25, sharey = False)

#list of marker genes
#currently is the default list from the tutorial, consider changing
marker_genes = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14', 'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1', 'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP']

#reload object that was saved with the wilcox test result
adata = sc.read(results_file)

#show top 10 ranked genes per cluster in a dataframe
pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(5)

#table with scores and groups
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
pd.DataFrame(
	{group + '_' + key[:1]: result[key][group]
	for group in groups for key in ['names', 'pvals']}).head(5)

#compare to a single cluster
sc.tl.rank_genes_groups(adata, 'louvain', groups = ['0'], refrence = '1', method = 'wilcoxon')
sc.pl.rank_genes_groups(adata, groups = ['0'], n_genes = 20)

#violin plot for more detailed view
sc.pl.rank_genes_groups_violin(adata, groups = '0', n_genes = 8)

#reload object with computed diff expression
adata = sc.read(results_file)
sc.pl.rank_genes_groups_violin(adata, groups = '0', n_genes = 8)

#How to compare a certain gene across groups
sc.pl.violin(adata, ['CST3', 'NKG7', 'PPBP'], groupby = 'louvain')

#mark the cell types
#new_cluster_names = [
#	'CD4 T', 'CD14+ Monocytes',
#	'B', 'CD8 T',
#	'NK', 'FCGR3A+ Monocytes',
#	'Dendritic', 'Megakaryocytes']
#adata.rename_categories('louvain', new_cluster_names)
#sc.pl.umap(adata, color = 'louvain', legend_loc = 'on data', title = '', frameon = False, save = '.pdf')

#visualize marker genes
ax = sc.pl.dotplot(adata, marker_genes, groupby = 'louvain')

#compact violin plot
ax = sc.pl.stacked_violin(adata, marker_genes, groupby = 'louvain', rotation = 90)

#view adata and it's annotations
adata

#write to results file
adata.write(results_file, compression = 'gzip')