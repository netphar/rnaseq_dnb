if __name__ == "__main__":
	import os
	import pandas as pd
	import numpy as np
	import typing
	import matplotlib.pyplot as plt
	from scipy.stats import pearsonr
	from joblib import Parallel, delayed

def calculate_entropy_faster(central_gene: str, local_network: list[str], df_input: pd.DataFrame, refcols: list[str], sample_colname: str='') -> float:
    if not sample_colname:
        cols_idx = df_input.columns.isin(refcols)
    else:
        cols = refcols + [sample_colname]
        cols_idx = df_input.columns.isin(cols)
    
    ar_central = df_input.loc[central_gene,cols_idx].values
    ar_local = df_input.loc[local_network,cols_idx].values
    ar_corrs = abs(np.apply_along_axis(func1d=pearsonr, axis=1, arr=ar_local, y=ar_central)[:, 0])
    ar_p = ar_corrs/np.sum(ar_corrs)
    entropy = -1.0*np.sum(ar_p * np.log2(ar_p))/ar_local.shape[0]
    return entropy
    
def dotproduct(x,y):
    a=np.asarray(x)
    b=np.asarray(y)
    #return(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    return(np.dot(a,b))
    
def mi(x,y):
	'''
	https://www.biorxiv.org/content/10.1101/2024.05.09.593467v1.full.pdf
	'''
	a=np.asarray(x)
	b=np.asarray(y)
	r = pearsonr(a,b)[0]
	return(-np.log((1-r**2))/2)
    
def calculate_entropy_alt(central_gene: str, local_network: list[str], df_input: pd.DataFrame, refcols: list[str], sample_colname: str='') -> float:
    if not sample_colname:
        cols_idx = df_input.columns.isin(refcols)
    else:
        cols = refcols + [sample_colname]
        cols_idx = df_input.columns.isin(cols)
    
    ar_central = df_input.loc[central_gene,cols_idx].values
    ar_local = df_input.loc[local_network,cols_idx].values
    ar_corrs = np.apply_along_axis(func1d=pearsonr, axis=1, arr=ar_local, y=ar_central)
    mi = -0.5*np.log2(1-ar_corrs[:,0]**2) # gaussian mutual information
    ar_p = mi/np.sum(mi)
    entropy = -1.0*np.sum(ar_p * np.log2(ar_p))/ar_local.shape[0]
    return entropy

def calculate_entropy_mi(central_gene: str, local_network: list[str], df_input: pd.DataFrame, refcols: list[str], sample_colname: str='') -> float:
    if not sample_colname:
        cols_idx = df_input.columns.isin(refcols)
    else:
        cols = refcols + [sample_colname]
        cols_idx = df_input.columns.isin(cols)
    
    ar_central = df_input.loc[central_gene,cols_idx].values
    ar_local = df_input.loc[local_network,cols_idx].values
    ar_corrs = np.apply_along_axis(func1d=mi, axis=1, arr=ar_local, y=ar_central)
    ar_p = ar_corrs/np.sum(ar_corrs)
    ar_mi = np.sum(ar_p)/ar_local.shape[0]
    return ar_mi

def transform_df(df_input: pd.DataFrame, method: str) -> pd.DataFrame:
    if method=='log10':
        df_out = np.log10(df_input+1)
    elif method=='ln':
        df_out = np.log1p(df_input)
    elif method=='zeromean':
        df_out = pd.DataFrame((df_input.values - df_input.values.mean(axis=0)) / df_input.values.std(axis=0), columns=df_input.columns, index=df_input.index)
    else:
        pass
    return df_out
    
def cleanexp_byquant(df_input: pd.DataFrame, quantile: float=0.1) -> list[str]:
    ar_vars = np.var(df_input, axis=1)
    #idx_sort = np.argsort(vars)
    qval = np.quantile(ar_vars, q=quantile, method='interpolated_inverted_cdf')
    return(ar_vars[ar_vars>qval].index.tolist())
    
def reduce_ppi(df_ppi: pd.DataFrame, index_genes_from_df: pd.DataFrame.index) -> pd.DataFrame:
    '''
    cleanup input ppi using list of genes found in df rnaseq, provided that genes are used as df index
    clean up filtering columns and removing self-loops
    '''
    ppi_reduced = df_ppi[df_ppi.InteractorA.isin(index_genes_from_df)].reset_index(drop=True)
    ppi_reduced = ppi_reduced[ppi_reduced.InteractorB.isin(index_genes_from_df)].reset_index(drop=True)
    ppi_reduced = ppi_reduced.loc[ppi_reduced[ppi_reduced.InteractorA != ppi_reduced.InteractorB].index, :] # remove self-loops
    return ppi_reduced
    
def filter_dfgenes_byppi(df_ppi: pd.DataFrame, index_genes_from_df: pd.DataFrame.index) -> pd.DataFrame.index:
    '''
    leave only genes found in rnaseq df.index, provided that genes are used as df.index
    '''
    unique_genes = np.unique(df_ppi.values.reshape(-1))
    idxar_bool = index_genes_from_df.isin(unique_genes)
    #assert idxar_bool.shape[0] == 
    return idxar_bool
    
def get_neighbours(genename_ensembl: str, df_ppi: pd.DataFrame) -> list[str]:
    n1 = df_ppi.loc[ df_ppi.InteractorA==genename_ensembl,'InteractorB'].values.tolist()
    n2 = df_ppi.loc[ df_ppi.InteractorB==genename_ensembl,'InteractorA'].values.tolist()
    #n = np.unique(np.concatenate((n1,n2))).tolist()
    n = list( set(n1+n2) )
    return n
    
def get_onehop(geneset: pd.DataFrame.index, df_ppi: pd.DataFrame) -> dict:
    return {k:get_neighbours(k, df_ppi) for k in geneset}

def calculate_diffsd(central_gene: str, df_input: pd.DataFrame, refcols: list, sample_colname: str) -> float:
    
    cols_idx = df_input.columns.isin(refcols)
    sd_ref = df_input.loc[central_gene, df_input.columns.isin(refcols)].std()
    
    cols = refcols + [sample_colname]
    cols_idx = df_input.columns.isin(cols)
    sd_mixin = df_input.loc[central_gene, df_input.columns.isin(cols)].std()
    #print(sd_ref, sd_mixin)

    return(abs(sd_ref-sd_mixin))

if __name__ == "__main__":
	os.chdir('/Users/zagidull/Documents/netzoo/')
	
	# read
	df_ref = pd.read_csv('raw/ref.csv', index_col=0)
	df = pd.read_csv('raw/df.csv', index_col=0)
	ppi = pd.read_csv('raw/ppi_19k.csv')
	
	# process
	# select genes by var, ie filter out with var lower thant quantile percentage
	# quantile = 0.8
	# l_genevarC123 = cleanexp_byquant(transform_df(df.loc[:, df_ref.loc['c1':'c3',:].values.reshape(-1)], method='zeromean'), quantile=quantile)
	# l_genevarS125 = cleanexp_byquant(transform_df(df.loc[:, df_ref.loc[['s1','s2','s5'],:].values.reshape(-1)], method='zeromean'), quantile=quantile)
	# l_genevarS3 = cleanexp_byquant(transform_df(df.loc[:, df_ref.loc[['s3'],:].dropna(axis=1).values.reshape(-1)], method='zeromean'), quantile=quantile)
	# l_genevarS4 = cleanexp_byquant(transform_df(df.loc[:, df_ref.loc[['s4'],:].dropna(axis=1).values.reshape(-1)], method='zeromean'), quantile=quantile)
	# l_genevarS6 = cleanexp_byquant(transform_df(df.loc[:, df_ref.loc[['s6'],:].dropna(axis=1).values.reshape(-1)], method='zeromean'), quantile=quantile)
	# geneset_var80perc = list(set(l_genevarC123 + l_genevarS125 + l_genevarS3 + l_genevarS4 + l_genevarS6))
	# df_ppi_reducedbyvar = reduce_ppi(df_ppi_reduced, geneset_var80perc) 
	ppi = reduce_ppi(ppi, df.index)
	df = df.loc[filter_dfgenes_byppi(ppi, df.index), :]
	holderDict = get_onehop(df.index, ppi)
	
	holder = []
	df_toprocess = transform_df(transform_df(df, method='ln'), method='zeromean')
	d_neighbours = holderDict
	n_jobs = 10
	    
	for i in df_ref.columns:
	    temp = []
	    d_samples = df_ref.loc['s1':'s6',i].dropna(inplace=False).tolist() # get sample ids
	    d_controls = df_ref.loc['c1':'c3',i].tolist() # get control ids
	    df_week = df_toprocess.loc[:, d_controls+d_samples ] # select data for a given week number and sample set
	    
	    for sample in d_samples:
	        
	        params = [ [x, d_neighbours[x], df_week, d_controls ] for x in df_toprocess.index.tolist()]
	        entropy_local = Parallel(n_jobs=n_jobs)(delayed(calculate_entropy_alt)(central_gene,local_network,df_input,refcols) \
	                                           for central_gene,local_network,df_input,refcols in params)
	        
	        params = [ [x,d_neighbours[x],df_week,d_controls,sample] for x in df_toprocess.index.tolist()]
	        entropy_sample = Parallel(n_jobs=n_jobs)(delayed(calculate_entropy_alt)(central_gene,local_network,df_input,refcols,sample_colname) \
	                                            for central_gene,local_network,df_input,refcols,sample_colname in params)
	        ent_local = np.array(entropy_local)
	        ent_local[np.isnan(ent_local)]=0
	    
	        ent_sample = np.array(entropy_sample)
	        ent_sample[np.isnan(ent_sample)]=0
	        
	        params = [ [x,df_week,d_controls,sample] for x in df_toprocess.index.tolist()]
	        sd_differential = Parallel(n_jobs=7)(delayed(calculate_diffsd)(central_gene,df_input,refcols,sample_colname) \
	                                        for central_gene,df_input,refcols,sample_colname in params)
	        sd_diff = np.array(sd_differential)
	    
	        entropy_differential = abs(ent_sample-ent_local)
	        #temp.append(np.sum(entropy_differential)/len(df_toprocess.index))
	        val=entropy_differential*sd_diff
	        temp.append(np.sum(val)/len(df_toprocess.index))
	    print(i, temp)
	    holder.append(temp)	
	
	# combine data
	df0to8 = pd.DataFrame(holder[0:8], index=['wk0','wk1','wk2','wk3','wk4','wk5','wk6','wk7'], columns=['s1','s2','s3','s4','s5','s6']).T
	df8to12= pd.DataFrame(holder[8:12], index=['wk8','wk9','wk10','wk11'], columns=['s1','s2','s3','s5','s6']).T
	df12to17= pd.DataFrame(holder[12:17], index=['wk12','wk13','wk14','wk15','wk16'], columns=['s1','s2','s5','s6']).T	
	df17end= pd.DataFrame(holder[17:], index=['wk17','wk18'], columns=['s1','s2','s5']).T
	df_wk0toend_individual = pd.concat([df0to8, df8to12, df12to17, df17end], axis=1, join="outer") # indi animals
	df_wk0toend_individual.to_csv('saves/wk0toend_ind_zeromeanln_8kppi_gaussMI.csv')
	
	# plot individual curves
	fig, ax = plt.subplots(figsize=(18, 8))
	plt.plot(df_wk0toend_individual.T.fillna(0), label=df_wk0toend_individual.T.columns)	
	ax.set_xticks(range(len(df_wk0toend_individual.T.index)), labels=df_wk0toend_individual.T.index)
	ax.set_ylabel('global differential entropy')
	ax.set_title('global per-sample entropy on tf-ppi (8k pairs) using zeromean-ln rnaseq data and gaussMI. s4 died on wk7, s3 on 11 and s6 on wk16')
	ax.legend()
	plt.savefig('saves/wk0toend_ind_zeromeanln_8kppi_gaussMI.png')


