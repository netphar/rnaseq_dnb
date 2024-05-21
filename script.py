if __name__ == "__main__":
	import os
	import pandas as pd
	import numpy as np
	import typing
	import matplotlib.pyplot as plt
	from scipy.stats import pearsonr, ttest_ind
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
    
def get_neighbours(genename, df_ppi):
    n1 = df_ppi.loc[ df_ppi.InteractorA==genename,'InteractorB'].values
    n2 = df_ppi.loc[ df_ppi.InteractorB==genename,'InteractorA'].values
    n = np.unique(np.concatenate((n1,n2))).tolist()
    return n

def calculate_diffsd(central_gene: str, 
                     df_input: pd.DataFrame, 
                     refcols: list, 
                     sample_colname: str) -> float:
    
    cols_idx = df_input.columns.isin(refcols)
    sd_ref = df_input.loc[central_gene, df_input.columns.isin(refcols)].std()
    
    cols = refcols + [sample_colname]
    cols_idx = df_input.columns.isin(cols)
    sd_mixin = df_input.loc[central_gene, df_input.columns.isin(cols)].std()
    #print(sd_ref, sd_mixin)

    return(abs(sd_ref-sd_mixin))

if __name__ == "__main__":
	os.chdir('/Users/zagidull/Documents/netzoo/')
    	
	# controls
	# Tet-on control mice where BCR-ABL expression was suppressed N=3
	cList1 = ['COHP_44959','COHP_44960','COHP_44961','COHP_44962','COHP_44963','COHP_44964','COHP_44965','COHP_44966','COHP_44967','COHP_44968',
	         'COHP_44969','COHP_44970','COHP_44971','COHP_44972','COHP_44973','COHP_44974','COHP_44975','COHP_44976','COHP_44977']
	cList2 = ['COHP_44978','COHP_44979','COHP_44980','COHP_44981','COHP_44982','COHP_44983','COHP_44984','COHP_44985','COHP_44986','COHP_44987',
	         'COHP_44988','COHP_44989','COHP_44990','COHP_44991','COHP_44992','COHP_44993','COHP_44994','COHP_44995','COHP_44996']
	cList3 = ['COHP_46947','COHP_46948','COHP_46949','COHP_46950','COHP_46951','COHP_46952','COHP_46953','COHP_46954','COHP_46955','COHP_46956',
	         'COHP_46957','COHP_46958','COHP_46959','COHP_46960','COHP_46961','COHP_46962','COHP_46963','COHP_46964','COHP_46965']
	# samples
	# Tet-off CML mice had BCR-ABL expression that induced disease that mimics human chronic phase (CP) CML N=6 
	sList1 = ['COHP_44940','COHP_44941','COHP_44942','COHP_44943','COHP_44944','COHP_44945','COHP_44946','COHP_44947','COHP_44948','COHP_44949','COHP_44950',
	      'COHP_44951','COHP_44952','COHP_44953','COHP_44954','COHP_44955','COHP_44956','COHP_44957','COHP_44958']
	sList2 = ['COHP_44997','COHP_44998','COHP_44999','COHP_45000','COHP_45001','COHP_45002','COHP_45003','COHP_45004','COHP_45005','COHP_45006','COHP_45007',
	      'COHP_45008','COHP_45009','COHP_45010','COHP_45011','COHP_45012','COHP_45013','COHP_45014','COHP_45015']
	sList3 = ['COHP_46927','COHP_46928','COHP_46929','COHP_46930','COHP_46931','COHP_46932','COHP_46933','COHP_46934','COHP_46935','COHP_46936','COHP_46937',
	      'COHP_46938']
	sList4 = ['COHP_46939','COHP_46940','COHP_46941','COHP_46942','COHP_46943','COHP_46944','COHP_46945','COHP_46946']
	sList5 = ['COHP_49205','COHP_49206','COHP_49207','COHP_49208','COHP_49209','COHP_49210','COHP_49211','COHP_49212','COHP_49213','COHP_49214','COHP_49215',
	      'COHP_49216','COHP_49217','COHP_49218','COHP_49219','COHP_49220','COHP_49221','COHP_49222','COHP_49223']
	sList6 = ['COHP_49224','COHP_49225','COHP_49226','COHP_49227','COHP_49228','COHP_49229','COHP_49230','COHP_49231','COHP_49232','COHP_49233','COHP_49234',
	      'COHP_49235','COHP_49236','COHP_49237','COHP_49238','COHP_49239','COHP_49240']

	# reference for controls
	df_c = pd.DataFrame([cList1, cList2, cList3])
	df_c.columns = [f'wk{x}' for x in range(len(cList1))]
	df_c.index = ['c1','c2','c3']
	# reference for samples
	df_s = pd.DataFrame([sList1, sList2, sList3, sList4, sList5, sList6])
	df_s.columns = [f'wk{x}' for x in range(len(sList1))]
	df_s.index = [f's{x}' for x in range(1,7,1)]
	df_ref = pd.concat([df_c, df_s], axis=0, join='outer').replace({None: np.nan}) # samples as rows, timepoints as columns

	# read in rnaseq data, it is qnormed i believe
	df = pd.read_csv('GSE244990_cml_mrna_processed_1tpm_in_5_samples.tsv') # https://www.ncbi.xyz/geo/query/acc.cgi?acc=GSE244990
	df.iloc[:,0] = df.iloc[:,0].str.split('.',expand=True).iloc[:,0].values
	df.rename(columns={'Unnamed: 0':'GeneName'}, inplace=True)
	df.set_index('GeneName', inplace=True)
	
	# leave only test and control samples. they have two more sets where bcr-abl was turned off halfway through and where stimulated treatment, so ignore those
	df = df.loc[:, df_ref.stack().values]
	
	# read ppi. it is a normalized at gene level list from pickle3
	ppi = pd.read_csv('ppi.csv')
	ppi_reduced = ppi[ppi.InteractorA.isin(df.index)].reset_index(drop=True)
	ppi_reduced = ppi_reduced[ppi_reduced.InteractorB.isin(df.index)].reset_index(drop=True)
	unique_genes = np.unique(ppi_reduced.values.reshape(-1))
	df = df.loc[df.index.isin(unique_genes),:]
	
	# get one-hop neighbours in ppi-tf network
	holderDict = {}
	for idx in df.index:
		holderDict[idx]=get_neighbours(idx, ppi_reduced)
	
	# calculate entropy
	holder = []
	df_log = np.log2(df+0.000000001) # doing log-transform seems to stabilizes entropy calculation, vs using rnaseq values as-is. it also makes the differential entropy scores much smaller 
	df_toprocess = df_log
	n_jobs = 10
	    
	for i in df_ref.columns:
	    temp = []
	    d_samples = df_ref.loc['s1':'s6',i].dropna(inplace=False).tolist() # get sample ids
	    d_controls = df_ref.loc['c1':'c3',i].tolist() # get control ids
	    df_week = df_toprocess.loc[:, d_controls+d_samples ] # select data for a given week number and sample set
	    
	    for sample in d_samples:
	        
	        params = [ [x, holderDict[x], df_week, d_controls ] for x in df_toprocess.index.tolist()]
	        entropy_local = Parallel(n_jobs=n_jobs)(delayed(calculate_entropy_faster)(central_gene,local_network,df_input,refcols) \
	                                           for central_gene,local_network,df_input,refcols in params)
	        
	        params = [ [x,holderDict[x],df_week,d_controls,sample] for x in df_toprocess.index.tolist()]
	        entropy_sample = Parallel(n_jobs=n_jobs)(delayed(calculate_entropy_faster)(central_gene,local_network,df_input,refcols,sample_colname) \
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
	        val=entropy_differential*sd_diff
	        temp.append(np.sum(val)/len(df_toprocess.index))
	    holder.append(temp)
	
	# combine data
	df0to8 = pd.DataFrame(holder[0:8], index=['wk0','wk1','wk2','wk3','wk4','wk5','wk6','wk7'], columns=['s1','s2','s3','s4','s5','s6']).T
	df8to12= pd.DataFrame(holder[8:12], index=['wk8','wk9','wk10','wk11'], columns=['s1','s2','s3','s5','s6']).T
	df12to17= pd.DataFrame(holder[12:17], index=['wk12','wk13','wk14','wk15','wk16'], columns=['s1','s2','s5','s6']).T	
	df17end= pd.DataFrame(holder[17:], index=['wk17','wk18'], columns=['s1','s2','s5']).T
	df_wk0toend_individual = pd.concat([df0to8, df8to12, df12to17, df17end], axis=1, join="outer") # indi animals
	df_wk0toend_individual.to_csv('wk0toend_individualsamples.csv')
	
	# plot individual curves
	fig, ax = plt.subplots(figsize=(18, 8))
	plt.plot(df_wk0toend_individual.T.fillna(0), label=df_wk0toend_individual.T.columns)	
	ax.set_xticks(range(len(df_wk0toend_individual.T.index)), labels=df_wk0toend_individual.T.index)
	ax.set_ylabel('global differential entropy')
	ax.set_title('global per-sample entropy on ppi-tf using log2 rnaseq data. s3 died on wk7, s4 on 11 and s6 on wk16')
	ax.legend()
	plt.savefig('wk0toend_ind_log2rnaseq.png')
	
	# plot averaged curve
	df_wk0toend_averaged = df_wk0toend_individual.mean() # averaged for all ignoring diseased animals
	fig, ax = plt.subplots(figsize=(18, 8))
	plt.plot(df_wk0toend_averaged, label='averaged across samples')
	ax.set_xticks(range(len(df_wk0toend_averaged.index)), labels=df_wk0toend_averaged.index)
	ax.set_ylabel('averaged global differential entropy')
	ax.set_title('avg')
	ax.legend()
	plt.savefig('wk0toend_avg_log2rnaseq.png')


