if __name__ == "__main__":
	import os
	import pandas as pd
	import numpy as np
	import typing
	from scipy.stats import pearsonr, ttest_ind
	from joblib import Parallel, delayed

def calculate_entropy(central_gene: str, 
                            local_network: list, 
                            df_input: pd.DataFrame, 
                            refcols: list, 
                            sample_colname: str='') -> float:
    if not sample_colname:
        cols_idx = df_input.columns.isin(refcols)
    else:
        refcols.append(sample_colname)
        cols_idx = df_input.columns.isin(refcols)
    
    df_central = df_input.loc[central_gene,cols_idx].values

    local_entropy = 0
    for gene in local_network:
        upper = abs(pearsonr( df_input.loc[gene,cols_idx].values, df_central)[0])
        lower = 0
        for gene1 in local_network:
            lower += abs(pearsonr( df_input.loc[gene1,cols_idx].values, df_central)[0])
        p_i = upper/lower
        e_i = p_i*np.log2(p_i)
        local_entropy +=e_i

    return(-1*local_entropy/len(local_network))
    
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
    
    refcols.append(sample_colname)
    cols_idx = df_input.columns.isin(refcols)
    sd_mixin = df_input.loc[central_gene, df_input.columns.isin(refcols)].std()
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
	
	df = pd.read_csv('GSE244990_cml_mrna_processed_1tpm_in_5_samples.tsv') # https://www.ncbi.xyz/geo/query/acc.cgi?acc=GSE244990
	df.iloc[:,0] = df.iloc[:,0].str.split('.',expand=True).iloc[:,0].values
	df.rename(columns={'Unnamed: 0':'GeneName'}, inplace=True)
	df.set_index('GeneName', inplace=True)
	
	# leave only test and control
	df = df.loc[:, cList1+cList2+cList3+sList1+sList2+sList3+sList4+sList5+sList6]
	# read ppi
	ppi = pd.read_csv('ppi.csv')
	ppi_reduced = ppi[ppi.InteractorA.isin(df.index)].reset_index(drop=True)
	ppi_reduced = ppi_reduced[ppi_reduced.InteractorB.isin(df.index)].reset_index(drop=True)
	unique_genes = np.unique(ppi_reduced.values.reshape(-1))
	df = df.loc[df.index.isin(unique_genes),:]
	
	holderDict = {}
	for idx in df.index:
		holderDict[idx]=get_neighbours(idx, ppi_reduced)
	
	# weeks 8 to 12 for all but sample4
	holder = []
	for i in list(range(8,12)):
	    temp = []
	    df_week = df.loc[:, [cList1[i],cList2[i],cList3[i],sList1[i],sList2[i],sList3[i],sList5[i],sList6[i] ]] 
	    df_week.columns = ['c1', 'c2', 'c3', 's1', 's2', 's3', 's5', 's6']
	    for sample in ['s1', 's2', 's3', 's5', 's6']:
	        
	        params = [ [x,holderDict[x],df_week,['c1', 'c2', 'c3']] for x in df.index.tolist()]
	        entropy_local = Parallel(n_jobs=9)(delayed(calculate_entropy)(central_gene,local_network,df_input,refcols) \
	                                           for central_gene,local_network,df_input,refcols in params)
	        
	        params = [ [x,holderDict[x],df_week,['c1', 'c2', 'c3'],sample] for x in df.index.tolist()]
	        entropy_sample = Parallel(n_jobs=9)(delayed(calculate_entropy)(central_gene,local_network,df_input,refcols,sample_colname) \
	                                            for central_gene,local_network,df_input,refcols,sample_colname in params)
	        ent_local = np.array(entropy_local)
	        ent_local[np.isnan(ent_local)]=0
	    
	        ent_sample = np.array(entropy_sample)
	        ent_sample[np.isnan(ent_sample)]=0
	        
	        params = [ [x,df_week,['c1', 'c2', 'c3'],sample] for x in df.index.tolist()]
	        sd_differential = Parallel(n_jobs=9)(delayed(calculate_diffsd)(central_gene,df_input,refcols,sample_colname) \
	                                        for central_gene,df_input,refcols,sample_colname in params)
	        sd_diff = np.array(sd_differential)
	    
	        entropy_differential = abs(ent_sample-ent_local)
	        val=entropy_differential*sd_diff
	        temp.append(np.sum(val)/len(df.index))
	    print(i,temp)
	    holder.append(temp)
	   
	df_week8to12 = pd.DataFrame(holder).T
	df_week8to12.columns = ['wk8','wk9','wk10','wk11']
	df_week8to12.index = ['s1','s2','s3','s5','s6']
	df_week8to12.to_csv('week8to12.csv')
	
	# weeks 12 to 17 for all but sample4,sample3
	holder = []
	for i in list(range(12,17)):
	    temp = []
	    df_week = df.loc[:, [cList1[i],cList2[i],cList3[i],sList1[i],sList2[i],sList5[i],sList6[i] ]] 
	    df_week.columns = ['c1', 'c2', 'c3', 's1', 's2', 's5', 's6']
	    for sample in ['s1', 's2', 's5', 's6']:
	        
	        params = [ [x,holderDict[x],df_week,['c1', 'c2', 'c3']] for x in df.index.tolist()]
	        entropy_local = Parallel(n_jobs=9)(delayed(calculate_entropy)(central_gene,local_network,df_input,refcols) \
	                                           for central_gene,local_network,df_input,refcols in params)
	        
	        params = [ [x,holderDict[x],df_week,['c1', 'c2', 'c3'],sample] for x in df.index.tolist()]
	        entropy_sample = Parallel(n_jobs=9)(delayed(calculate_entropy)(central_gene,local_network,df_input,refcols,sample_colname) \
	                                            for central_gene,local_network,df_input,refcols,sample_colname in params)
	        ent_local = np.array(entropy_local)
	        ent_local[np.isnan(ent_local)]=0
	    
	        ent_sample = np.array(entropy_sample)
	        ent_sample[np.isnan(ent_sample)]=0
	        
	        params = [ [x,df_week,['c1', 'c2', 'c3'],sample] for x in df.index.tolist()]
	        sd_differential = Parallel(n_jobs=9)(delayed(calculate_diffsd)(central_gene,df_input,refcols,sample_colname) \
	                                        for central_gene,df_input,refcols,sample_colname in params)
	        sd_diff = np.array(sd_differential)
	    
	        entropy_differential = abs(ent_sample-ent_local)
	        val=entropy_differential*sd_diff
	        temp.append(np.sum(val)/len(df.index))
	    print(i,temp)
	    holder.append(temp)
	   
	df_week12to17 = pd.DataFrame(holder).T
	df_week12to17.columns = ['wk12','wk13','wk14','wk15','wk16']
	df_week12to17.index = ['s1','s2','s5','s6']
	df_week12to17.to_csv('week12to17.csv')
	
	# weeks 17 to 19 for all but sample4,sample3
	# farts here. list out of range
	# Traceback (most recent call last):
	# File "/Users/zagidull/Documents/netzoo/script.py", line 183, in <module>
	# df_week = df.loc[:, [cList1[i],cList2[i],cList3[i],sList1[i],sList2[i],sList5[i],sList6[i] ]]
	# IndexError: list index out of range
	holder = []
	for i in [17,18]:
	    temp = []
	    df_week = df.loc[:, [cList1[i],cList2[i],cList3[i],sList1[i],sList2[i],sList5[i],sList6[i] ]] 
	    df_week.columns = ['c1', 'c2', 'c3', 's1', 's2', 's5']
	    for sample in ['s1', 's2', 's5']:
	        
	        params = [ [x,holderDict[x],df_week,['c1', 'c2', 'c3']] for x in df.index.tolist()]
	        entropy_local = Parallel(n_jobs=9)(delayed(calculate_entropy)(central_gene,local_network,df_input,refcols) \
	                                           for central_gene,local_network,df_input,refcols in params)
	        
	        params = [ [x,holderDict[x],df_week,['c1', 'c2', 'c3'],sample] for x in df.index.tolist()]
	        entropy_sample = Parallel(n_jobs=9)(delayed(calculate_entropy)(central_gene,local_network,df_input,refcols,sample_colname) \
	                                            for central_gene,local_network,df_input,refcols,sample_colname in params)
	        ent_local = np.array(entropy_local)
	        ent_local[np.isnan(ent_local)]=0
	    
	        ent_sample = np.array(entropy_sample)
	        ent_sample[np.isnan(ent_sample)]=0
	        
	        params = [ [x,df_week,['c1', 'c2', 'c3'],sample] for x in df.index.tolist()]
	        sd_differential = Parallel(n_jobs=9)(delayed(calculate_diffsd)(central_gene,df_input,refcols,sample_colname) \
	                                        for central_gene,df_input,refcols,sample_colname in params)
	        sd_diff = np.array(sd_differential)
	    
	        entropy_differential = abs(ent_sample-ent_local)
	        val=entropy_differential*sd_diff
	        temp.append(np.sum(val)/len(df.index))
	    print(i,temp)
	    holder.append(temp)
	   
	df_week12to17 = pd.DataFrame(holder).T
	df_week12to17.columns = ['wk17','wk18']
	df_week12to17.index = ['s1','s2','s5']
	df_week12to17.to_csv('week17to19.csv')



