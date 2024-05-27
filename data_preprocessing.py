if __name__ == "__main__":
	import requests
	import pandas as pd
	import numpy as np
	
	fname = '/Users/zagidull/Documents/netzoo/rnaseq_dnb/raw/GSE244990_cml_mrna_processed_1tpm_in_5_samples.tsv.gz'
	url = 'https://www.ncbi.xyz/geo/download/?acc=GSE244990&format=file&file=GSE244990%5Fcml%5Fmrna%5Fprocessed%5F1tpm%5Fin%5F5%5Fsamples%2Etsv%2Egz'
	r = requests.get(url)
	with open(fname , 'wb') as f:
		f.write(r.content)

	df = pd.read_csv(fname, compression='gzip') # https://www.ncbi.xyz/geo/query/acc.cgi?acc=GSE244990
	df.iloc[:,0] = df.iloc[:,0].str.split('.',expand=True).iloc[:,0].values
	df.rename(columns={'Unnamed: 0':'GeneName'}, inplace=True)
	df.set_index('GeneName', inplace=True)
	df.to_csv('/Users/zagidull/Documents/netzoo/rnaseq_dnb/raw/df.csv', index=True)
	
	# controls
	# Tet-on control mice where BCR-ABL expression was suppressed N=3
	
	cList1 = ['COHP_44959','COHP_44960','COHP_44961','COHP_44962','COHP_44963','COHP_44964','COHP_44965','COHP_44966','COHP_44967','COHP_44968','COHP_44969','COHP_44970','COHP_44971','COHP_44972','COHP_44973','COHP_44974','COHP_44975','COHP_44976','COHP_44977']
	
	cList2 = ['COHP_44978','COHP_44979','COHP_44980','COHP_44981','COHP_44982','COHP_44983','COHP_44984','COHP_44985','COHP_44986','COHP_44987','COHP_44988','COHP_44989','COHP_44990','COHP_44991','COHP_44992','COHP_44993','COHP_44994','COHP_44995','COHP_44996']
	
	cList3 = ['COHP_46947','COHP_46948','COHP_46949','COHP_46950','COHP_46951','COHP_46952','COHP_46953','COHP_46954','COHP_46955','COHP_46956','COHP_46957','COHP_46958','COHP_46959','COHP_46960','COHP_46961','COHP_46962','COHP_46963','COHP_46964','COHP_46965']
	
	# samples
	# Tet-off CML mice had BCR-ABL expression that induced disease that mimics human chronic phase (CP) CML N=6 
	
	sList1 = ['COHP_44940','COHP_44941','COHP_44942','COHP_44943','COHP_44944','COHP_44945','COHP_44946','COHP_44947','COHP_44948','COHP_44949','COHP_44950','COHP_44951','COHP_44952','COHP_44953','COHP_44954','COHP_44955','COHP_44956','COHP_44957','COHP_44958']
	
	sList2 = ['COHP_44997','COHP_44998','COHP_44999','COHP_45000','COHP_45001','COHP_45002','COHP_45003','COHP_45004','COHP_45005','COHP_45006','COHP_45007','COHP_45008','COHP_45009','COHP_45010','COHP_45011','COHP_45012','COHP_45013','COHP_45014','COHP_45015']
	
	sList3 = ['COHP_46927','COHP_46928','COHP_46929','COHP_46930','COHP_46931','COHP_46932','COHP_46933','COHP_46934','COHP_46935','COHP_46936','COHP_46937','COHP_46938']
	sList4 = ['COHP_46939','COHP_46940','COHP_46941','COHP_46942','COHP_46943','COHP_46944','COHP_46945','COHP_46946']
	
	sList5 = ['COHP_49205','COHP_49206','COHP_49207','COHP_49208','COHP_49209','COHP_49210','COHP_49211','COHP_49212','COHP_49213','COHP_49214','COHP_49215','COHP_49216','COHP_49217','COHP_49218','COHP_49219','COHP_49220','COHP_49221','COHP_49222','COHP_49223']
	
	sList6 = ['COHP_49224','COHP_49225','COHP_49226','COHP_49227','COHP_49228','COHP_49229','COHP_49230','COHP_49231','COHP_49232','COHP_49233','COHP_49234','COHP_49235','COHP_49236','COHP_49237','COHP_49238','COHP_49239','COHP_49240']

	# reference for controls
	df_c = pd.DataFrame([cList1, cList2, cList3])
	df_c.columns = [f'wk{x}' for x in range(len(cList1))]
	df_c.index = ['c1','c2','c3']
	
	# reference for samples
	df_s = pd.DataFrame([sList1, sList2, sList3, sList4, sList5, sList6])
	df_s.columns = [f'wk{x}' for x in range(len(sList1))]
	df_s.index = [f's{x}' for x in range(1,7,1)]
	
	df_ref = pd.concat([df_c, df_s], axis=0, join='outer').replace({None: np.nan}) # samples as rows, timepoints as columns
	df_ref.to_csv('/Users/zagidull/Documents/netzoo/rnaseq_dnb/raw/ref.csv', index=True)