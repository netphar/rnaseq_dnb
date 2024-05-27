if __name__ == "__main__":
	import pandas as pd
	import matplotlib.pyplot as plt
	
	df=pd.read_csv('saves/wk0toend_ind_0u1var_ln_5kPPI_40percCV_gaussMI.csv', index_col=0)
	fig, ax = plt.subplots(nrows=2, ncols=1,figsize=(18, 10), sharex=True)
	#fig.subplots_adjust(hspace=0)
	
	ymax = 0.015
	
	ax[1].set_ylim([0, ymax])
	ax[1].set_ylabel('global differential entropy')
	ax[1].set_title(f'averaged gaussMI entropy for lived (s1,s2,s5) and died (s3,s4,s6)')
	ax[1].plot(df.loc[:,['s1','s2','s5']].mean(axis=1),'--', label='lived')
	ax[1].plot(df.loc[:,['s3','s4','s6']].mean(axis=1),':', label='died')
	ax[1].axvline(x=7, color='crimson', linestyle='-')
	ax[1].axvline(x=11, color='g', linestyle='-')
	ax[1].axvline(x=16, color='brown', linestyle='-')
	ax[1].legend()
	
	ax[0].set_ylim([0, ymax])
	ax[0].set_ylabel('global differential entropy')
	ax[0].set_title(f'per-sample gaussMI entropy on tf-ppi=5k, rnaseq is Ln-transformed, normalized to $\mu$=0 var=1 and filtered at 40% of CoefVar')
	ax[0].plot(df, label=df.columns)
	ax[0].axvline(x=7, ymin=0, ymax=df.loc['wk7', 's4']/ymax, color='crimson', linestyle='-')
	ax[0].text(7, 0.0005, 's4 died', bbox={'facecolor':'white', 'edgecolor':'crimson'})
	ax[0].axvline(x=11, ymin=0, ymax=df.loc['wk11','s3']/ymax, color='green', linestyle='-')
	ax[0].text(11, 0.0005, 's3 died', bbox={'facecolor':'white', 'edgecolor':'green'})
	ax[0].axvline(x=16, ymin=0, ymax=df.loc['wk16','s6']/ymax, color='brown', linestyle='-')
	ax[0].text(16, 0.0005, 's6 died', bbox={'facecolor':'white', 'edgecolor':'brown'})
	ax[0].legend()
	
	plt.savefig('saves/wk0toend_0u1var_ln_5kPPI_40percCV_gaussMI.png')