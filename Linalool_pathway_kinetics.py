""" Authors:    Hamid Doosthosseini < hdoosth@mit.edu > , Voigt Lab , MIT
                Jong Hyun Park <  > , Voigt Lab , MIT """
""" Last updated: 10/17/2022"""
# ---------------------------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------------------------------- #

import numpy as np
import pandas as pd
import warnings
from datetime import datetime
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

gamma = 0.00577                                 # [min^-1] measured dilution rate
rpu_to_numpercell = 169000/13.66                # [RPU^-1] pTDH3 expression count (Global analysis of protein expression in yeast. Nature 2003, 425 (6959), 737-741) / pTDH3 gene RPU (Genetic circuit design automation for yeast. Nat. Microbiol. 2020, 5 (11), 1349-1360.)
numpercell_to_uM = 3.93e-5                      # 1/(6.022e23*42e-15*1e6) [uM] (BioNumbers, Nucleic Acids Res. 2010, 38 (suppl_1), D750-D753)
OD1cellvol_per_brothvol = 42*1.5e7*1e-12        # [um^3/cell]*[OD1 cell/ml * ml/um^3] (BioNumbers, Nucleic Acids Res. 2010, 38 (suppl_1), D750-D753)
linalool_mw = 154.25                            # [g/mol]
rpu_to_proteinpermin = rpu_to_numpercell*gamma  # [min^-1] expression rate
max_OD = 7.0

def format_exp(n,base=10): 
    superscript_dict={'-':'\u207B','0':'\u2070','1':'\u00B9','2':'\u00B2','3':'\u00B3','4':'\u2074','5':'\u2075','6':'\u2076','7':'\u2077','8':'\u2078','9':'\u2079'}
    return str(base)+"".join([superscript_dict[i] for i in str(n)])

plt.rcParams["font.family"] = 'Arial'
plt.rcParams["font.size"] = 8
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
inch_plot_width = 2.0

folder_name = '2022_10_17_noglucose_v3_highq/'

class chemical:
    def __init__(self, name, value=0.0, unit='M'):
        self.name = name
        self.unit = unit
        self.reset_value = value
        self.conc=value

    def set_value(self, value, unit):
        self.conc = value
        self.unit = unit

    def reset(self):
        self.conc = self.reset_value

    def make_current_reset(self):
        self.reset_value = self.conc

class enzyme:
    def __init__(self, name='', start=0.0, y0=0.0, y=0.0, k={}):
        self.type = 'enzyme'
        self.starttime = start
        self.name = name
        self.y0 = y0*rpu_to_numpercell*numpercell_to_uM
        self.y = y*rpu_to_numpercell*numpercell_to_uM
        self.k = k
        self.reset_value = self.y0
        self.conc = self.y0

    def update(self,t):
        if t > self.starttime:
            self.conc = self.y0 + ((self.y-self.y0))*(1-np.exp(-gamma*(t-self.starttime)))
        else:
            self.conc=self.reset_value

    def reset(self):
        self.conc = self.reset_value

    def make_current_reset(self):
        self.reset_value = self.conc

THMGR = enzyme('THMGR',0,11.57e-3,11.62,{'hMG_CoA':[0.0035*53.01,50]})          # [min^-1,uM]
IDI = enzyme('IDI',0,148.57e-3,5.92,{'iPP':[20*33.352,43],'dMAPP':[456.75,43]}) #[min^-1,uM]
ERG20 = enzyme('ERG20',0,62.85e-3,14.78,{'gPP':[0.021*60/10,27.56],'dMAPP':[0.012*60/10,0.49]})     #[min^-1, uM]
LIS = enzyme('LIS',0,4.28e-3,21.28,{'gPP':[14.4,25]})                           #[min^-1,uM]
enzymes = [THMGR,IDI,ERG20,LIS]

hMG_CoA = chemical('hMG_CoA',0.111/200*1e6,'uM')            # assumed excess, doi:10.1016/j.ymben.2011.07.001, and Jour. Biol. ChemistryVol. 236, No. 9, September 1960
iPP = chemical('iPP',0.001,'uM')
dMAPP = chemical('dMAPP',0.001,'uM')
gPP = chemical('gPP',0.001,'uM')
fPP = chemical('fPP',0.001,'uM')
linalool = chemical('linalool',1e-9,'uM')
metabolites = [hMG_CoA, iPP, dMAPP, gPP, fPP, linalool]

def run_sim(induction_state={},publish=False,dt=0.001):

    for key,value in induction_state.items():
        count_inducers = 0
        for enzyme in enzymes:
            if enzyme.name == key:
                enzyme.starttime = value
                count_inducers += 1
        if count_inducers < 1: 
            warnings.warn("No " + key + " found in list of enzymes", UserWarning, stacklevel=2)
        if count_inducers > 1: 
            warnings.warn("Multiple entries for " + key + " found in list of enzymes", UserWarning, stacklevel=2)

    od = 0.005
    t = 0.0
    data = {}
    for enzyme in enzymes:
        enzyme.reset()
        data[enzyme.name+'_conc'] = [enzyme.reset_value,enzyme.reset_value]
    for metabolite in metabolites:
        metabolite.reset()
        data[metabolite.name+'_conc'] = [metabolite.reset_value,metabolite.reset_value]
    data['time'] = [-60,-1e-5]
    data['OD'] = [od,od]

    while od<max_OD:
        v_mevalonate = THMGR.k['hMG_CoA'][0]*THMGR.conc*hMG_CoA.conc/(THMGR.k['hMG_CoA'][1]+hMG_CoA.conc)
        v_IDI_IPP = IDI.k['iPP'][0]*IDI.conc*iPP.conc/(IDI.k['iPP'][1]+iPP.conc)
        v_IDI_DMAPP = IDI.k['dMAPP'][0]*IDI.conc*dMAPP.conc/(IDI.k['dMAPP'][1]+dMAPP.conc)
        v_ERG_DMAPP = ERG20.k['dMAPP'][0]*ERG20.conc*dMAPP.conc*iPP.conc/((ERG20.k['dMAPP'][1]+dMAPP.conc)*(4.7+iPP.conc))
        v_ERG_GPP = ERG20.k['gPP'][0]*ERG20.conc*gPP.conc*iPP.conc/((ERG20.k['gPP'][1]+gPP.conc)*(4.7+iPP.conc))
        v_LIS_GPP = LIS.k['gPP'][0]*LIS.conc*gPP.conc/(LIS.k['gPP'][1]+gPP.conc)
        d = {}
        d['hMG_CoA'] = 0
        d['iPP'] = v_mevalonate + v_IDI_DMAPP - v_IDI_IPP - v_ERG_DMAPP - v_ERG_GPP - gamma*iPP.conc
        d['dMAPP'] = v_IDI_IPP - v_IDI_DMAPP - v_ERG_DMAPP - gamma*dMAPP.conc
        d['gPP'] = v_ERG_DMAPP - v_ERG_GPP - v_LIS_GPP - gamma*gPP.conc
        d['fPP'] = v_ERG_GPP - gamma*fPP.conc
        d['linalool'] = v_LIS_GPP - gamma*linalool.conc

        # for metabolite in metabolites:
        #     if metabolite.conc < -2*d[metabolite.name]*dt:
        #         print("WARNING: rate v. timestep error")
        #         print(metabolite.name,metabolite.conc,d[metabolite.name]*dt,t)

        t += dt
        od += od*gamma*dt
        for enzyme in enzymes:
            enzyme.update(t)
        for metabolite in metabolites:
            metabolite.conc += d[metabolite.name]*dt
        data['time'].append(t)
        data['OD'].append(od)
        for enzyme in enzymes:
            data[enzyme.name+'_conc'].append(enzyme.conc)
        for metabolite in metabolites:
            data[metabolite.name+'_conc'].append(metabolite.conc)
    s = '_'
    for x in induction_state.values():
        s += str(x/60).split('.')[0]+'_'    
    output_file = 'run_'+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+s+'.csv'
    if publish:
        pd.DataFrame.from_dict(data).to_csv(folder_name+output_file)
        time.sleep(0.5)
    data_ss = {}
    for key in data:
        data_ss[key] = [data[key][-1]]
    return(data_ss,output_file)

def plot_sim():
    data_end = pd.read_csv(folder_name+'experiments_simulated.csv')

    gs = gridspec.GridSpec(1,1)
    fig1 = plt.figure(figsize=(inch_plot_width,inch_plot_width))
    ax1 = plt.subplot(gs[0,0])
    ax1.scatter(data_end['linalool_conc']*OD1cellvol_per_brothvol*linalool_mw,data_end['LIN_per_OD'],s=16,color='black',marker='.',alpha=0.8,linewidth=0,zorder=10)
    correlation_matrix = np.corrcoef(data_end['linalool_conc']*OD1cellvol_per_brothvol*linalool_mw,data_end['LIN_per_OD'])
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    print(r_squared)
    ax1.set_ylim([0,7.0])
    ax1.set_yticks([0,1,2,3,4,5,6])
    ax1.set_xlim([0,7.0])
    ax1.set_xticks([0,1,2,3,4,5,6])
    for axis in ['bottom', 'left','top','right']:
        ax1.spines[axis].set_linewidth(0.5)
        ax1.spines[axis].set_color('black')
    ax1.tick_params(axis='both', which='major', labelsize=8, length=3, width=0.5, direction='out', pad=4)
    ax1.tick_params(axis='both', which='minor', labelsize=8, length=2, width=0.5, direction='out')
    ax1.set_ylabel('Measured titer (ug/L-OD600)')
    ax1.set_xlabel('Simulated titer (ug/L-OD600)')
    plt.savefig(folder_name+'scatter_lin2.pdf',dpi=300)
    plt.savefig(folder_name+'scatter_lin2.png',dpi=300)
    plt.clf()
    plt.close()

induction_state0={'THMGR':48*60,'IDI':48*60,'ERG20':48*60,'LIS':48*60}      # initial induction state
data0, run_file = run_sim(induction_state0,publish=True,dt=0.001)           # simulates state with induction_state0

for enzyme in enzymes:                                                      # set all parts and inductions to current steady state
    enzyme.make_current_reset()
    enzyme.reset()

exps = pd.read_csv(folder_name+'experiments.csv')
for exp in exps.iterrows():
    induction_state={'THMGR':exp[1]['THMGR']*60,'IDI':exp[1]['IDI']*60,'ERG20':exp[1]['ERG20']*60,'LIS':exp[1]['LIS']*60}
    data, run_file = run_sim(induction_state,publish=True,runtime=24*60,dt=0.002)
    for key, dat in data.items():
        if key not in exps.columns:
            exps[key]=list(exps['sample'])
        exps.loc[exps['sample']==exp[1]['sample'],key] = dat
exps.to_csv('experiments_simulated.csv')

plot_sim()