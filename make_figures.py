#!/usr/bin/env python3
"""Export evidence figures declared in the model contract."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent; RESULTS=ROOT/'results'; FIG=ROOT/'figures'
SKILL='/Users/tarangjammalamadaka/.codex/skills/math-modeling'
sys.path.insert(0, str(ROOT)); sys.path.insert(0, f'{SKILL}/tools/figure/scripts')
from utils.plot_style import apply_publication_style, PALETTE
from export_figure import export_figure

def save(fig, name):
    export_figure(fig, str(FIG/name), formats=['svg','png'], size_inches=(6.3,3.5), dpi=300, grayscale_preview=False, tight=False)
    plt.close(fig)

def main():
    FIG.mkdir(exist_ok=True); apply_publication_style(language='en', width='report')
    market=pd.read_csv(ROOT/'data/synthetic_market.csv'); diag=pd.read_csv(RESULTS/'filter_diagnostics.csv')
    costs=pd.read_csv(RESULTS/'path_costs.csv'); trades=pd.read_csv(RESULTS/'trade_schedules.csv'); summary=pd.read_csv(RESULTS/'summary_metrics.csv'); paired=pd.read_csv(RESULTS/'paired_comparisons.csv'); stress=pd.read_csv(RESULTS/'stress_test.csv')
    online=pd.read_csv(RESULTS/'online_learning_metrics.csv'); online_diag=pd.read_csv(RESULTS/'online_holdout_diagnostics.csv')
    # Raw q1: distributions; raw q1: observed spread-depth relationship.
    fig,axs=plt.subplots(1,3,figsize=(6.3,2.2));
    for ax,col,label in zip(axs,['log_trailing_vol','log_spread','log_depth'],['log trailing volatility','log spread','log displayed depth']):
        for z,c,l in [(0,PALETTE['primary'],'calm'),(1,PALETTE['contrast'],'stressed')]: ax.hist(market.loc[market.regime_true==z,col],bins=35,density=True,histtype='step',lw=1.2,color=c,label=l)
        ax.set_xlabel(label); ax.set_ylabel('density')
    axs[0].legend(); fig.suptitle('Raw q1 — overlapping liquidity-state emissions'); save(fig,'raw_q1_observation_distributions')
    fig,ax=plt.subplots(figsize=(6.3,3.5));
    for z,c,l in [(0,PALETTE['primary'],'calm'),(1,PALETTE['contrast'],'stressed')]:
        d=market[market.regime_true==z].sample(2000,random_state=7); ax.scatter(d.log_depth,d.log_spread,s=4,alpha=.16,color=c,label=l)
    ax.set(xlabel='log displayed depth',ylabel='log spread (bps)',title='Raw q1 — depth/spread relation'); ax.legend(); save(fig,'raw_q1_depth_spread')
    # Raw q2: outcome distribution
    fig,ax=plt.subplots(figsize=(6.3,3.5)); order=['twap','static_ac','regime_aware_mpc']; data=[costs[costs.policy==p].implementation_shortfall_bps for p in order]
    ax.boxplot(data,tick_labels=['TWAP','Static AC','Regime-aware'],showfliers=False); ax.axhline(0,color='black',lw=.7); ax.set(ylabel='implementation shortfall (bps)',title='Raw q2 — distribution of path costs (n=1,200 each)'); save(fig,'raw_q2_shortfall_distribution')
    # Process q1: posterior timeline; process q1: confusion matrix
    fig,ax=plt.subplots(figsize=(6.3,3.5));
    for pid in [1,8,15]:
        d=diag[diag.path_id==pid]; ax.plot(d.t,d.posterior_stress,lw=1,label=f'path {pid}')
    ax.set(xlabel='interval',ylabel='filtered stress probability',ylim=(-.03,1.03),title='Process q1 — causal posterior updates'); ax.legend(ncol=3); save(fig,'process_q1_posterior_timeline')
    fig,ax=plt.subplots(figsize=(6.3,3.5)); cm=pd.crosstab(diag.regime_true,diag.predicted_stress).reindex(index=[0,1],columns=[0,1],fill_value=0).to_numpy(); ax.pcolormesh(np.arange(3),np.arange(3),cm,cmap='Blues',shading='flat');
    for i in range(2):
        for j in range(2): ax.text(j+.5,i+.5,str(cm[i,j]),ha='center',va='center')
    ax.set(xlim=(0,2),ylim=(2,0),xticks=[.5,1.5],xticklabels=['calm','stress'],yticks=[.5,1.5],yticklabels=['calm','stress'],xlabel='predicted state',ylabel='true state',title='Process q1 — filter confusion matrix'); save(fig,'process_q1_confusion_matrix')
    # Process q2: execution trajectory by state inference
    fig,ax=plt.subplots(figsize=(6.3,3.5));
    for p,c,l in [('twap',PALETTE['neutral'],'TWAP'),('static_ac',PALETTE['secondary'],'Static AC'),('regime_aware_mpc',PALETTE['positive'],'Regime-aware')]:
        d=trades[trades.policy==p].groupby('t').trade_fraction.mean(); ax.plot(d.index,d.values,marker='o',ms=2.5,color=c,label=l)
    ax.set(xlabel='interval',ylabel='mean child-order fraction',title='Process q2 — average execution schedules'); ax.legend(); save(fig,'process_q2_execution_schedules')
    # Result q1: reliability; result q2: tail cost; result q2: feasibility.
    cal=pd.read_csv(RESULTS/'filter_calibration.csv').dropna()
    fig,ax=plt.subplots(figsize=(6.3,3.5)); ax.scatter(cal.mean_posterior,cal.observed_stress_rate,s=30,color=PALETTE['primary']); ax.plot([0,1],[0,1],'--',color='black',lw=.8); ax.set(xlim=(0,1),ylim=(0,1),xlabel='mean posterior',ylabel='observed stress frequency',title='Result q1 — posterior calibration'); save(fig,'result_q1_calibration')
    fig,ax=plt.subplots(figsize=(6.3,3.5)); s=summary.set_index('policy').loc[order]; labels=['TWAP','Static AC','Regime-aware']; values=s.cvar_95_bps.to_numpy(); ax.scatter(labels,values,s=45,color=[PALETTE['contrast'],PALETTE['secondary'],PALETTE['positive']]);
    for label,value in zip(labels,values): ax.annotate(f'{value:.2f}',(label,value),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
    ax.set(ylabel='95% CVaR shortfall (bps)',ylim=(0,values.max()*1.1),title='Result q2 — tail-cost comparison'); save(fig,'result_q2_tail_cost')
    fig,ax=plt.subplots(figsize=(6.3,3.5)); paired_plot=paired[paired.baseline!='regime_aware_mpc'].copy(); pretty={'twap':'TWAP','static_ac':'Static AC','regime_aware_mpc':'Regime-aware MPC'}; labels=[f"{pretty[row.candidate]} − {pretty[row.baseline]}" for row in paired_plot.itertuples()]; means=paired_plot.mean_difference_bps.to_numpy(); low=paired_plot.difference_ci_low_bps.to_numpy(); high=paired_plot.difference_ci_high_bps.to_numpy(); y=np.arange(len(paired_plot)); ax.errorbar(means,y,xerr=np.vstack((means-low,high-means)),fmt='o',color=PALETTE['primary'],capsize=3); ax.axvline(0,color='black',lw=.8,ls='--'); ax.set(yticks=y,yticklabels=labels,xlabel='paired mean IS difference (bps)',title='Result q2 — paired contrasts (negative favors candidate)'); save(fig,'result_q2_paired_effects')
    fig,ax=plt.subplots(figsize=(6.3,3.5)); stress_plot=stress[stress.policy.isin(order)]; scenario_labels=['Baseline regime','Persistent severe stress'];
    for policy,color,label in [('twap',PALETTE['contrast'],'TWAP'),('static_ac',PALETTE['secondary'],'Static AC'),('regime_aware_mpc',PALETTE['positive'],'Regime-aware')]:
        d=stress_plot[stress_plot.policy==policy].set_index('scenario').reindex(['baseline_regime','persistent_severe_stress']); ax.scatter([0,1],d.cvar_95_bps,s=48,color=color,label=label)
    ax.set(xticks=[0,1],xticklabels=scenario_labels,ylabel='95% CVaR shortfall (bps)',title='Result q2 — persistent-stress stress test'); ax.legend(); save(fig,'result_q2_stress_cvar')
    fig,ax=plt.subplots(figsize=(6.3,3.5)); feasibility=np.array([100*summary.loc[summary.policy==p,'max_completion_error'].iloc[0] for p in order]); labels=['TWAP','Static AC','Regime-aware']; colors=[PALETTE['neutral'],PALETTE['secondary'],PALETTE['positive']]; ax.axhline(0,color='black',lw=.8); ax.scatter(labels,feasibility,s=62,color=colors,zorder=3)
    for label,value in zip(labels,feasibility): ax.annotate(f'{value:.2f}%', (label,value), xytext=(0,8), textcoords='offset points',ha='center',fontsize=8)
    ax.text(.5,.92,'Every simulated path completed exactly within tolerance',transform=ax.transAxes,ha='center',va='top',fontsize=8); ax.set(ylabel='maximum completion error (%)',ylim=(-.00025,.00055),title='Result q2 — explicit zero completion error'); save(fig,'result_q2_feasibility')
    # Raw q3: posterior outcomes from independently seeded holdout batches.
    fig,ax=plt.subplots(figsize=(6.3,3.5))
    for state,color,label in [(0,PALETTE['primary'],'calm truth'),(1,PALETTE['contrast'],'stressed truth')]:
        ax.hist(online_diag.loc[online_diag.regime_true==state,'posterior_stress'],bins=30,density=True,histtype='step',lw=1.3,color=color,label=label)
    ax.set(xlabel='causal stress posterior on holdout observations',ylabel='density',title='Raw q3 — holdout posterior distribution'); ax.legend(); save(fig,'raw_q3_holdout_posteriors')
    # Process q3: versioned filter parameter estimates after each completed batch.
    fig,ax=plt.subplots(figsize=(6.3,3.5)); ax.plot(online.cycle,online.p00_estimate,marker='o',color=PALETTE['primary'],label='estimated calm persistence $p_{00}$'); ax.plot(online.cycle,online.p11_estimate,marker='o',color=PALETTE['contrast'],label='estimated stress persistence $p_{11}$'); ax.axhline(.96,color=PALETTE['primary'],lw=.8,ls='--',alpha=.7); ax.axhline(.86,color=PALETTE['contrast'],lw=.8,ls='--',alpha=.7); ax.set(xlabel='recalibration cycle',ylabel='transition persistence',ylim=(0,1),title='Process q3 — completed-batch filter updates'); ax.legend(fontsize=7); save(fig,'process_q3_parameter_updates')
    # Result q3: out-of-sample filter quality by future holdout cycle.
    fig,ax=plt.subplots(figsize=(6.3,3.5)); ax.plot(online.cycle,online.holdout_accuracy,marker='o',color=PALETTE['primary'],label='holdout accuracy'); ax.plot(online.cycle,1-online.holdout_brier,marker='s',color=PALETTE['positive'],label='1 − Brier score'); ax.set(xlabel='future holdout cycle',ylabel='score (higher is better)',ylim=(0,1),title='Result q3 — independently seeded holdout quality'); ax.legend(); save(fig,'result_q3_holdout_quality')
if __name__=='__main__': main()
