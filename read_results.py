import numpy as np
import xgb_utils as xu

fname = 'log/xg_cv_eta_0.0125_md_14_ss_0.9_mw_2_g_1_ct_0.8_nr_2000.csv'
xu.plot_cv_results(fname)
