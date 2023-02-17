from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
from pyspark.ml.feature import Bucketizer
from pyspark.sql.functions import min,max
from pyspark.sql.functions import lit,array,col,split
from typing import List
import re
import warnings
import time
import os


def woebin2_init_bin(dtm, init_count_distr:float, breaks:List[float], spl_val:List[float]):
    '''
    initial binning
    
    Params
    ------
    dtm: pyspark dataframe
    init_count_distr: the minimal precentage in the fine binning process
    breaks: breaks
    breaks: breaks list
    spl_val: speical values list
    
    Returns
    ------
    dict
        returns a dict with initial binning and special_value binning
    '''
    
    if ((dict(dtm.dtypes)['value'] == "double") or (dict(dtm.dtypes)['value'] == "int")): # numeric variable
        # breaks vector & outlier
        iq:list = dtm.approxQuantile('value',[0.01, 0.25, 0.75, 0.99],relativeError=0)
        iqr:float = iq[2]-iq[1]
        if iqr == 0:
          prob_down = 0.01
          prob_up = 0.99
        else:
          prob_down = 0.25
          prob_up = 0.75
        xvalue_rm_outlier = dtm[(dtm['value'] >= iq[1]-3*iqr) & (dtm['value'] <= iq[2]+3*iqr)]
        # number of initial binning
        n:float = np.trunc(1/init_count_distr)
        len_uniq_x:int = xvalue_rm_outlier.select('value').distinct().count()
        if len_uniq_x < n: n = len_uniq_x
        
        minimum = xvalue_rm_outlier.agg(min("value")).collect()[0][0]
        maximum = xvalue_rm_outlier.agg(max("value")).collect()[0][0]
        # initial breaks
        brk = list(xvalue_rm_outlier.select('value').distinct().collect()[0]) if len_uniq_x < 10 else pretty(minimum,maximum, n)
        minimum:float = dtm.agg(min("value")).collect()[0][0]
        maximum:float = dtm.agg(max("value")).collect()[0][0]
        brk:List[float] = list(filter(lambda x: x>minimum and x<=maximum, brk))
        brk = [float('-inf')] + sorted(brk) + [float('inf')]
        # initial binning datatable
        # cut
        labels:List[str] = ['[{},{})'.format(brk[i], brk[i+1]) for i in range(len(brk)-1)]
        bucketizer = Bucketizer(splits=brk,inputCol="value", outputCol="bin")
        df_buck = bucketizer.setHandleInvalid("keep").transform(dtm)

        label_array = array(*(lit(label) for label in labels))
        df_buck = df_buck.withColumn("bin", label_array.getItem(col("bin").cast("integer")))
       
        init_bin = df_buck.groupby('bin').pivot("y").count()

  
        # check empty bins for unmeric variable
        init_bin = check_empty_bins(df_buck,init_bin)
        init_bin = init_bin.withColumn('order', split(init_bin['bins'], ',').getItem(0))
        init_bin = init_bin.withColumn('order', split(init_bin['order'], '\[').getItem(1).cast('Float'))
        init_bin = init_bin.sort('order')
        init_bin = init_bin.drop('order')
        init_bin:pd.DataFrame = init_bin.toPandas()
        init_bin.rename(columns={'0':'good','1':'bad'},inplace=True)
        init_bin.fillna(0,inplace=True)
        init_bin = init_bin.assign(
          variable = df_buck.select('variable').collect()[0][0], #! time can be reduced here
          brkp = lambda x: [float(re.match('^\[(.*),.+', i).group(1)) for i in x['bins']],
          badprob = lambda x: x['bad']/(x['bad']+x['good'])
        )[['variable', 'bins', 'brkp', 'good', 'bad', 'badprob']]
    else: # other type variable
        # initial binning datatable
        init_bin = dtm.groupby('value').pivot("y").count()
        init_bin = init_bin.toPandas()
        init_bin.rename(columns={'n0':'good','n1':'bad'})\
        .assign(
          variable = dtm.select('variable').collect()[0][0],
          badprob = lambda x: x['bad']/(x['bad']+x['good'])
        ).reset_index()
        # order by badprob if is.character
        if dtm.value.dtype.name not in ['category', 'bool']:
            init_bin = init_bin.sort_values(by='badprob').reset_index()
        # add index as brkp column
        init_bin = init_bin.assign(brkp = lambda x: x.index)\
            [['variable', 'value', 'brkp', 'good', 'bad', 'badprob']]\
            .rename(columns={'value':'bin'})
    
    init_bin = init_bin.fillna(0)
    # remove brkp that good == 0 or bad == 0 ------
    while len(init_bin.query('(good==0) or (bad==0)')) > 0:
        # brkp needs to be removed if good==0 or bad==0
        rm_brkp = init_bin.assign(count = lambda x: x['good']+x['bad'])\
        .assign(
          count_lag  = lambda x: x['count'].shift(1).fillna(dtm.count()+1),
          count_lead = lambda x: x['count'].shift(-1).fillna(dtm.count()+1)
        ).assign(merge_tolead = lambda x: x['count_lag'] > x['count_lead'])\
        .query('(good==0) or (bad==0)')\
        .query('count == count.min()').iloc[0,]
        # set brkp to lead's or lag's
        shift_period:int = -1 if rm_brkp['merge_tolead'] else 1
        init_bin = init_bin.assign(brkp2  = lambda x: x['brkp'].shift(shift_period))\
        .assign(brkp = lambda x:np.where(x['brkp'] == rm_brkp['brkp'], x['brkp2'], x['brkp']))
        # groupby brkp
        init_bin = init_bin.groupby('brkp').agg({
          'variable':lambda x: np.unique(x),
          'bins': lambda x: '%,%'.join(x),
          'good': sum,
          'bad': sum
        }).assign(badprob = lambda x: x['bad']/(x['good']+x['bad']))\
        .reset_index()
    # format init_bin
    if ((dict(dtm.dtypes)['value'] == "double") or (dict(dtm.dtypes)['value'] == "int")):
        init_bin = init_bin\
        .assign(bin = lambda x: [re.sub(r'(?<=,).+%,%.+,', '', i) if ('%,%' in i) else i for i in x['bins']])\
        .assign(brkp = lambda x: [float(re.match('^\[(.*),.+', i).group(1)) for i in x['bin']])
    # return 
    return {'initial_binning':init_bin}

def woebin2_breaks(dtm, brk, spl_val):
    brk = [float('-inf')] + sorted(brk) + [float('inf')]
    # initial binning datatable
    # cut
    labels:List[str] = ['[{},{})'.format(brk[i], brk[i+1]) for i in range(len(brk)-1)]
    bucketizer = Bucketizer(splits=brk,inputCol="value", outputCol="bin")
    df_buck = bucketizer.setHandleInvalid("keep").transform(dtm)

    label_array = array(*(lit(label) for label in labels))
    df_buck = df_buck.withColumn("bin", label_array.getItem(col("bin").cast("integer")))
    
    init_bin = df_buck.groupby('bin').pivot("y").count()


    # check empty bins for unmeric variable
    init_bin = check_empty_bins(df_buck,init_bin)
    init_bin = init_bin.withColumn('order', split(init_bin['bins'], ',').getItem(0))
    init_bin = init_bin.withColumn('order', split(init_bin['order'], '\[').getItem(1).cast('Float'))
    init_bin = init_bin.sort('order')
    init_bin = init_bin.drop('order')
    init_bin:pd.DataFrame = init_bin.toPandas()
    init_bin.rename(columns={'0':'good','1':'bad'},inplace=True)
    init_bin.fillna(0,inplace=True)
    init_bin = init_bin.assign(
      variable = df_buck.select('variable').collect()[0][0], #! time can be reduced here
      brkp = lambda x: [float(re.match('^\[(.*),.+', i).group(1)) for i in x['bins']],
      badprob = lambda x: x['bad']/(x['bad']+x['good'])
    )[['variable', 'bins', 'brkp', 'good', 'bad', 'badprob']]

    init_bin = init_bin.fillna(0)
    # format init_bin
    if ((dict(dtm.dtypes)['value'] == "double") or (dict(dtm.dtypes)['value'] == "int")):
        init_bin = init_bin\
        .assign(bin = lambda x: [re.sub(r'(?<=,).+%,%.+,', '', i) if ('%,%' in i) else i for i in x['bins']])\
        .assign(brkp = lambda x: [float(re.match('^\[(.*),.+', i).group(1)) for i in x['bin']])
    init_bin.drop('bins',axis=1,inplace=True)
    # return 
    return {'initial_binning':init_bin}

def pretty(low:float, high:float, n:int) -> np.ndarray:
    '''
    pretty breakpoints, the same as pretty function in R
    
    Params
    ------
    low: minimal value 
    high: maximal value 
    n: number of intervals
    
    Returns
    ------
    numpy.ndarray
        returns a breakpoints array
    '''
    # nicenumber
    def nicenumber(x):
        exp = np.floor(np.log10(abs(x)))
        f   = abs(x) / 10**exp
        if f < 1.5:
            nf = 1.
        elif f < 3.:
            nf = 2.
        elif f < 7.:
            nf = 5.
        else:
            nf = 10.
        return np.sign(x) * nf * 10.**exp
    # pretty breakpoints
    d     = abs(nicenumber((high-low)/(n-1)))
    miny  = np.floor(low  / d) * d
    maxy  = np.ceil (high / d) * d
    return np.arange(miny, maxy+0.5*d, d)

def check_empty_bins(dtm, binning):
    # check empty bins
    lists = dtm.select('bin').distinct().collect()
    bin_list = []
    for i in range(len(lists)):
        bin_list.append(lists[i][0])
    if 'nan' in bin_list: 
        bin_list.remove('nan')
    binleft = set([re.match(r'\[(.+),(.+)\)', i).group(1) for i in bin_list]).difference(set(['-inf', 'inf']))
    binright = set([re.match(r'\[(.+),(.+)\)', i).group(2) for i in bin_list]).difference(set(['-inf', 'inf']))
    if binleft != binright:
        bstbrks = sorted(list(map(float, ['-inf'] + list(binright) + ['inf'])))
        labels = ['[{},{})'.format(bstbrks[i], bstbrks[i+1]) for i in range(len(bstbrks)-1)]
        bucketizer = Bucketizer(splits=bstbrks,inputCol="value", outputCol="bins")
        df_buck = bucketizer.setHandleInvalid("keep").transform(dtm)
        label_array = array(*(lit(label) for label in labels))
        df_buck = df_buck.withColumn(
        "bins", label_array.getItem(col("bins").cast("integer"))
            )
        binning = df_buck.groupby('bins').pivot("y").count()
    binning = binning.withColumnRenamed('bin', 'bins')

    return binning

def miv_01(good:pd.Series, bad:pd.Series) -> pd.Series:
    # iv calculation
    infovalue = pd.DataFrame({'good':good,'bad':bad}) \
      .replace(0, 0.9) \
      .assign(
        DistrBad = lambda x: x.bad/sum(x.bad),
        DistrGood = lambda x: x.good/sum(x.good)
      ) \
      .assign(iv = lambda x: (x.DistrBad-x.DistrGood)*np.log(x.DistrBad/x.DistrGood)) \
      .iv
    # return iv
    return infovalue

def woe_01(good:pd.Series, bad:pd.Series) -> pd.DataFrame:
    # woe calculation
    woe = pd.DataFrame({'good':good,'bad':bad}) \
      .replace(0, 0.9) \
      .assign(
        DistrBad = lambda x: x.bad/sum(x.bad),
        DistrGood = lambda x: x.good/sum(x.good)
      ) \
      .assign(woe = lambda x: np.log(x.DistrBad/x.DistrGood)) \
      .woe
    # return woe
    return woe

def binning_format(binning):
    '''
    format binning dataframe
    
    Params
    ------
    binning: with columns of variable, bin, good, bad
    
    Returns
    ------
    DataFrame
        binning dataframe with columns of 'variable', 'bin', 
        'count', 'count_distr', 'good', 'bad', 'badprob', 'woe', 
        'bin_iv', 'total_iv',  'breaks', 'is_special_values'
    '''
    
    binning['count'] = binning['good'] + binning['bad']
    binning['count_distr'] = binning['count']/sum(binning['count'])
    binning['badprob'] = binning['bad']/binning['count']
    binning['woe'] = woe_01(binning['good'],binning['bad'])
    binning['bin_iv'] = miv_01(binning['good'],binning['bad'])
    binning['total_iv'] = binning['bin_iv'].sum()
    # breaks
    binning['breaks'] = binning['bin']
    if any([r'[' in str(i) for i in binning['bin']]):
        def re_extract_all(x): 
            gp23 = re.match(r"^\[(.*), *(.*)\)((%,%missing)*)", x)
            breaks_string = x if gp23 is None else gp23.group(2)+gp23.group(3)
            return breaks_string
        binning['breaks'] = [re_extract_all(i) for i in binning['bin']]
    # is_sv    
    binning['is_special_values'] = binning['is_sv']
    # return
    return binning[['variable', 'bin', 'count', 'count_distr', 'good', 'bad', 'badprob', 'woe', 'bin_iv', 'total_iv',  'breaks', 'is_special_values']]

def woebin2(dtm, breaks=None, spl_val=None, 
            init_count_distr=0.02, count_distr_limit=0.05, 
            stop_limit=0.1, bin_num_limit=8, method="tree"):
    '''
    provides woe binning for only two series
    
    Params
    ------
    
    
    Returns
    ------
    DataFrame
        
    '''
    # binning
    if breaks is not None:
        # 1.return binning if breaks provided
        bin_list = woebin2_breaks(dtm=dtm, brk=breaks, spl_val=spl_val)
    else:
        if stop_limit == 'N':
            # binning of initial & specialvalues
            bin_list = woebin2_init_bin(dtm, init_count_distr=init_count_distr, breaks=breaks, spl_val=spl_val)
        else:     
              # 2.chimerge optimal binning
              bin_list = woebin2_chimerge(
                dtm, init_count_distr=init_count_distr, count_distr_limit=count_distr_limit, 
                stop_limit=stop_limit, bin_num_limit=bin_num_limit, breaks=breaks, spl_val=spl_val)
    # rbind binning_sv and binning
    binning = pd.concat(bin_list, keys=bin_list.keys()).reset_index()\
              .assign(is_sv = lambda x: x.level_0 =='binning_sv')
    # return
    return binning_format(binning)

def woebin2_chimerge(dtm, init_count_distr=0.02, count_distr_limit=0.05, 
                     stop_limit=0.1, bin_num_limit=8, breaks=None, spl_val=None):
    '''
    binning using chimerge method
    
    Params
    ------
    dtm:
    init_count_distr:
    count_distr_limit:
    stop_limit:
    bin_num_limit:
    breaks:
    spl_val:
    
    Returns
    ------
    dict
        returns a dict with initial binning and special_value binning
    '''
    # [chimerge](http://blog.csdn.net/qunxingvip/article/details/50449376)
    # [ChiMerge:Discretization of numeric attributs](http://www.aaai.org/Papers/AAAI/1992/AAAI92-019.pdf)
    # chisq = function(a11, a12, a21, a22) {
    #   A = list(a1 = c(a11, a12), a2 = c(a21, a22))
    #   Adf = do.call(rbind, A)
    #
    #   Edf =
    #     matrix(rowSums(Adf), ncol = 1) %*%
    #     matrix(colSums(Adf), nrow = 1) /
    #     sum(Adf)
    #
    #   sum((Adf-Edf)^2/Edf)
    # }
    # function to create a chisq column in initial_binning
    def add_chisq(initial_binning):
        chisq_df = pd.melt(initial_binning, 
          id_vars=["brkp", "variable", "bin"], value_vars=["good", "bad"],
          var_name='goodbad', value_name='a')\
        .sort_values(by=['goodbad', 'brkp']).reset_index(drop=True)
        ###
        chisq_df['a_lag'] = chisq_df.groupby('goodbad')['a'].apply(lambda x: x.shift(1))#.reset_index(drop=True)
        chisq_df['a_rowsum'] = chisq_df.groupby('brkp')['a'].transform(lambda x: sum(x))#.reset_index(drop=True)
        chisq_df['a_lag_rowsum'] = chisq_df.groupby('brkp')['a_lag'].transform(lambda x: sum(x))#.reset_index(drop=True)
        ###
        chisq_df = pd.merge(
          chisq_df.assign(a_colsum = lambda df: df.a+df.a_lag), 
          chisq_df.groupby('brkp').apply(lambda df: sum(df.a+df.a_lag)).reset_index(name='a_sum'))\
        .assign(
          e = lambda df: df.a_rowsum*df.a_colsum/df.a_sum,
          e_lag = lambda df: df.a_lag_rowsum*df.a_colsum/df.a_sum
        ).assign(
          ae = lambda df: (df.a-df.e)**2/df.e + (df.a_lag-df.e_lag)**2/df.e_lag
        ).groupby('brkp').apply(lambda x: sum(x.ae)).reset_index(name='chisq')
        # return
        return pd.merge(initial_binning.assign(count = lambda x: x['good']+x['bad']), chisq_df, how='left')
    # initial binning
    bin_list = woebin2_init_bin(dtm, init_count_distr=init_count_distr, breaks=breaks, spl_val=spl_val)
    initial_binning = bin_list['initial_binning']
    # return initial binning if its row number equals 1
    if len(initial_binning.index)==1: 
        return {'binning':initial_binning}

    # dtm_rows
    dtm_rows:int = dtm.count()   
    # chisq limit
    from scipy.special import chdtri
    chisq_limit:float = chdtri(1, stop_limit)
    # binning with chisq column
    binning_chisq:pd.DataFrame = add_chisq(initial_binning)
    
    # param
    bin_chisq_min:float = binning_chisq.chisq.min()
    bin_count_distr_min:float = (binning_chisq['count']/dtm_rows).min()
    bin_nrow:int = len(binning_chisq.index)
    # remove brkp if chisq < chisq_limit
    while bin_chisq_min < chisq_limit or bin_count_distr_min < count_distr_limit or bin_nrow > bin_num_limit:
        # brkp needs to be removed
        if bin_chisq_min < chisq_limit:
            rm_brkp:pd.Series = binning_chisq.assign(merge_tolead = False).sort_values(by=['chisq', 'count']).iloc[0,]
        elif bin_count_distr_min < count_distr_limit:
            rm_brkp = binning_chisq.assign(
              count_distr = lambda x: x['count']/sum(x['count']),
              chisq_lead = lambda x: x['chisq'].shift(-1).fillna(float('inf'))
            ).assign(merge_tolead = lambda x: x['chisq'] > x['chisq_lead'])
            # replace merge_tolead as True
            rm_brkp.loc[np.isnan(rm_brkp['chisq']), 'merge_tolead']=True
            # order select 1st
            rm_brkp = rm_brkp.sort_values(by=['count_distr']).iloc[0,]
        elif bin_nrow > bin_num_limit:
            rm_brkp = binning_chisq.assign(merge_tolead = False).sort_values(by=['chisq', 'count']).iloc[0,]
        else:
            break
        # set brkp to lead's or lag's
        shift_period:int = -1 if rm_brkp['merge_tolead'] else 1
        binning_chisq:pd.DataFrame = binning_chisq.assign(brkp2  = lambda x: x['brkp'].shift(shift_period))\
        .assign(brkp = lambda x:np.where(x['brkp'] == rm_brkp['brkp'], x['brkp2'], x['brkp']))
        # groupby brkp
        binning_chisq = binning_chisq.groupby('brkp').agg({
          'variable':lambda x:x,
          'bin': lambda x: '%,%'.join(x),
          'good': sum,
          'bad': sum
        }).assign(badprob = lambda x: x['bad']/(x['good']+x['bad']))\
        .reset_index()
        # update
        ## add chisq to new binning dataframe
        binning_chisq = add_chisq(binning_chisq)
        ## param
        bin_nrow = len(binning_chisq.index)
        if bin_nrow == 1:
            break
        bin_chisq_min:float = binning_chisq.chisq.min()
        bin_count_distr_min:float = (binning_chisq['count']/dtm_rows).min()
        
    # format init_bin # remove (.+\\)%,%\\[.+,)
    if ((dict(dtm.dtypes)['value'] == "double") or (dict(dtm.dtypes)['value'] == "int")):
        binning_chisq = binning_chisq\
        .assign(bin = lambda x: [re.sub(r'(?<=,).+%,%.+,', '', i) if ('%,%' in i) else i for i in x['bin']])\
        .assign(brkp = lambda x: [float(re.match('^\[(.*),.+', i).group(1)) for i in x['bin']])
    # return 
    return {'binning':binning_chisq}
     
def str_to_list(x):
    if x is not None and isinstance(x, str):
        x = [x]
    return x

def x_variable(dat, y, x, var_skip=None):
    y = str_to_list(y)
    if var_skip is not None: y = y + str_to_list(var_skip)
    x_all = list(set(dat.columns) - set(y))
    
    if x is None:
        x = x_all
    else:
        x = str_to_list(x)
            
        if any([i in list(x_all) for i in x]) is False:
            x = x_all
        else:
            x_notin_xall = set(x).difference(x_all)
            if len(x_notin_xall) > 0:
                warnings.warn("Incorrect inputs; there are {} x variables are not exist in input data, which are removed from x. \n({})".format(len(x_notin_xall), ', '.join(x_notin_xall)))
                x = set(x).intersection(x_all)
            
    return list(x)

def woebin(dt, y, x=None, 
           var_skip=None, breaks_list=None, special_values=None, 
           stop_limit=0.1, count_distr_limit=0.05, bin_num_limit=8, 
           # min_perc_fine_bin=0.02, min_perc_coarse_bin=0.05, max_num_bin=8, 
           positive="bad|1", no_cores=None, print_step=0, method="tree",
           ignore_const_cols=True, ignore_datetime_cols=True, 
           check_cate_num=True, replace_blank=True, 
           save_breaks_list=None, **kwargs):
    '''
    WOE Binning
    ------
    `woebin` generates optimal binning for numerical, factor and categorical 
    variables using methods including tree-like segmentation or chi-square 
    merge. woebin can also customizing breakpoints if the breaks_list or 
    special_values was provided.
    
    The default woe is defined as ln(Distr_Bad_i/Distr_Good_i). If you 
    prefer ln(Distr_Good_i/Distr_Bad_i), please set the argument `positive` 
    as negative value, such as '0' or 'good'. If there is a zero frequency 
    class when calculating woe, the zero will replaced by 0.99 to make the 
    woe calculable.
    
    Params
    ------
    dt: A data frame with both x (predictor/feature) and y (response/label) variables.
    y: Name of y variable.
    x: Name of x variables. Default is None. If x is None, 
      then all variables except y are counted as x variables.
    var_skip: Name of variables that will skip for binning. Defaults to None.
    breaks_list: List of break points, default is None. 
      If it is not None, variable binning will based on the 
      provided breaks.
    special_values: the values specified in special_values 
      will be in separate bins. Default is None.
    count_distr_limit: The minimum percentage of final binning 
      class number over total. Accepted range: 0.01-0.2; default 
      is 0.05.
    stop_limit: Stop binning segmentation when information value 
      gain ratio less than the stop_limit, or stop binning merge 
      when the minimum of chi-square less than 'qchisq(1-stoplimit, 1)'. 
      Accepted range: 0-0.5; default is 0.1.
    bin_num_limit: Integer. The maximum number of binning.
    positive: Value of positive class, default "bad|1".
    no_cores: Number of CPU cores for parallel computation. 
      Defaults None. If no_cores is None, the no_cores will 
      set as 1 if length of x variables less than 10, and will 
      set as the number of all CPU cores if the length of x variables 
      greater than or equal to 10.
    print_step: A non-negative integer. Default is 1. If print_step>0, 
      print variable names by each print_step-th iteration. 
      If print_step=0 or no_cores>1, no message is print.
    method: Optimal binning method, it should be "tree" or "chimerge". 
      Default is "tree".
    ignore_const_cols: Logical. Ignore constant columns. Defaults to True.
    ignore_datetime_cols: Logical. Ignore datetime columns. Defaults to True.
    check_cate_num: Logical. Check whether the number of unique values in 
      categorical columns larger than 50. It might make the binning process slow 
      if there are too many unique categories. Defaults to True.
    replace_blank: Logical. Replace blank values with None. Defaults to True.
    save_breaks_list: The file name to save breaks_list. Default is None.
    
    Returns
    ------
    dictionary
        Optimal or customized binning dataframe.
    
    Examples
    ------
    import scorecardpy as sc
    import pandas as pd
    
    # load data
    dat = sc.germancredit()
    
    # Example I
    # binning of two variables in germancredit dataset
    bins_2var = sc.woebin(dat, y = "creditability", 
      x = ["credit_amount", "purpose"])
    
    # Example II
    # binning of the germancredit dataset
    bins_germ = sc.woebin(dat, y = "creditability")
    
    # Example III
    # customizing the breakpoints of binning
    dat2 = pd.DataFrame({'creditability':['good','bad']}).sample(50, replace=True)
    dat_nan = pd.concat([dat, dat2], ignore_index=True)
    
    breaks_list = {
      'age_in_years': [26, 35, 37, "Inf%,%missing"],
      'housing': ["own", "for free%,%rent"]
    }
    special_values = {
      'credit_amount': [2600, 9960, "6850%,%missing"],
      'purpose': ["education", "others%,%missing"]
    }
    
    bins_cus_brk = sc.woebin(dat_nan, y="creditability",
      x=["age_in_years","credit_amount","housing","purpose"],
      breaks_list=breaks_list, special_values=special_values)
    '''
    # start time
    start_time:float = time.time()
    dt = remove_dots_in_column_names(dt)
    print("column names after removing '.' : ")
    print(dt.columns)
    ## print_info
    print_info = kwargs.get('print_info', True)
    ## init_count_distr
    min_perc_fine_bin = kwargs.get('min_perc_fine_bin', None)
    init_count_distr = kwargs.get('init_count_distr', min_perc_fine_bin)
    if init_count_distr is None: init_count_distr:float = 0.02
    ## count_distr_limit
    min_perc_coarse_bin = kwargs.get('min_perc_coarse_bin', None)
    if min_perc_coarse_bin is not None: count_distr_limit = min_perc_coarse_bin
    ## bin_num_limit
    max_num_bin:int = kwargs.get('max_num_bin', None)
    if max_num_bin is not None: bin_num_limit = max_num_bin
    
    # print infomation
    if print_info: print('[INFO] creating woe binning ...')
    
    
    if isinstance(y, str):
        y = [y]
    if isinstance(x, str) and x is not None:
        x = [x]
    if x is not None: 
        dt = dt[y+x]
    # x variable names
    xs:List[str] = x_variable(dt, y, x, var_skip)
    xs_len:int = len(xs)
   
    # stop_limit range
    if stop_limit<0 or stop_limit>0.5 or not isinstance(stop_limit, (float, int)):
        warnings.warn("Incorrect parameter specification; accepted stop_limit parameter range is 0-0.5. Parameter was set to default (0.1).")
        stop_limit = 0.1
    # init_count_distr range
    if init_count_distr<0.01 or init_count_distr>0.2 or not isinstance(init_count_distr, (float, int)):
        warnings.warn("Incorrect parameter specification; accepted init_count_distr parameter range is 0.01-0.2. Parameter was set to default (0.02).")
        init_count_distr = 0.02
    # count_distr_limit
    if count_distr_limit<0.01 or count_distr_limit>0.2 or not isinstance(count_distr_limit, (float, int)):
        warnings.warn("Incorrect parameter specification; accepted count_distr_limit parameter range is 0.01-0.2. Parameter was set to default (0.05).")
        count_distr_limit = 0.05
    # bin_num_limit
    if not isinstance(bin_num_limit, (float, int)):
        warnings.warn("Incorrect inputs; bin_num_limit should be numeric variable. Parameter was set to default (8).")
        bin_num_limit = 8
    # method
    if method not in ["tree", "chimerge"]:
        warnings.warn("Incorrect inputs; method should be tree or chimerge. Parameter was set to default (tree).")
        method = "tree"
    ### ### 
    # binning for each x variable     
    # ylist to str
    y = y[0]
    # binning for variables

      # create empty bins dict
    bins = {}
    for i in np.arange(xs_len):
        x_i:str = xs[i]
        # print(x_i)
        # print xs
        if print_step>0 and bool((i+1)%print_step): 
            print(('{:'+str(len(str(xs_len)))+'.0f}/{} {}').format(i, xs_len, x_i), flush=True)
        # woebining on one variable
        dtm = dt[[x_i,y]]
        dtm = dtm.withColumn("variable", lit(x_i))
        dtm = dtm.withColumnRenamed(x_i,"value")
        dtm = dtm.withColumnRenamed(y,"y")
        dtm.printSchema()
        bins[x_i] = woebin2(
          dtm,
          breaks = breaks_list[x_i] if (breaks_list is not None) and (x_i in breaks_list.keys()) else None,
          spl_val=special_values[x_i] if (special_values is not None) and (x_i in special_values.keys()) else None,
          init_count_distr=init_count_distr,
          count_distr_limit=count_distr_limit,
          stop_limit=stop_limit, 
          bin_num_limit=bin_num_limit,
          method=method
        )

    # runingtime
    runingtime = time.time() - start_time
    if runingtime >= 10 and print_info:
        # print(time.strftime("%H:%M:%S", time.gmtime(runingtime)))
        print('Binning on {} rows and {} columns in {}'.format(dt.count(), len(dt.columns), time.strftime("%H:%M:%S", time.gmtime(runingtime))))
    # return
    return bins

def remove_dots_in_column_names(df):
  # Use a list comprehension to check for columns with a dot in the name
  columns_to_rename:list = [col for col in df.columns if "." in col]

  # Use expr to rename the columns that contain a dot
  for col in columns_to_rename:
      df = df.withColumnRenamed(col, col.replace(".", "_"))
  return df

if __name__ == '__main__':
    spark = SparkSession.builder.appName("SBM").config("spark.memory.offHeap.enabled","true")\
        .config("spark.executor.memory", "4g")\
        .config("spark.driver.memory", "4g")\
        .config("spark.memory.offHeap.enabled","true")\
        .config("spark.memory.offHeap.size","4g") .getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    # df = spark.read.csv("sample_py_df.csv",sep=",", header=True, inferSchema=True)
    # df = df.select("M97","def_trig")
    data = pd.DataFrame({'M97': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
                         'def_trig': [0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]})
    df = spark.createDataFrame(data)
    # print(len(df.columns),df.columns)
    #df = df.drop('Date.1','dummylockdown')
    # print(len(df.columns),df.columns)
    bins = woebin(df,y='def_trig',method='chimerge',breaks_list={'M97':[0.5,1.0,1.5,2.0]})

    bins['M97'].drop('variable',axis=1,inplace=True)
    # bins['M98'].drop('variable',axis=1,inplace=True)
    # bins['M100'].drop('variable',axis=1,inplace=True)

    iv = []
    print(bins['M97'])
    # print(bins['M98']['bin'])
    # print(bins['M100']['bin'])

    print(bins['M97']['woe'])
    # print(bins['M98']['woe'])
    # print(bins['M100']['woe'])
    #print(bins['dummylockdown']['woe'])
    
    print(bins['M97']['total_iv'][0])
    # print(bins['M98']['total_iv'][0])
    # print(bins['M100']['total_iv'][0])
    #print(bins['dummylockdown']['total_iv'][0])
    for i in bins.keys():
        iv.append(bins[i]['total_iv'][0])
        print(i,bins[i]['total_iv'][0])
    print(iv)
    col_selected =[]
    for i in iv:    
      if ((i >= 0.1) & (i <= 0.5)):
          col_selected.append(i)
