import seaborn as sns
import altair as alt
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc("font",family="SimHei") ###增加了这一行
st.set_option('deprecation.showPyplotGlobalUse', False)
import igraph as ig
from matplotlib.artist import Artist
from igraph import BoundingBox, Graph, palettes
import streamlit.components.v1 as components
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder
st.set_page_config(page_title="rasch IRT demo", page_icon=" ️", layout="centered")
import utils as UT
import random
import math
import chart_studio
chart_studio.tools.set_credentials_file(username='xuzhenhua', api_key='cH74AtkJFsdgQejTCVnk')
import chart_studio.plotly as py
from plotly.graph_objects import Scatter, Layout, Figure
from streamlit_multipage import MultiPage



def model_setup(df):
    '''rasch model solve
    
    Return:
        @item_dfty: item difficulty
        @theta: case ability estimated
    '''
    n_row = len(df)
    n_col = len(df.columns)
    
    # 初始化：theta和delta均为0
    # theta: 代表患者病情难度
    # delta: 代表项目难度
    theta = [0] * n_row
    delta = [0] * n_col
    var_sum = 1
    counter = 0
    
    # 将连续数值转化成[0,1]之间的概率
    max_list = list(df.max())
    min_list = list(df.min())

    for i in list(df.columns):
        df_min = df[i].min()
        df_max = df[i].max()

        df[i] = df[i].apply(lambda x: (x - df_min) / (df_max - df_min))
        
    # 设定拟合的判断标准，可用0.05或0.01置信区间为标准。也可用次数上限为标准。这里0.05只是推算值，实际需要根据情况调整
    while var_sum > 0.05:
        print('第%s次拟合开始' % counter)
        p_list = []
       
        for n in theta:
            # 遍历每个患者
            
            for i in delta:
                # 遍历每个项目
                
                # 使用one-parameter rasch model 概率分布
                p_list.append(math.exp(n - i) / (1 + math.exp(n - i)))
                
        df_p = pd.DataFrame(np.array(p_list).reshape(n_row, n_col), index=df.index, columns=df.columns)
        
        # variance
        df_var = df_p.applymap(lambda x: x * (1 - x))
        
        # residual
        df_delta = df - df_p

        counter_1 = 0
        counter_2 = 0
        new_theta = []
        new_delta = []
         
        # 使用Newton-Raphson理论
        # update theta
        for i in theta:   
            # new theta = theta + residual / variance
            new_theta.append(i + df_delta.iloc[counter_1, :].sum() / df_var.iloc[counter_1, :].sum())
            counter_1 = counter_1 + 1
        theta = new_theta
        
        # update delta
        for n in delta:
            # new delta = delta - residual / variance
            new_delta.append(n - df_delta.iloc[:, counter_2].sum() / df_var.iloc[:, counter_2].sum())
            counter_2 = counter_2 + 1
        #delta = np.array(new_delta) - np.array(new_delta).mean()
        delta = new_delta
        
        var_sum_new = (df_delta.applymap(lambda x: x * x).sum()).sum()
        # 设定第一终点，减少的方差小于0.05
        print(var_sum_new, var_sum)
        if (var_sum - var_sum_new < 0.05) and (counter > 1):
            break
        var_sum = var_sum_new
        counter = counter + 1
        # 设定第二个终结点，推荐用50次
        if counter > 30:
            break
            
    df_y2 = df_delta.applymap(lambda x: x*x)
    df_z = df_delta.applymap(lambda x: x*x) / df_var

    item_mnsq = list(df_y2.sum() / df_var.sum())
    item_dfty = pd.DataFrame([delta, max_list, min_list, item_mnsq], columns=df.columns, index=['value', 'max', 'min', 'infit_MNSQs'])
    item_dfty = pd.DataFrame(item_dfty.values.T, columns=item_dfty.index, index=item_dfty.columns).reset_index()

    return item_dfty, theta


def train_page(st, **state):
    cols = ['Diagnosis', 'Room', 'Diet', 'exxamine', 'x-ray', 'treatment', 'operation', 
            'rehabilitation', 'Blood', 'Hmo', 'annesiology', 'material', 'drug', 
            'administration', 'psychiatry', 'injection', 'baby']
    df = pd.read_csv('drgsyyy.csv')[cols]
    st.text('选取DRG组YYY进行演示')
    st.text('所有数据共%d条，部分病案数据实例如下' % len(df))
    UT.ag_df(st, df.head(10), 150, selectable=False)
    df_train = df.copy()
    item_dfty, theta = model_setup(df_train)
    
    # show case ablity 
    theta_df = pd.DataFrame(theta, columns=['theta']).reset_index()
    fig = plt.figure()
    sns.lineplot(data=theta_df, x="index", y="theta") 
    plt.title('avg theta: %4.2f' % np.mean(theta_df['theta']))
    st.pyplot(fig)
    
    # show item difficulty
    fig = plt.figure()
    item_dfty = item_dfty.sort_values(by='value', ascending=False)
    sns.lineplot(data=item_dfty, x="index", y="value")   
    plt.xticks(rotation=20, ha='right', fontsize=7)
    plt.grid(color='silver')
    st.pyplot(fig)
    
    MultiPage.save({'df':df, "item_dfty": item_dfty, "theta": theta})

    
    
def get_closest_item(theta, item_dfty_dict):
    '''给定theta，选择难度最接近的项目
    Args:
        @theta: case ability value
        @item_dfty_dict: {'item1':v1, 'item2':v2,..., }        
    '''
    
    item_dfty = list(item_dfty_dict.values())
    item_dfty_diff = [abs(itm - theta) for itm in item_dfty]
    item = list(item_dfty_dict.keys())
    min_id = item_dfty_diff.index(min(item_dfty_diff))
    return item[min_id], item_dfty[min_id]
    
    
    
def cat_test(case_df, item_dfty, init_theta):  
    key_used = item_dfty.copy()
    
    # 去掉最小值和最大值都是0的项目
    key_used = key_used[key_used['max']>0]

    key_used = key_used.set_index('index')
    col_list = list(key_used.index)
    
    item_z = {}
    theta = init_theta
    flag = 0
    theta_list = []
    item_dfty = []
    
    rid = 0
    while len(col_list) > 0:
        st.text('*'*40)
        st.text('round:%d' % rid)
        rid = rid + 1
        # 获取最接近的item
        item_dict = dict(zip(col_list, key_used.loc[col_list, 'value']))
        item, item_value = get_closest_item(theta, item_dict)
        st.text('1. theta:%4.2f, most closest item: %s (%4.2f)' % (theta, item, item_value))
       
        col_list.remove(item)
        
        # 归一化：变成[0,1]之间的概率
        pct_i = (case_df.loc[0, item] - key_used.loc[item, 'min']) / (key_used.loc[item, 'max'] - key_used.loc[item, 'min'])
        # pct_i = min(pct_i, 1)
        # pct_i = max(pct_i, 0)
        st.text('2. item fee normalization: %4.4f->%4.4f' % (case_df.loc[0, item], pct_i))
        
        # 计算个体z值
        # 计算个体难度与项目难度之间的差异
        diff = theta - key_used.loc[item, 'value']
        # 计算预测值
        try:
            p = math.exp(diff) / (1 + math.exp(diff))
        except Exception as e:
            p = 0.9999
            
        st.text('3. item difficulty: %4.2f, residual: %4.4f, predicted item fee: %4.4f' % (key_used.loc[item, 'value'], diff, p))
        
        #　计算z值
        z = (pct_i - p) / math.sqrt(p * (1 - p))        
        st.text('4. z=%4.2f' % z)
        
        item_z[item] = round(z, 2)
        theta_list.append(theta)
        item_dfty.append(item_value)
        
        # 如果z值在合理范围内，更新theta。否则不更新theta。
        # 计算总体theta值: theta = theta + residual / variance
        if abs(z) < 3:
            old_theta = theta
            theta = theta + (pct_i - p) / (p * (1 - p))
            st.text('5. new theta(%4.2f) = theta(%4.2f) + residual(%4.2f) / variance(%4.2f)' % (theta, old_theta, (pct_i - p), (p * (1 - p))))
    
        
    z_list = list(item_z.values())
    theta = round(theta, 2)
    
 
    for i in z_list:
        if abs(i) > 2:
            ret = 'RED Label'
    ret = 'Normal'
    
    return item_z, theta, ret, theta_list, item_dfty



def predict_page(st, **state):
    if "df" not in state or "item_dfty" not in state or 'theta' not in state:
        st.warning("Go to the train Page firstly")
        return
    df = state['df'].reset_index()
    item_dfty = state['item_dfty']
    theta = state['theta']
    
    st.header('拟合出的项目难度信息')
    for f in ['value', 'infit_MNSQs']:
        item_dfty[f] = item_dfty[f].apply(lambda x: round(x, 2))
    
    UT.ag_df(st, item_dfty, 400, selectable=False)
    
    
    st.header('选择数据进行预测')
    selected = UT.ag_df(st, df, 500)[0]  
    
    case_df = df[df['index']==selected['index']].reset_index(drop=True)
    st.header('所选病案信息：')
    case_show = case_df.drop('index', axis=1).T.rename(columns={0:'实际费用'}).reset_index()
   
    case_show = case_show.merge(item_dfty, on = 'index', how='left')
    
    UT.ag_df(st, case_show, height=500, selectable=False)
    init_theta = np.mean(theta)
    
    st.header('利用CAT计算每个项目的Z值')
    item_z, theta, ret, theta_list, item_dfty = cat_test(case_df, item_dfty, init_theta)

    st.header('rasch模型预测结果')
    st.info('当前病案: %s' % ret)
    st.text('患者病情严重程度: %4.2f' % theta)
    
    for k, v in item_z.items():
        item_z[k] = [v]
    item_df = pd.DataFrame(item_z).T.rename(columns={0:'z_score'})

    # 图1：item与theta变化曲线
    trace0 = Scatter(
        x=list(range(len(theta_list))),
        y=theta_list,
        name='theta',
        mode='markers+lines',
        marker=dict(
            size=10,    # 设置点的宽度
            color='rgba(0, 0, 200, 0.8)',   # 设置点的颜色
            ),
        )    
    trace1 = Scatter(
        x=list(range(len(theta_list))),
        y=item_dfty,
        name='item',
        mode='markers+lines',
        marker=dict(
            size=10,    # 设置点的宽度
            color='rgba(152, 0, 0, 0.8)',   # 设置点的颜色
            ),
        )       
    data = [trace0, trace1]
    layout = Layout(
        title='Styled Scatter',
        yaxis=dict(zeroline=True),  # 显示y轴的0刻度线
        xaxis=dict(zeroline=False, tickmode='array', tickvals = list(range(len(item_df))), ticktext = item_df.index.tolist())  # 不显示x轴的0刻度线
        )
    fig = Figure(data, layout=layout)
    st.plotly_chart(fig, filename='CAT计算过程')
    
    
    # 图2：每个项目的结果图   
    trace0 = Scatter(
        x=list(range(len(item_df))),
        y=item_df['z_score'],
        name='项目难度Z-Score',
        mode='markers',
        marker=dict(
            size=10,    # 设置点的宽度
            color='rgba(152, 0, 0, 0.8)',   # 设置点的颜色
            ),
        )
    trace1 = Scatter(
        x=list(range(len(item_df))),
        y=[-2]*len(item_df),
        name='正常值下限',
        mode='lines',
        line=dict(
            color='green'
        )
    )

    trace2 = Scatter(
        x=list(range(len(item_df))),
        y=[2]*len(item_df),
        name='正常值上限',
        mode='lines',
        line=dict(
            color='green'
        )
    )

    data = [trace0, trace1, trace2]
    layout = Layout(
        title='Styled Scatter',
        yaxis=dict(zeroline=True),  # 显示y轴的0刻度线
        xaxis=dict(zeroline=False, tickmode='array', tickvals = list(range(len(item_df))), ticktext = item_df.index.tolist())  # 不显示x轴的0刻度线
        )
    fig = Figure(data, layout=layout)
    st.plotly_chart(fig, filename='DRG费用异常')



app = MultiPage()
app.st = st

app.hide_menu = True
app.hide_navigation = True
app.add_app("item难度计算", train_page)
app.add_app('CAT', predict_page)

app.run()


