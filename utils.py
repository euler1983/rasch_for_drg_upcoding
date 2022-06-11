import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
#import igraph as ig
#from matplotlib.artist import Artist
#from igraph import BoundingBox, Graph, palettes
import streamlit.components.v1 as components
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder
 

def ag_df(st, df, height, selectable=True):    
    options_builder = GridOptionsBuilder.from_dataframe(df)
    options_builder.configure_default_column(groupable=True, flex=1, value=True, enableRowGroup=True, aggFunc='sum', editable=True, wrapText=True, autoHeight=True)
    if selectable:
        options_builder.configure_selection(selection_mode='single', use_checkbox=True, pre_selected_rows=[0])
    options_builder.configure_pagination(True, paginationPageSize=10)
                                        
    grid_options = options_builder.build()
    
    fc = True
    if df.shape[1] > 5:
        fc = False
        
    grid_return = AgGrid(df, 
                         grid_options, 
                         theme='blue', 
                         fit_columns_on_grid_load=fc,
                         update_mode = GridUpdateMode.SELECTION_CHANGED,
                         data_return_mode = DataReturnMode.FILTERED,
                         height = height
                        )
    
    selected_rows = grid_return['selected_rows']
    return selected_rows



