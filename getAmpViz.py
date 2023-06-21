import numpy as np
import pandas as pd
pd.options.plotting.backend = "plotly"
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
# print(pd.__version__)

class FluorVis:
    probe: str
    dfExcel: pd.DataFrame
    def __init__(self, probe, dfExcel):
        self.probe = probe
        self.dfExcel = dfExcel
    def dfCreate(self): # make a df from the excel data
        # get the sheetnames from the excel file
        sheetNames = list(self.dfExcel.keys())
        # find probe in sheet names
        probe_sheet = [sheet for sheet in sheetNames if self.probe in sheet]
        # create dataframe from sheet name
        df = self.dfExcel[probe_sheet[0]]
        # assign first row to be column header
        df.columns = df.iloc[0]
        # drop the first row (duplicate of header)
        df = df.drop(df.index[0])
        # convert all columns to numberic
        df = df.apply(pd.to_numeric, errors='coerce')
        # make index the first column
        df.set_index('Cycle', inplace=True)
        # fluor values less than 1 get 0
        df = df.where(df>=1, 0)
        return df
    def getMaxRow(self):
        df = self.dfCreate()
        # get max value for each row and convert to list
        maxRow = df.loc[df.index == 40].values.flatten().tolist()
        # round every element in list to one decimal place, convert to str
        maxRowRound = [str(round(x,1)) for x in maxRow]
        return maxRowRound
    def getTargetName(self):
        # get quan result sheet from dfExcel workbook
        quanResult = self.dfExcel['Quan. Result']
        # make the first row column headers
        new_header = quanResult.iloc[0] #grab the first row for the header
        quanResult = quanResult[1:] #take the data less the header row
        quanResult.columns = new_header #set the header row as the df header
        # find the value in "Target" column when "Dye" = "FAM"
        targetName = quanResult.loc[quanResult['Dye'] == self.probe, 'Target'].iloc[0]
        return targetName
    def plotAllWells(self):
        df = self.dfCreate()
        # make plot of wells
        fig = px.line(
            df,
            x=df.index,
            y=list(df.columns),
            color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(
            height=500,
            width=1000,
            xaxis_title='Cycle',
            yaxis_title='RFU ({})'.format(self.probe),
            title_text='Cycle vs Fluorescence {} ({})'.format(self.getTargetName(),self.probe),
            legend_title='Well')
        return fig.show()
    def plotIndWells(self):
        df = self.dfCreate() 
        maxRow = self.getMaxRow()
        # make subplots of individual wells
        colorsList = px.colors.qualitative.Set3
        figSub = make_subplots(
            rows=8, cols=12,
            shared_xaxes=True,
            shared_yaxes='all',
            start_cell='top-left',
            subplot_titles=list(df.columns)) #, subplot_titles=sub_titles, print_grid=False)
        for r in range (0,8): #loop through rows
            for c in range(0,12):
                figSub.add_trace(go.Scatter(
                    x=df.index,
                    y=df.iloc[:,r+(r*11)+c], # this pattern translates r=8, c=12 to 0..95 columns in df (0 indexed)
                    name=df.columns[r+(r*11)+c], # gets column name eg 'A01'
                    mode="lines+text", # allows for text to be inserted onto ind plot
                    line_color=colorsList[c], # assign color to each well consistent with all-well plot
                    text= ['']*32 + [maxRow[r+(r*11)+c]]+['']*7, # inserts max value onto plot at 33rd point; looks nice here
                    textposition='top left',
                    textfont=dict(
                        family="sans serif",
                        size=8,
                        color="black")), # want to stand out
                    row=r+1,
                    col=c+1)
        figSub.update_layout(
            height=800,
            width=1000,
            showlegend=False,
            title_text='Cycle vs Fluorescence {} ({})'.format(self.getTargetName(), self.probe))
        return figSub.show()
    
class PrimerMatrixVis:
    def __init__(self, probe, dfExcel):
        self.probe = probe
        self.dfExcel = dfExcel
    def dfCreate(self): # make a df from the excel data
        # make df from passed sheet name
        df = self.dfExcel['Quan. Result']
        # assign first row to be column header
        df.columns = df.iloc[0]
        # drop the first row (duplicate of header)
        df = df.drop(df.index[0])
        # drop columns 'Ct Mean' and 'Ct SD' and 'Concentration' and 'Aver. Conc.' and 'Conc. SD'
        df = df.drop(['Ct Mean', 'Ct SD', 'Concentration', 'Aver. Con.', 'Con. SD'], axis=1)
        # keep rows where 'Dye' column =  probe. Drop other rows
        dfProbe = df[df['Dye'] == self.probe]
        # reindex dfProbe beginning at 1
        dfProbe.index = np.arange(1, len(dfProbe)+1)
        return dfProbe
    
    def getTargetName(self):
        # get quan result sheet from dfExcel workbook
        quanResult = self.dfExcel['Quan. Result']
        # make the first row column headers
        new_header = quanResult.iloc[0] #grab the first row for the header
        quanResult = quanResult[1:] #take the data less the header row
        quanResult.columns = new_header #set the header row as the df header
        # find the value in "Target" column when "Dye" = "FAM"
        targetName = quanResult.loc[quanResult['Dye'] == self.probe, 'Target'].iloc[0]
        return targetName
    
    def addHeatMap(self):
        df = self.dfCreate()
        plateMatrix = []
        for r in range (0,8): #loop through rows
            colList = []
            for c in range(0,12):
                # add Ct values to list                
                colList.append(df['Ct'].iloc[r+(r*11)+c])
            plateMatrix.append(colList)
        # print(plateMatrix)
        # substitute nan for 0
        plateMatrix = np.nan_to_num(plateMatrix)
        # for list in plateMatrix:
        #     print(list)           
        # reverse order of rows because heatmap put rows g and h at top
        plateMatrix = plateMatrix[::-1]
        heatmap = go.Figure(data=go.Heatmap(
            z=plateMatrix,
            zmin=15,
            zmax=40))
        colText = ['1\n50nM','2\n50nM','3\n100nM','4\n100nM','5\n200nM','6\n200nM','7\n400nM','8\n400nM','9\n600nM','10\n600nM','11\n800nM','12\n800nM']
        heatmap.update_layout(
            height=500,
            width=800,
            title={'text': 'Heatmap of Ct in Plate {} ({})'.format(self.getTargetName(), self.probe),
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
            },
            # title='Heatmap of Ct in Plate {} ({})'.format(self.sheet, self.probe),
            xaxis=dict(title='', tickvals=[0,1,2,3,4,5,6,7,8,9,10,11], ticktext=colText, side='top'),
            yaxis=dict(title='ROWS', tickmode='array', tickvals=[0, 1, 2, 3, 4, 5, 6, 7], ticktext=['H std curve', 'G std curve', '800nM F', '600nM E', '400nM D', '200nM C', '100nM B', '50nM A'])
        )
        return heatmap.show()
    
    def addPrimerConcToDf(self):
        df = self.dfCreate()
        primerConcs = [50, 100, 200, 400, 600, 800] # Fwd and rev primer concentrations
        # add fwd primerConc values to two rows all the way to 72
        df['F_primer'] =np.nan
        for i in range(6): # loop through rows except last two rows containing standard curve
            for j in range(6): # loop through primer concentration list
                for k in range(2): # every concentration twice
                    pos = (12*i)+(2*j)+(k+1) # translate row and column to position in df 1..96
                    df.loc[pos, 'F_primer'] = primerConcs[j] # set value to primer concentration
        # add rev primerConc values across single row for 6 rows
        df['R_primer'] =np.nan
        for m in range(6): # loop through primer concentration list
            for n in range(12): # every concentration as a row
                pos = (12*m)+(n+1) # translate row and column to position in df 1..96
                df.loc[pos, 'R_primer'] = primerConcs[m] # set value to primer concentration
        return df 
    
    def Periodicity(self):
        df=self.addPrimerConcToDf()
        # group by F and R primer concentrations and take mean
        avg_across_row_fwd = df.groupby(by=['R_primer', 'F_primer'], as_index=False)['Ct'].mean()
        avg_across_col_rev = df.groupby(by=['F_primer', 'R_primer'], as_index=False)['Ct'].mean()
        # stringify 'F_primer' and 'R_primer' concentrations and concatenate and add to list
        barHeadingsFwd = (avg_across_row_fwd['F_primer'].astype(str) + '_' + avg_across_row_fwd['R_primer'].astype(str)).tolist()
        barHeadingsRev = (avg_across_col_rev['R_primer'].astype(str) + '_' + avg_across_col_rev['F_primer'].astype(str)).tolist()
        figAvgFwd = px.bar(
            avg_across_row_fwd,
            y=avg_across_row_fwd['Ct'],
            x=barHeadingsFwd, #avg_across_row['F_primer'],
            labels={'x':'F Primer Concentration', 'y':'Cq'},
            color='R_primer',
            barmode='group',
            text_auto=True,
            title='Periodicity in Increasing [Fwd] (Going Across Row)<br>[Rev] Held Constant, Two Points per Condition Averaged')
        figAvgFwd.update_layout(width=800, height=500)
        figAvgFwd.show()
        figAvgRev = px.bar(
            avg_across_col_rev,
            y=avg_across_col_rev['Ct'],
            x=barHeadingsRev, #avg_across_row['F_primer'],
            labels={'x':'R Primer Concentration', 'y':'Cq'},
            color='F_primer',
            barmode='group',
            text_auto=True,
            title='Periodicity in Increasing [Rev] (Going Down Column)<br>[Fwd] Held Constant, Two Points per Condition Averaged')
        figAvgRev.update_layout(width=800, height=500)
        return figAvgRev.show()
    
    def onePhaseDecay(self, X, y0, Plateau, K):
        return (y0-Plateau)*np.exp(-K*X)+Plateau
    
    def RowAndColMeans(self):  
        df=self.addPrimerConcToDf()  
        row_mean = df.groupby(by='R_primer', as_index=False)['Ct'].mean() # group by R primer concentration and take mean
        col_mean = df.groupby(by='F_primer', as_index=False)['Ct'].mean() # group by F primer concentration and take mean
        row_conc = row_mean['R_primer'].tolist() # make list of concentrations
        row_cq_mean = row_mean['Ct'].tolist() # make list of Cqs
        col_conc = col_mean['F_primer'].tolist() 
        col_cq_mean = col_mean['Ct'].tolist()
        # solve for one phase decay with scipy
        # initialGuesses = [y0, Plateau, K] 
        # Y0 is the Y value when X (Conc) is zero. It is expressed in the same units as Y,
        # Plateau is the Y value at infinite conc, expressed in the same units as Y.
        # K is the rate constant, expressed in reciprocal of the X axis conc units. If X is in nM, then K is expressed in inverse nM.
        initialGuesses = [100, 15, 0.01] # converges faster with initial guesses
        ropt,rcov = curve_fit(self.onePhaseDecay, row_conc, row_cq_mean, initialGuesses, maxfev=10000)
        copt,ccov = curve_fit(self.onePhaseDecay, col_conc, col_cq_mean, initialGuesses, maxfev=10000)
        row_cq_pred = np.empty(len(row_conc)) #empty list to receive data
        for i in range(len(row_conc)): #loop through concentrations to make cq prediction
            # cq_pred[i]=onePhaseDecay(f_r_conc[i], initialGuesses[0], initialGuesses[1], initialGuesses[2])
            row_cq_pred[i]=self.onePhaseDecay(row_conc[i], ropt[0], ropt[1], ropt[2])
        col_cq_pred = np.empty(len(col_conc)) #empty list to receive data
        for i in range(len(col_conc)): #loop through concentrations to make cq prediction
            # cq_pred[i]=onePhaseDecay(f_r_conc[i], initialGuesses[0], initialGuesses[1], initialGuesses[2])
            col_cq_pred[i]=self.onePhaseDecay(col_conc[i], copt[0], copt[1], copt[2])
        # from sklearn.metrics import r2_score
        row_R2score = r2_score(row_cq_pred, row_cq_mean)
        col_R2score = r2_score(col_cq_pred, col_cq_mean)
        # Generate figure
        figRowVsColMean = go.Figure()
        figRowVsColMean.add_trace(
            go.Scatter(
                x=row_conc,
                y=row_cq_mean,
                name='Row Mean', 
                mode='markers',
                marker_color='#2038A8'
                # line=dict(color='rgb(255, 0, 0)', width=2),
                # line_shape = 'spline' # make smooth line
            ))
        figRowVsColMean.add_trace(
            go.Scatter(
                x=row_conc,
                y=row_cq_pred,
                name='Row (Modeled)',
                mode='lines',
                line=dict(color='#7564F5', width=1),
                line_shape = 'spline' # make smooth line
            ))
        figRowVsColMean.add_trace(
            go.Scatter(
                x=col_conc,
                y=col_cq_mean,
                name='Col Mean',
                mode='markers',
                marker_color='#F58E17'
            ))
        figRowVsColMean.add_trace(
            go.Scatter(
                x=col_conc,
                y=col_cq_pred,
                name='Col (Modeled)',
                mode='lines',
                # marker_color='#F5BD69', 
                line=dict(color='#F5BD69', width=1),
                line_shape = 'spline' # make smooth line
            ))
        figRowVsColMean.update_layout(
        height=500,
        width=800,
        title='Row vs Col Means',
        xaxis_title='Mean',
        yaxis_title='Cq')
        return figRowVsColMean.show()
    
    def FwdAndRevEquivalence(self):
        df=self.addPrimerConcToDf()  
        # create a new df where 'F_primer' = 'R_primer' 
        df_fwd_rev = df[df['F_primer'] == df['R_primer']]
        df_fwd_rev_mean = df_fwd_rev.groupby(by=['R_primer', 'F_primer'], as_index=False)['Ct'].mean().dropna() # take mean, remove NaN
        # equivHeadings = (df_fwd_rev_mean['F_primer'].astype(str) + '_' + df_fwd_rev_mean['R_primer'].astype(str)).tolist()
        # solve for one phase decay with scipy
        # f_r_conc = [100, 200, 400, 600, 800] # F-R primer concentrations
        # cq_actual = [37.32, 25.21, 21.27, 20.225, 18.81] # Cq values
        f_r_conc = df_fwd_rev_mean['R_primer'].tolist() # both F_primer and R_primer are the same
        cq_actual = df_fwd_rev_mean['Ct'].tolist()
        # found each element in cq_actual to two decimal places
        cq_actual = [round(x, 2) for x in cq_actual]
        # initialGuesses = [y0, Plateau, K] 
        # Y0 is the Y value when X (Conc) is zero. It is expressed in the same units as Y,
        # Plateau is the Y value at infinite conc, expressed in the same units as Y.
        # K is the rate constant, expressed in reciprocal of the X axis conc units. If X is in nM, then K is expressed in inverse nM.
        # initialGuesses = [73.29, 19.8, 0.01120] # GraphPad values
        initialGuesses = [100, 15, 0.01]
        popt,pcov = curve_fit(self.onePhaseDecay, f_r_conc, cq_actual, initialGuesses, maxfev=10000)
        cq_pred = np.empty(len(cq_actual)) #empty list to receive data
        for i in range(len(f_r_conc)): #loop through concentrations to make cq prediction
            # cq_pred[i]=onePhaseDecay(f_r_conc[i], initialGuesses[0], initialGuesses[1], initialGuesses[2])
            cq_pred[i]=self.onePhaseDecay(f_r_conc[i], popt[0], popt[1], popt[2])
        # from sklearn.metrics import r2_score
        equivR2score = r2_score(cq_pred, cq_actual)

        # Plot 3 more predicted points @ 300, 500, 700nM
        cq_pred_300 = self.onePhaseDecay(300, popt[0], popt[1], popt[2])
        cq_pred_500 = self.onePhaseDecay(500, popt[0], popt[1], popt[2])
        cq_pred_700 = self.onePhaseDecay(700, popt[0], popt[1], popt[2])
        figEquivalent = go.Figure()
        figEquivalent.add_trace(
            go.Scatter(
                x=f_r_conc,
                y=cq_pred,
                name='Cq Predicted (R^2 = ' + str(round(equivR2score, 4)) + ')', 
                mode='lines+markers',
                line=dict(color='rgb(255, 0, 0)', width=2),
                line_shape = 'spline' # make smooth line
            ))
        figEquivalent.add_trace(
            go.Bar(
                x=f_r_conc,
                y=cq_actual,
                text = cq_actual,
                name='Cq Actual',
                textposition='inside',
                marker_color='#636EFA'
            ))
        figEquivalent.add_trace(
            go.Bar(
                x=[300, 500, 700],
                y=[cq_pred_300, cq_pred_500, cq_pred_700],
                name='Cq Predicted @300, 500, 700nM',
                textposition='inside',
                marker_color='#94B6FA'
            ))
        figEquivalent.update_layout(
        height=500,
        width=800,
        title='<b>Cq when [Fwd] = [Rev]</b><br>Actual vs Predicted (One Phase Decay)<br>Two Points per Condition Averaged',
        xaxis_title='[F=R] Primer Concentration',
        yaxis_title='Cq')
        return figEquivalent.show()
    
    def plotStandards(self, concList=[]):
        # a list of concentrations provided by the user in form of [1e10, 1e9...] 7 elements
        self.concList = concList 
        # if concList is empty, use default values
        if len(self.concList) == 0:
            # self.concList = [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7]
            self.concList = [1e10,1e9,1e8,1e7,1e6,1e5,1e4] # don't include 8 because neg; plot separately
        # triplicate each element in concList
        tripConcList = [np.log10(x) for x in self.concList for i in range(3)] # triplicate Cts for each concentration
        # rename "Property" column with standard and Negative
        df = self.dfCreate()
        df.loc[73:93, "Property"]= "Standard" # make property in rows 73-93 "Standard"
        df.loc[94:96, "Property"]= "Negative" 
        df.loc[73:93, 'CopiesPerRxn'] = tripConcList
        # df.loc[73:93, "CopiesPerRxn"]= tripConcList
        standardNumberList = ['Std_1', 'Std_2', 'Std_3', 'Std_4', 'Std_5', 'Std_6', 'Std_7', 'Negative']
        # create a new column with the standard number e.g. std_1, std_2, etc.
        df['StandardNumber'] =np.nan # create new column for standard number
        for j in range(8): # 8 sets of triplicates on two rows
            for k in range(3): # every concentration 3x for std curve
                pos = 73+k+(j*3) # translate row and column to position in df 1..96
                df.loc[pos, 'StandardNumber'] = standardNumberList[j] # set value to primer concentration
        # select df with only standards; don't want negatives
        standards = df.loc[df['Property'] == 'Standard']
        # drop rows where standards['Ct'] is NaN
        standards = standards.dropna(subset=['Ct'])
        standardNumberMeans = standards.groupby(by=['CopiesPerRxn'], as_index=False)['Ct'].mean().dropna() # take mean, remove NaN
        # print (standardNumberMeans)
        # calculate the regression line
        m, b = np.polyfit(standardNumberMeans['CopiesPerRxn'].values, standardNumberMeans['Ct'].values ,1 ) #want single value for each standard
        regression_line = m * standardNumberMeans['CopiesPerRxn'].values + b
        # calculate PCR efficiency
        pcrEfficiency = (10**(-1/m) - 1)*100
        # add pcrEfficiency to plot legend
        # create the scatter plot with the regression line
        figStandards = go.Figure()
        figStandards.add_trace(go.Scatter(x=standards['CopiesPerRxn'], y=standards['Ct'], mode='markers', name='Standards'))
        figStandards.add_trace(go.Scatter(x=standardNumberMeans['CopiesPerRxn'].values, y=regression_line, mode='lines', name='Regression Line'))
        figStandards.update_layout(title='Standard Curve for {}. PCR Efficiency {:.3f} %'.format(self.probe, pcrEfficiency), xaxis=dict(title='Log10 (CopiesPerReaction)'), yaxis=dict(title='Ct Value'))
        # return figStandards.show()
        barx = standardNumberMeans['CopiesPerRxn'].values
        bary = standardNumberMeans['Ct'].values
        # create the bar graph trace
        bartrace = go.Bar(x=barx, y=bary, marker=dict(color='#94B6FA'))
        # add the trace
        figStandards.add_trace(bartrace)        
        figStandards.update_layout(width=800, height=500)
        return figStandards.show()
                        



class ProbeMatrixVis():
    probe: str
    dfExcel: pd.DataFrame
    def __init__(self, probe, dfExcel):
        self.probe = probe
        self.dfExcel = dfExcel
    def dfCreate(self): # make a df from the excel data
        # make df from passed sheet name
        df = self.dfExcel['Quan. Result']
        # assign first row to be column header
        df.columns = df.iloc[0]
        # drop the first row (duplicate of header)
        df = df.drop(df.index[0])
        # drop columns 'Ct Mean' and 'Ct SD' and 'Concentration' and 'Aver. Conc.' and 'Conc. SD'
        df = df.drop(['Ct Mean', 'Ct SD', 'Concentration', 'Aver. Con.', 'Con. SD'], axis=1)
        # keep rows where 'Dye' column =  probe. Drop other rows
        dfProbe = df[df['Dye'] == self.probe]
        # reindex dfProbe beginning at 1
        dfProbe.index = np.arange(1, len(dfProbe)+1)
        return dfProbe
    
    def getTargetName(self):
        # get quan result sheet from dfExcel workbook
        quanResult = self.dfExcel['Quan. Result']
        # make the first row column headers
        new_header = quanResult.iloc[0] #grab the first row for the header
        quanResult = quanResult[1:] #take the data less the header row
        quanResult.columns = new_header #set the header row as the df header
        # find the value in "Target" column when "Dye" = "FAM"
        targetName = quanResult.loc[quanResult['Dye'] == self.probe, 'Target'].iloc[0]
        return targetName
    

    # def addProbePropertiesToDf(self): # make a df from the excel data
    #     df = self.dfCreateFromQuanResult()
    #     df.loc[72:92, "Property"]= "Standard" # make property in rows 73-93 "Standard"
    #     df.loc[93:95, "Property"]= "Negative" 
    #     standardNumberList = ['Std_1', 'Std_2', 'Std_3', 'Std_4', 'Std_5', 'Std_6', 'Std_7', 'Negative']
    #     # create a new column with the standard number e.g. std_1, std_2, etc.
    #     df['StandardNumber'] =np.nan # create new column for standard number
    #     for j in range(8): # 8 sets of triplicates on two rows
    #         for k in range(3): # every concentration 3x for std curve
    #             pos = 72+k+(j*3) # translate row and column to position in df 1..96
    #             df.loc[pos, 'StandardNumber'] = standardNumberList[j] # set value to primer concentration
    #     return df
    # def addPrimerAndProbeConcToDf(self, primerConc):
    #     primerConc: list
    #     df = self.addProbePropertiesToDf()
    #     probeConcs = [50, 100, 200, 400, 600, 800] # Fwd and rev primer concentrations
    #     if not primerConc: # if list is empty use default values
    #         primerConc = [200, 300, 400]
    #     # add probe and primerConc values to two rows all the way to 72
    #     df['Probe_Conc'] =np.nan
    #     for i in range(6): # loop through rows except last two rows containing standard curve
    #         for j in range(3): # loop through primer concentration list
    #             for k in range(4): # every concentration 4x
    #                 pos = (12*i)+(4*j)+(k) # translate row and column to position in df 1..96
    #                 df.loc[pos, 'Probe_Conc'] = probeConcs[i] # set value to primer concentration
    #     df['F_primer'] =np.nan
    #     for i in range(6): # loop through rows except last two rows containing standard curve
    #         for j in range(3): # loop through primer concentration list
    #             for k in range(4): # every concentration 4x
    #                 pos = (12*i)+(4*j)+(k) # translate row and column to position in df 1..96
    #                 df.loc[pos, ['F_primer', 'R_primer']] = primerConc[j] # set value to primer concentration
    #     return df
#     def plotProbeStandards(self, primerConc):
#         # df = self.dfCreate()
#         df = self.addPrimerAndProbeConcToDf(primerConc)
#         standards = df.loc[df['Property'] == 'Standard']
#         standardNumberMeans = standards.groupby(by=['StandardNumber'], as_index=False)['Ct'].mean().dropna()
#         figStandards = px.bar(
#             standardNumberMeans,
#             y=standardNumberMeans['Ct'],
#             x=standardNumberMeans['StandardNumber'], #avg_across_row['F_primer'],
#             labels={'x':'Standard', 'y':'Cq'},
#             # color='R_primer',
#             barmode='group',
#             text_auto=True,
#             title='Standard Number vs Cq' )
#         figStandards.show()
#     def ProbePeriodicity(self, primerConc):
#         primerConc: list
#         df = self.addPrimerAndProbeConcToDf(primerConc)
#         # group by Probe and F,R primer concentrations and take mean
#         avg_across_row_fwd_rev = df.groupby(by=['Probe_Conc', 'F_primer', 'R_primer'], as_index=False)['Ct'].mean()
#         # avg_across_col_probe = df.groupby(by=['Probe_Conc', 'F_primer', 'R_primer'], as_index=False)['Ct'].mean()
#         # # round column Ct to nearest two digits in avg_across_row_fwd_rev
#         # stringify 'F_primer','R_primer' and Probe concentrations and concatenate and add to list
#         barHeadings_row = (avg_across_row_fwd_rev['F_primer'].astype(str) +
#             '_' + avg_across_row_fwd_rev['R_primer'].astype(str) + 
#             '_' + avg_across_row_fwd_rev['Probe_Conc'].astype(str)).tolist()
#         # barHeadings_col = (avg_across_row_fwd_rev['F_primer'].astype(str) +
#         #     '_' + avg_across_row_fwd_rev['R_primer'].astype(str) + 
#         #     '_' + avg_across_row_fwd_rev['Probe_Conc'].astype(str)).tolist()
#         figAvgProbe = px.bar(
#             avg_across_row_fwd_rev,
#             y=avg_across_row_fwd_rev['Ct'],
#             x=barHeadings_row, #avg_across_row['F_primer'],
#             labels={'x':'[Fwd_Rev_Probe] Concentration', 'y':'Cq'},
#             color='Probe_Conc',
#             barmode='group',
#             text_auto=True,
#             title='Periodicity in Increasing [Fwd, Rev] (Going Across Row)<br>[Probe] Held Constant, Four Points per Condition Averaged')
#         return figAvgProbe.show()

# class LOD(FluorVis): # calculates LOD for a plate with each conc on a row
#     # probe: str
#     # fileDir: str
#     # fileName: str
#     def __init__(self, probe, fileDir, fileName):
#         super().__init__(probe, fileDir, fileName)
    