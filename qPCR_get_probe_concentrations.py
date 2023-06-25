import numpy as np
import pandas as pd
pd.options.plotting.backend = "plotly"
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
# from getAmpViz import PrimerMatrixVis
from getAmpViz import ProbeMatrixVis
from getAmpViz import FluorVis
from tkinter import Tk, filedialog


# root = Tk()
# root.withdraw()

# file_path = filedialog.askopenfilename()

# print(file_path)

# open file
file_path = r'C:\Users\HarleyKing\OneDrive - LuminUltra Technologies Ltd\Documents - LTI Research & Development\Automation and Assay Dev\Exp800.31 IAC in Ecoli Assay\Exp800.31.02 probe matrix\20230608_130048_E.coli_hi_probe_matrix.xls'
# file_path = r"C:\Users\HarleyKing\OneDrive - LuminUltra Technologies Ltd\Documents - LTI Research & Development\Automation and Assay Dev\Exp800.31 IAC in Ecoli Assay\Exp800.31.02 probe matrix\20230608_142840_Ecoli_IAC_probe_matrix_p2.xls"
# # create dataframe
dfExcel = pd.read_excel(file_path, sheet_name=None, header=None) #get all sheets

# probeCy5 = ProbeMatrixVis('CY5', dfExcel)


# probeCy5 = FluorVis('CY5', dfExcel)
# probeFAM = FluorVis('FAM', dfExcel)
# FluorVis.plotIndWells(probeCy5)
# FluorVis.plotAllWells(probeCy5)
# FluorVis.plotIndWells(probeFAM)
# FluorVis.plotAllWells(probeFAM)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

probeCy5 = ProbeMatrixVis('CY5', dfExcel)
# target = ProbeMatrixVis.getTargetName(probeCy5)
# df = ProbeMatrixVis.addProbePropertiesToDf(probeCy5)
# df = ProbeMatrixVis.dfCreate(probeCy5)
# df = ProbeMatrixVis.ProbePeriodicity(probeCy5)
df = ProbeMatrixVis.addProbeHeatMap(probeCy5)
# print (df)

# primerMatrix = PrimerMatrixVis('CY5', dfExcel)
# df = PrimerMatrixVis.dfCreate(primerMatrix)
# df = PrimerMatrixVis.addPrimerConcToDf(primerMatrix)
# PrimerMatrixVis.RowAndColMeans(primerMatrix)
# PrimerMatrixVis.addHeatMap(primerMatrix)
# PrimerMatrixVis.Periodicity(primerMatrix)
# PrimerMatrixVis.FwdAndRevEquivalence(primerMatrix)
# PrimerMatrixVis.plotStandards(primerMatrix)