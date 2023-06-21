import numpy as np
import pandas as pd
pd.options.plotting.backend = "plotly"
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from getAmpViz import PrimerMatrixVis
from tkinter import Tk, filedialog


root = Tk()
root.withdraw()

file_path = filedialog.askopenfilename()

print(file_path)
# # create dataframe
dfExcel = pd.read_excel(file_path, sheet_name=None, header=None) #get all sheets


# probeCy5 = FluorVis('CY5', dfExcel)
# probeFAM = FluorVis('FAM', dfExcel)
# FluorVis.plotIndWells(probeCy5)
# FluorVis.plotAllWells(probeCy5)
# FluorVis.plotIndWells(probeFAM)
# FluorVis.plotAllWells(probeFAM)

primerMatrix = PrimerMatrixVis('CY5', dfExcel)
# df = PrimerMatrixVis.dfCreate(primerMatrix)
# df = PrimerMatrixVis.addPrimerConcToDf(primerMatrix)
PrimerMatrixVis.RowAndColMeans(primerMatrix)
PrimerMatrixVis.addHeatMap(primerMatrix)
PrimerMatrixVis.Periodicity(primerMatrix)
PrimerMatrixVis.FwdAndRevEquivalence(primerMatrix)
PrimerMatrixVis.plotStandards(primerMatrix)