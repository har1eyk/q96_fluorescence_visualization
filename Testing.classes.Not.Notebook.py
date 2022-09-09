import FluorVis as fv

import sys
# print(sys.executable)

# assign file directory and name for Excel file in Windows
fileDir = r'C:\Users\HarleyKing\LuminUltra Technologies Ltd\LTI Research & Development - Documents\Automation and Assay Dev\Exp800.24_vibrio_cholerae\Exp800.24.04_probe_matrix'
fileName = r'20220905_085250_Exp800.24_Vch_probe_matrix.xls' #fam data

# instantiate probes
# probeCy5 = fv.FluorVis('CY5', fileDir, fileName)
probeFAM = fv.ProbeMatrixVis('FAM', fileDir, fileName)
# probeFAM = FluorVis('FAM', 'sheetNames')
# probeHEX = FluorVis('HEX')
# probeSYBR = FluorVis('SYBR')
# display (probeFAM)
# probeFAM.getSheetNames()
# probeFAM.plotAllWells()
# probeFAM.plotIndWells()
# probeFAM= fv.ProbeMatrixVis('FAM', fileDir, fileName)
# probeFAM.addProbePropertiesToDf()
# print (probeFAM.addProbePropertiesToDf().tail(50))
# print (probeFAM.addPrimerAndProbeConcToDf([500,400,300]).head(50))
# print (probeFAM.plotProbeStandards([500,400,300]).head(50))
print (probeFAM.ProbePeriodicity([500,400,300]))

# fv.ProbeMatrixVis.ProbePeriodicity(probeFAM)
# fv.FluorVis.dfCreate(probeFAM)
# fv.FluorVis.dfCreate(probeCy5)
# fv.FluorVis.plotIndWells(probeFAM)
# fv.FluorVis.plotAllWells(probeCy5)
# fv.FluorVis.plotIndWells(probeCy5)