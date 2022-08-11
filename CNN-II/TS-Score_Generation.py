import os
import xlrd
import xlwt


base_path = './testing_inffer'

dirs = os.listdir(base_path)
indicators = []
for d in dirs:
    bp = os.path.join(base_path, d)
    preds = os.listdir(bp)
    pcr_prob = 0.0
    for pf in preds:
        pfs = pf.split('_')
        pcrp = pfs[1]
        pcr_prob += float(pcrp)
    pcr_prob /= len(preds)
    indicators.append(pcr_prob)
    
  
workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('TS-Score')


for j in range(len(dirs)):
    worksheet.write(j,0,dirs[j])
    worksheet.write(j,1,indicators[j])

workbook.save('TS-Score.xls')



