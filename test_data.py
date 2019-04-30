#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 22:37:33 2019

@author: akshay
"""

import pandas as pd

def prepared_test_data(model_data):
   
   """
   we will remove selected columns based on intution here
   we will transform the column data between 0 and 1 here
   """
   ## column names those should not matter much
   """
   column Name                Column number
   1. UniqueID                   0
   2. branch_id                  4
   3. supplier_id                5
   4. Current_pincode_ID         7
   5. State_ID                   11
   6. Employee_code_ID           12
   7. MobileNo_Avl_Flag          13
   8. AAdhar_flag                14
   9. VoterID_flag               16
   10. Passport_flag              18
   11. SEC.NO.OF.ACCTS            27            
   12. SEC.ACTIVE.ACCTS           28            
   13. SEC.OVERDUE.ACCTS          29            
   14. SEC.CURRENT.BALANCE        30            
   15. SEC.SANCTIONED.AMOUNT      31            
   16. SEC.DISBURSED.AMOUNT       32 
   17.PERFORM_CNS.SCORE.DESCRIPTION 20 
   18. manufacturer_id            6
   
   """

   
   ## columns to be normalized 
   """
   column name                   column number
   1. disbursed_amount           0
   2. asset_cost                 1
   3. ltv                        2
   4. PERFORM_CNS.SCORE          9
   5. PRI.CURRENT.BALANCE        14
   6. PRI.SANCTIONED.AMOUNT      15
   7. PRI.DISBURSED.AMOUNT       16
   8. PRIMARY.INSTAL.AMT         17
   9. SEC.INSTAL.AMT             18
   
   """
   new_data = model_data.dropna()
   new_data.isnull().sum()
   data_list = new_data.iloc[:,[1,2,3,8,9,10,15,17,19,21,22,23,24,25,26,33,34,35,36,37,38,39]]
   
   data_list.columns
   ### normalizing columns
   data_list['disbursed_amount'] = data_list['disbursed_amount'] / data_list['disbursed_amount'].max()
   data_list['asset_cost'] = data_list['asset_cost'] / data_list['asset_cost'].max()
   data_list['ltv'] = data_list['ltv'] / data_list['ltv'].max()
   data_list['PERFORM_CNS.SCORE'] = data_list['PERFORM_CNS.SCORE'] / data_list['PERFORM_CNS.SCORE'].max()
   data_list['PRI.CURRENT.BALANCE'] = data_list['PRI.CURRENT.BALANCE'] / data_list['PRI.CURRENT.BALANCE'].max()
   data_list['PRI.SANCTIONED.AMOUNT'] = data_list['PRI.SANCTIONED.AMOUNT'] / data_list['PRI.SANCTIONED.AMOUNT'].max()
   data_list['PRI.DISBURSED.AMOUNT'] = data_list['PRI.DISBURSED.AMOUNT'] / data_list['PRI.DISBURSED.AMOUNT'].max()
   data_list['PRIMARY.INSTAL.AMT'] = data_list['PRIMARY.INSTAL.AMT'] / data_list['PRIMARY.INSTAL.AMT'].max()
   data_list['SEC.INSTAL.AMT'] = data_list['SEC.INSTAL.AMT'] / data_list['SEC.INSTAL.AMT'].max()
   
   # transforming Date of birth and loan distribution date into one column
   DOB = list(data_list['Date.of.Birth'])
   DOB_year = []
   for i in DOB:
      if int(i[-2:]) < 20:
         DOB_year.append(str(20) + i[-2:])
      else:
         DOB_year.append(str(19) + i[-2:])
   LOAN = list(data_list['DisbursalDate'])
   LOAN_year = []
   for i in LOAN:
      if int(i[-2:]) < 20:
         LOAN_year.append(str(20) + i[-2:])
      else:
         LOAN_year.append(str(19) + i[-2:])
   age_loan_year = []
   for i,j in zip(DOB_year,LOAN_year):
      age_loan_year.append(int(j)-int(i))
   data_list['age_loan_year'] = age_loan_year
   data_list['age_loan_year'] = data_list['age_loan_year'] / data_list['age_loan_year'].max()
   
   ## Creating Employee type flag
   emp_type = list(data_list['Employment.Type'])
   # 0 for salaried and 1 for self employed
   emp_flag = []
   for i in emp_type:
      if i == 'Salaried':
         emp_flag.append(0)
      else:
         emp_flag.append(1)
   data_list['Employment.Type'] = emp_flag
   avg_acct_age = list(data_list['AVERAGE.ACCT.AGE'])
   avg_acct = []
   for i in avg_acct_age:
      str_split = i.split(' ')
      avg_acct.append(float(str_split[0].strip('yrs')) + float(str_split[1].strip('mon'))/12)
   data_list['AVERAGE.ACCT.AGE'] = avg_acct
   data_list['AVERAGE.ACCT.AGE'] = data_list['AVERAGE.ACCT.AGE'] / data_list['AVERAGE.ACCT.AGE'].max()
   
   credit_history_len = data_list['CREDIT.HISTORY.LENGTH']
   credit_his = []
   for i in credit_history_len:
      str_split = i.split(' ')
      credit_his.append(float(str_split[0].strip('yrs')) + float(str_split[1].strip('mon'))/12)
   data_list['CREDIT.HISTORY.LENGTH'] = credit_his
   data_list['CREDIT.HISTORY.LENGTH'] = data_list['CREDIT.HISTORY.LENGTH'] / data_list['CREDIT.HISTORY.LENGTH'].max()

   ## we shall return the final data_list which should not contain 
   ## DOB and DOD
   data_list.drop(["Date.of.Birth","DisbursalDate"],axis=1,inplace=True)
   
   
   
   return data_list

test_data = pd.read_csv("test_bqCt9Pv.csv")
lll = list(test_data.columns)
proc_test_data = prepared_test_data(test_data)
